"""Generate a place-into-cabinet task: an open-front cabinet on the table, the
task object starts on the table and is placed onto a randomly cluttered shelf.

Writes dextoolbench/trajectories/<category>/<object>/<task_name>.json with:
  start_pose  object start on the table (beside or in front of the cabinet)
  cabinet     cabinet geometry (dextoolbench/cabinet.py)
  table_urdf  the table it stands on (table_narrow extended 5 cm toward the robot)
  object      the object's resting geometry (rest quat, origin height, AABB)
  clutter     clutter/ghost-goal sampling knobs (dextoolbench/clutter.py)
  goals       the 4 waypoints of one example layout (seed --layout_seed, env 0):
              lift -> pre-insert (outside the opening) -> over the goal -> place,
              carried just above the tallest clutter in the front corridor.
The goal itself is resampled per episode from the clutter block;
visualize_task.py --resample-clutter draws fresh layouts.

Usage:
  python3.12 dextoolbench/generate_cabinet_task.py [--object_category eraser --object_name flat_eraser]
      [--task_name cabinet_place] [--cabinet_pos 0.01 -0.05] [--cabinet_size 0.40 0.30 0.60]
      [--num_shelves 1] [--num_slots 10] [--count_range 4 10] [--front_count_range 0 2]
      [--goal_levels 0 1] [--clutter_all_levels --clutter_levels 0 1]
"""

import argparse
import json
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dextoolbench.cabinet import shelf_surfaces  # noqa: E402
from dextoolbench.clutter import (  # noqa: E402
    clutter_spec,
    env_items,
    front_height_cap,
    sample_clutter_pool,
    sample_layout,
)

TABLE_Z = 0.38  # table actor centre (tableResetZ)
TABLE_TOP_Z = TABLE_Z + 0.15  # table box is 0.3 m tall
TABLE_URDF = "assets/urdf/table_narrow_long.urdf"  # table_narrow extended 5 cm toward the robot
TABLE_X_HALF = 0.2375
TABLE_Y_RANGE = (-0.20, 0.25)


def quat_to_matrix(qx, qy, qz, qw):
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]


def object_mesh_vertices(object_category: str, object_name: str) -> list[list[float]]:
    """Scaled vertices of the object's (single) .obj collision mesh."""
    urdf = REPO_ROOT / "assets/urdf/dextoolbench" / object_category / object_name / f"{object_name}.urdf"
    mesh = ET.parse(urdf).getroot().find(".//collision/geometry/mesh")
    scale = [float(s) for s in (mesh.get("scale") or "1 1 1").split()]
    obj = urdf.parent / mesh.get("filename")
    return [
        [float(c) * s for c, s in zip(line.split()[1:4], scale)]
        for line in obj.read_text().splitlines()
        if line.startswith("v ")
    ]


def world_extents(verts, quat_xyzw):
    """Per-axis (min, max) of the mesh rotated to quat, relative to its origin."""
    R = quat_to_matrix(*quat_xyzw)
    rotated = [[sum(R[i][j] * v[j] for j in range(3)) for i in range(3)] for v in verts]
    return [(min(p[i] for p in rotated), max(p[i] for p in rotated)) for i in range(3)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--object_category", default="eraser")
    parser.add_argument("--object_name", default="flat_eraser")
    parser.add_argument("--task_name", default="cabinet_place")
    parser.add_argument("--rest_pose_from", default="wipe_smile",
                        help="Existing task of this object whose start_pose gives the resting z/orientation.")
    parser.add_argument("--cabinet_pos", type=float, nargs=2, default=[0.01, -0.05],
                        help="Cabinet centre x y (env-local); it stands on the table top.")
    parser.add_argument("--cabinet_yaw_deg", type=float, default=0.0)
    parser.add_argument("--cabinet_size", type=float, nargs=3, default=[0.40, 0.30, 0.60])
    parser.add_argument("--wall_thickness", type=float, default=0.015)
    parser.add_argument("--num_shelves", type=int, default=1)
    parser.add_argument("--start_gap", type=float, default=0.05, help="Gap between start footprint and cabinet.")
    parser.add_argument("--start_xy", type=float, nargs=2, default=None,
                        help="Object start x y. Default: first free spot left of, right of, "
                             "then in front of the cabinet.")
    # Clutter / ghost goal (defaults in dextoolbench/clutter.py CLUTTER_DEFAULTS).
    parser.add_argument("--num_slots", type=int, default=None)
    parser.add_argument("--count_range", type=int, nargs=2, default=None)
    parser.add_argument("--front_count_range", type=int, nargs=2, default=None)
    parser.add_argument("--goal_levels", type=int, nargs="+", default=None)
    parser.add_argument("--clutter_levels", type=int, nargs="+", default=None,
                        help="Levels that may hold clutter; only used with --clutter_all_levels.")
    parser.add_argument("--clutter_all_levels", action="store_true",
                        help="Also clutter shelves other than the goal's (default: goal shelf only).")
    parser.add_argument("--hand_up", type=float, default=None, help="Hand envelope above the object (m).")
    parser.add_argument("--hand_side", type=float, default=None, help="Hand envelope beside the object (m).")
    parser.add_argument("--clutter_seed", type=int, default=None, help="Seed of the build-time item pool.")
    parser.add_argument("--layout_seed", type=int, default=0, help="Seed of the example layout's goals.")
    args = parser.parse_args()

    traj_dir = REPO_ROOT / "dextoolbench/trajectories" / args.object_category / args.object_name
    rest = json.loads((traj_dir / f"{args.rest_pose_from}.json").read_text())["start_pose"]
    y0, rest_z, quat = rest[1], rest[2], rest[3:7]
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = world_extents(
        object_mesh_vertices(args.object_category, args.object_name), quat
    )
    obj = dict(
        rest_quat=quat,
        rest_dz=rest_z - TABLE_TOP_Z,  # object origin height above its support surface
        aabb_x=[x_min, x_max], aabb_y=[y_min, y_max], aabb_z=[z_min, z_max],
    )

    w, d, h = args.cabinet_size
    cabinet = {
        "pos": [args.cabinet_pos[0], args.cabinet_pos[1], TABLE_TOP_Z + h / 2],
        "yaw_deg": args.cabinet_yaw_deg,
        "size": [w, d, h],
        "wall_thickness": args.wall_thickness,
        "num_shelves": args.num_shelves,
        "shelf_heights": [],
        "color": [0.55, 0.40, 0.28],
        "friction": 0.5,
    }
    assert args.cabinet_yaw_deg == 0.0, "start-pose placement below assumes an unrotated cabinet"
    cab_x_lo, cab_x_hi = args.cabinet_pos[0] - w / 2, args.cabinet_pos[0] + w / 2
    cab_y_lo, cab_y_hi = args.cabinet_pos[1] - d / 2, args.cabinet_pos[1] + d / 2
    # The cabinet is kinematic, so overhanging the table edge is physically fine.
    overhang_x = max(-TABLE_X_HALF - cab_x_lo, cab_x_hi - TABLE_X_HALF)
    overhang_y = max(TABLE_Y_RANGE[0] - cab_y_lo, cab_y_hi - TABLE_Y_RANGE[1])
    if overhang_x > 1e-9 or overhang_y > 1e-9:
        print(f"WARNING: cabinet overhangs the table by x={max(overhang_x, 0):.3f} y={max(overhang_y, 0):.3f} m")

    # Start: on the table, clear of the cabinet footprint by start_gap.
    def footprint_ok(x, y):
        lo_x, hi_x, lo_y, hi_y = x + x_min, x + x_max, y + y_min, y + y_max
        on_table = -TABLE_X_HALF < lo_x and hi_x < TABLE_X_HALF and TABLE_Y_RANGE[0] < lo_y and hi_y < TABLE_Y_RANGE[1]
        g = args.start_gap - 1e-9  # candidates sit exactly start_gap away
        clear = (hi_x < cab_x_lo - g or lo_x > cab_x_hi + g or hi_y < cab_y_lo - g or lo_y > cab_y_hi + g)
        return on_table and clear

    if args.start_xy is not None:
        start_x, start_y = args.start_xy
        side = "given"
    else:
        candidates = [
            ("left", cab_x_lo - args.start_gap - x_max, y0),
            ("right", cab_x_hi + args.start_gap - x_min, y0),
            ("front", args.cabinet_pos[0] - (x_min + x_max) / 2, cab_y_hi + args.start_gap - y_min),
        ]
        fits = [c for c in candidates if footprint_ok(c[1], c[2])]
        assert fits, ("no free start spot left/right/in front of the cabinet; "
                      "shrink or move the cabinet, or pass --start_xy")
        side, start_x, start_y = fits[0]
    assert footprint_ok(start_x, start_y), (
        f"start footprint at ({start_x:.3f}, {start_y:.3f}) is off the table or within "
        f"start_gap={args.start_gap} of the cabinet"
    )
    start_pose = [start_x, start_y, rest_z] + quat

    # Clutter block: defaults overridden by the given flags.
    overrides = dict(
        num_slots=args.num_slots, count_range=args.count_range,
        front_count_range=args.front_count_range, goal_levels=args.goal_levels,
        levels=args.clutter_levels, seed=args.clutter_seed,
        goal_level_only=False if args.clutter_all_levels else None,
    )
    clutter = clutter_spec({k: v for k, v in overrides.items() if v is not None})
    clutter["hand_envelope"] = dict(clutter["hand_envelope"])
    if args.hand_up is not None:
        clutter["hand_envelope"]["up"] = args.hand_up
    if args.hand_side is not None:
        clutter["hand_envelope"]["side"] = args.hand_side

    surfaces = shelf_surfaces(cabinet)
    caps = [front_height_cap(s, obj, clutter) for s in surfaces]
    print(f"object AABB about origin: x [{x_min:+.4f}, {x_max:+.4f}] y [{y_min:+.4f}, {y_max:+.4f}] "
          f"z [{z_min:+.4f}, {z_max:+.4f}]")
    print(f"cabinet {w}x{d}x{h} at {cabinet['pos']}; levels (env z): "
          + ", ".join(f"{s['z_top'] + cabinet['pos'][2]:.3f} (clear {s['clear_height']:.3f}, "
                      f"front cap {c:.3f})" for s, c in zip(surfaces, caps)))
    # Pool sized against the tightest goal level's cap, so short items fit any corridor.
    goal_caps = [caps[lv] for lv in clutter["goal_levels"] if lv < len(surfaces) and caps[lv] >= 0]
    assert goal_caps, "no goal level leaves room for the hand; enlarge the compartments"
    pool = sample_clutter_pool(clutter, min(goal_caps))
    layout = sample_layout(cabinet, clutter, obj, env_items(pool, 0), start_pose,
                           random.Random(args.layout_seed))
    print(f"start ({side}) x={start_x:+.4f} y={start_y:+.4f}")
    print(f"example layout: goal level {layout['level']}, {len(layout['placed'])} placed "
          f"({layout['n_front']} in front), {len(layout['parked'])} parked; "
          f"carry z (env) {layout['waypoints'][1][2]:.3f}")
    for name, g in zip(("lift", "pre-insert", "over-goal", "place"), layout["waypoints"]):
        print(f"  {name:10s} ({g[0]:+.3f}, {g[1]:+.3f}, {g[2]:.3f})")

    out = traj_dir / f"{args.task_name}.json"
    out.write_text(json.dumps(
        {"start_pose": start_pose, "goals": layout["waypoints"], "cabinet": cabinet,
         "object": obj, "clutter": clutter, "table_urdf": TABLE_URDF},
        indent=4,
    ) + "\n")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
