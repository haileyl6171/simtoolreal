"""Generate side-by-side flat_eraser pick-and-place tasks, with and without an obstacle.

Writes, for the flat_eraser:
  dextoolbench/trajectories/eraser/flat_eraser/pick_place.json
  dextoolbench/trajectories/eraser/flat_eraser/pick_place_obstacle.json
  dextoolbench/trajectories/eraser/flat_eraser/pick_place_obstacle_{2goal,1goal}.json
  assets/urdf/dextoolbench/environments/eraser/flat_eraser/{pick_place,pick_place_obstacle{,_2goal,_1goal}}.urdf

Both tasks share start pose and goals: prepick above the eraser, a
"transfer" waypoint over the obstacle, then place beside it along x. The
transfer height comes from the Minkowski difference of the obstacle box and
the held body (eraser AABB inflated by --hand_margin for the fingers): the
eraser origin must sit outside obstacle ⊖ held_body, plus --clearance. The
no-obstacle task uses the same waypoint so the two differ only by the box. pick_place_obstacle adds one free-standing box between
the two footprints under an "obstacles" key; the Isaac Sim env spawns it as
its own rigid body, so it can be pushed or knocked over. Toggle the obstacle
by choosing the task.

Usage:
  python3 dextoolbench/generate_pick_place_task.py [--gap 0.05] [--obstacle_size 0.04 0.12 0.05]
      [--place_overlap 0.01]
      [--hand_margin 0.02] [--clearance 0.02]
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAJ_DIR = REPO_ROOT / "dextoolbench/trajectories/eraser/flat_eraser"
ENV_URDF_DIR = REPO_ROOT / "assets/urdf/dextoolbench/environments/eraser/flat_eraser"
ERASER_OBJ = REPO_ROOT / "assets/urdf/dextoolbench/eraser/flat_eraser/flat_eraser.obj"
PLAIN_TABLE_URDF = REPO_ROOT / "assets/urdf/table_narrow.urdf"

TABLE_Z = 0.38  # table actor centre (tableResetZ)
TABLE_TOP_Z = TABLE_Z + 0.15  # table box is 0.3 m tall
TABLE_X_HALF = 0.2375
TABLE_Y_HALF = 0.20


def quat_to_matrix(qx, qy, qz, qw):
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]


def eraser_world_extents(quat_xyzw):
    """Per-axis (min, max) of the eraser mesh, rotated to quat, relative to its origin."""
    R = quat_to_matrix(*quat_xyzw)
    verts = [
        [float(c) for c in line.split()[1:4]]
        for line in ERASER_OBJ.read_text().splitlines()
        if line.startswith("v ")
    ]
    rotated = [[sum(R[i][j] * v[j] for j in range(3)) for i in range(3)] for v in verts]
    return [(min(p[i] for p in rotated), max(p[i] for p in rotated)) for i in range(3)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gap", type=float, default=0.05,
                        help="clearance (m) between each eraser footprint and the obstacle along x")
    parser.add_argument("--obstacle_size", type=float, nargs=3, default=[0.04, 0.12, 0.05],
                        metavar=("SX", "SY", "SZ"), help="obstacle box size (m)")
    parser.add_argument("--obstacle_mass", type=float, default=0.1)
    parser.add_argument("--place_overlap", type=float, default=0.0,
                        help="shift the obstacle toward the place goal so it overlaps the place footprint "
                             "along x by this much (m); goals and transfer waypoint are unchanged")
    parser.add_argument("--prepick_lift", type=float, default=0.15)
    parser.add_argument("--hand_margin", type=float, default=0.02,
                        help="inflation (m) of the eraser AABB on every side to cover the grasping hand")
    parser.add_argument("--clearance", type=float, default=0.02,
                        help="extra height (m) of the transfer waypoint above the C-space obstacle")
    args = parser.parse_args()

    # Reuse the flat_eraser's resting pose (xy, z, orientation) from wipe_smile.
    wipe_start = json.loads((TRAJ_DIR / "wipe_smile.json").read_text())["start_pose"]
    y, rest_z, quat = wipe_start[1], wipe_start[2], wipe_start[3:7]

    (x_min, x_max), _, (z_min, z_max) = eraser_world_extents(quat)
    sx, sy, sz = args.obstacle_size
    obstacle_top = TABLE_TOP_Z + sz

    # Side by side along x: pick | gap | obstacle (centred at x=0) | gap | place.
    pick_x = -(sx / 2 + args.gap + x_max)
    place_x = sx / 2 + args.gap - x_min

    start_pose = [pick_x, y, rest_z] + quat
    prepick = [pick_x, y, rest_z + args.prepick_lift] + quat
    place = [place_x, y, rest_z] + quat

    # C-space obstacle for the eraser origin = obstacle ⊖ held body (Minkowski
    # difference), held body = eraser AABB inflated by hand_margin. In x-z:
    m = args.hand_margin
    cx_lo, cx_hi = -sx / 2 - (x_max + m), sx / 2 - (x_min - m)
    cz_top = obstacle_top - (z_min - m)
    # Transfer waypoint: held body centred over the obstacle, clear of its top.
    transfer_x = (cx_lo + cx_hi) / 2
    transfer_z = max(cz_top + args.clearance, prepick[2])
    transfer = [transfer_x, y, transfer_z] + quat
    # Obstacle x: centred at 0, or shifted so its +x face sits place_overlap past
    # the place footprint's -x edge (place_x + x_min = sx / 2 + gap).
    obstacle_x = args.gap + args.place_overlap if args.place_overlap > 0 else 0.0
    obstacle = {
        "size": [sx, sy, sz],
        "pos": [obstacle_x, y, obstacle_top - sz / 2],
        "yaw_deg": 0.0,
        "mass": args.obstacle_mass,
        "friction": 0.5,
        "color": [0.35, 0.35, 0.45],
    }

    # Sanity checks: everything on the table.
    for name, x in (("pick", pick_x), ("place", place_x)):
        lo, hi = x + x_min, x + x_max
        assert -TABLE_X_HALF < lo and hi < TABLE_X_HALF, f"{name} footprint x [{lo:.3f}, {hi:.3f}] off table"
    assert abs(y) + sy / 2 < TABLE_Y_HALF, "obstacle extends past table y-edge"

    assert not cx_lo < pick_x < cx_hi and not cx_lo < place_x < cx_hi, \
        "hand_margin exceeds gap: pick/place footprint overlaps the C-space obstacle"

    def segment_clearance(a, b):
        """Min height (m) of a straight a->b origin path above the C-space obstacle top, over its x span."""
        lo, hi = max(min(a[0], b[0]), cx_lo), min(max(a[0], b[0]), cx_hi)
        if lo >= hi:
            return float("inf")
        z_at = lambda x: a[2] + (x - a[0]) / (b[0] - a[0]) * (b[2] - a[2])
        return min(z_at(lo), z_at(hi)) - cz_top

    print(f"pick x={pick_x:.4f}  place x={place_x:.4f}  y={y:.4f}  rest z={rest_z:.4f}")
    print(f"eraser AABB about origin: x [{x_min:+.4f}, {x_max:+.4f}]  z [{z_min:+.4f}, {z_max:+.4f}]")
    print(f"obstacle {sx}x{sy}x{sz} m at x={obstacle_x:+.4f}, top z={obstacle_top:.3f}; gap each side={args.gap} m")
    if obstacle_x:
        print(f"obstacle x [{obstacle_x - sx / 2:+.4f}, {obstacle_x + sx / 2:+.4f}] overlaps place footprint "
              f"x [{place_x + x_min:+.4f}, {place_x + x_max:+.4f}] by {args.place_overlap} m")
    print(f"C-space obstacle (hand_margin={m}): x [{cx_lo:+.4f}, {cx_hi:+.4f}], top z={cz_top:.4f}")
    print(f"transfer waypoint: x={transfer_x:+.4f} z={transfer_z:.4f} "
          f"(held-body bottom {transfer_z + z_min - m:.4f}, {transfer_z - cz_top:+.4f} m over obstacle)")
    for name, a, b in (("prepick->place (old)", prepick, place),
                       ("prepick->transfer", prepick, transfer),
                       ("transfer->place", transfer, place)):
        print(f"  straight {name:22s} clearance over C-obstacle: {segment_clearance(a, b):+.4f} m")

    base = {"start_pose": start_pose, "goals": [prepick, transfer, place]}
    tasks = {
        "pick_place": base,
        "pick_place_obstacle": dict(base, obstacles=[obstacle]),
        # Same scene with shorter goal sequences: pick+place, and place only.
        "pick_place_obstacle_2goal": dict(base, goals=[prepick, place], obstacles=[obstacle]),
        "pick_place_obstacle_1goal": dict(base, goals=[place], obstacles=[obstacle]),
    }
    ENV_URDF_DIR.mkdir(parents=True, exist_ok=True)
    table_xml = PLAIN_TABLE_URDF.read_text()
    for task_name, data in tasks.items():
        traj_path = TRAJ_DIR / f"{task_name}.json"
        traj_path.write_text(json.dumps(data, indent=4) + "\n")
        # Both tasks use the plain table: the obstacle is not part of the table.
        urdf_path = ENV_URDF_DIR / f"{task_name}.urdf"
        urdf_path.write_text(table_xml.replace(
            '<robot name="box">', f'<robot name="table_flat_eraser_{task_name}">'
        ))
        print(f"wrote {traj_path.relative_to(REPO_ROOT)} and {urdf_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
