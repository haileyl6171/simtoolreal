"""Generate side-by-side flat_eraser pick-and-place tasks, with and without an obstacle.

Writes, for the flat_eraser:
  dextoolbench/trajectories/eraser/flat_eraser/pick_place.json
  dextoolbench/trajectories/eraser/flat_eraser/pick_place_obstacle.json
  dextoolbench/trajectories/eraser/flat_eraser/pick_place_obstacle_{2goal,1goal}.json
  dextoolbench/trajectories/eraser/flat_eraser/pick_place_push{,_noobstacle}.json
  assets/urdf/dextoolbench/environments/eraser/flat_eraser/{pick_place,pick_place_obstacle{,_2goal,_1goal},pick_place_push{,_noobstacle}}.urdf

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
      [--place_overlap 0.01] [--push_obstacle_size 0.04 0.04 0.12] [--push_scene_shift 0.02]
      [--push_shift 0.11] [--push_offset 0.15] [--push_clearance 0.08]

pick_place_push is a separate probe of whether the low-level policy follows a
hand-designed goal sequence: a tall, skinny box stands at the centre of the
place footprint, and the goals lift the eraser, carry it high above the box to
the right of the place pose, set it down there, then slide it left into the
place pose so it shoves the box out of the footprint (toward -x).
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
    parser.add_argument("--push_obstacle_size", type=float, nargs=3, default=[0.04, 0.04, 0.12],
                        metavar=("SX", "SY", "SZ"), help="pick_place_push: tall obstacle box size (m)")
    parser.add_argument("--push_scene_shift", type=float, default=0.02,
                        help="pick_place_push: shift (m) of the whole scene (start, goals, box) toward -x")
    parser.add_argument("--push_shift", type=float, default=0.11,
                        help="pick_place_push: extra shift (m) of the place pose and box toward -x, "
                             "leaving room on the right to come down clear of the box")
    parser.add_argument("--push_offset", type=float, default=0.15,
                        help="pick_place_push: +x shift (m) of the low-right goal from the place pose; "
                             "the held body (with hand_margin) must land clear of the box")
    parser.add_argument("--push_clearance", type=float, default=0.08,
                        help="pick_place_push: height (m) of the high place waypoint above the box C-obstacle")
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

    def segment_clearance(a, b, cx_lo=cx_lo, cx_hi=cx_hi, cz_top=cz_top):
        """Min height (m) of a straight a->b origin path above the C-space obstacle top, over its x span."""
        lo, hi = max(min(a[0], b[0]), cx_lo), min(max(a[0], b[0]), cx_hi)
        if lo > hi or (lo == hi and a[0] != b[0]):
            return float("inf")
        if a[0] == b[0]:  # vertical segment inside the x span
            return min(a[2], b[2]) - cz_top
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

    # pick_place_push: tall box at the centre of the place footprint; the eraser
    # lands right of it (low_right) and slides left into place, shoving it.
    # The whole scene is shifted left by push_scene_shift to make room on the right.
    px, py, pz = args.push_obstacle_size
    push_top = TABLE_TOP_Z + pz
    push_pick_x = pick_x - args.push_scene_shift
    push_start = [push_pick_x, y, rest_z] + quat
    push_prepick = [push_pick_x, y, prepick[2]] + quat
    push_place_x = place_x - args.push_scene_shift - args.push_shift
    push_x = push_place_x + (x_min + x_max) / 2
    push_place = [push_place_x, y, rest_z] + quat
    low_right = [push_place_x + args.push_offset, y, rest_z] + quat
    # High place sits directly above low_right, so the eraser comes straight
    # down beside the box rather than diagonally across it.
    high_place = [low_right[0], y, max(push_top - (z_min - m) + args.push_clearance, prepick[2])] + quat
    assert low_right[0] + x_min - m > push_x + px / 2, \
        "push_offset too small: held body at the low-right goal overlaps the box"
    assert push_pick_x + x_min > -TABLE_X_HALF, \
        f"push start footprint reaches x={push_pick_x + x_min:.3f}, past the table edge {-TABLE_X_HALF}"
    assert push_place_x + args.push_offset + x_max < TABLE_X_HALF, \
        f"low-right footprint reaches x={push_place_x + args.push_offset + x_max:.3f}, past the table edge {TABLE_X_HALF}"
    # C-space of the push box for the eraser origin (same construction as above).
    push_cspace = dict(cx_lo=push_x - px / 2 - (x_max + m), cx_hi=push_x + px / 2 - (x_min - m),
                       cz_top=push_top - (z_min - m))
    push_obstacle = dict(obstacle, size=[px, py, pz], pos=[push_x, y, push_top - pz / 2])
    print(f"push: box {px}x{py}x{pz} m at x={push_x:+.4f} (x [{push_x - px / 2:+.4f}, {push_x + px / 2:+.4f}], "
          f"top z={push_top:.3f}); high place z={high_place[2]:.4f}; low right x={low_right[0]:+.4f} "
          f"(eraser -x edge {low_right[0] + x_min:+.4f}, {low_right[0] + x_min - push_x - px / 2:.3f} m right of box); "
          f"place x={push_place_x:+.4f}; start x={push_pick_x:+.4f}")
    for name, a, b in (("prepick->high_place", push_prepick, high_place),
                       ("high_place->low_right", high_place, low_right)):
        print(f"  straight {name:22s} clearance over box C-obstacle: {segment_clearance(a, b, **push_cspace):+.4f} m")

    base = {"start_pose": start_pose, "goals": [prepick, transfer, place]}
    tasks = {
        "pick_place": base,
        "pick_place_obstacle": dict(base, obstacles=[obstacle]),
        # Same scene with shorter goal sequences: pick+place, and place only.
        "pick_place_obstacle_2goal": dict(base, goals=[prepick, place], obstacles=[obstacle]),
        "pick_place_obstacle_1goal": dict(base, goals=[place], obstacles=[obstacle]),
        "pick_place_push": dict(start_pose=push_start, goals=[push_prepick, high_place, low_right, push_place],
                                obstacles=[push_obstacle]),
        # Control: same goal sequence, no box.
        "pick_place_push_noobstacle": dict(start_pose=push_start, goals=[push_prepick, high_place, low_right, push_place]),
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
