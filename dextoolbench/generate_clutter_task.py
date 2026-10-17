"""
Generate a pick-and-place-in-clutter task for the flat_eraser.

Outputs:
  dextoolbench/trajectories/eraser/flat_eraser/clutter.json
  assets/urdf/table_narrow_clutter.urdf

Usage:
  python dextoolbench/generate_clutter_task.py
  python dextoolbench/generate_clutter_task.py --seed 42 --num_blocks 6
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Workspace constants  (all world-frame, metres)
# ---------------------------------------------------------------------------

# Table actor is placed at (0, 0, TABLE_CENTER_Z) in Isaac Gym sim.
# Table URDF box is 0.475 × 0.4 × 0.3 m, centred at URDF origin.
TABLE_CENTER_Z = 0.38
TABLE_HALF_HEIGHT = 0.15       # 0.3 / 2
TABLE_SURFACE_Z = 0.534        # back-calculated: wipe_smile start z (0.559) - eraser half-h (0.025)
TABLE_SURFACE_LOCAL_Z = TABLE_SURFACE_Z - TABLE_CENTER_Z  # = 0.154

TABLE_X_HALF = 0.2375          # 0.475 / 2
TABLE_Y_HALF = 0.20            # 0.4 / 2

# Eraser (flat_eraser) bounding-box half-extents from mesh vertex bounds:
#   L = 0.125 m  W = 0.028 m  H = 0.050 m
ERASER_HALF_L = 0.0625
ERASER_HALF_W = 0.014
ERASER_HALF_H = 0.025
# Half-diagonal in the horizontal plane (worst-case footprint radius):
ERASER_HALF_DIAG_XY = math.sqrt(ERASER_HALF_L**2 + ERASER_HALF_W**2)  # ≈ 0.064 m

# Eraser resting z in world frame (centre, no z_offset): wipe_smile start_pose z value.
# eval_isaacgym.py adds z_offset=0.03 to start_pose[2] at load time; goals are stored raw.
ERASER_REST_Z = 0.559

# Orientation quaternion [qx, qy, qz, qw] copied from wipe_smile.json start_pose.
ERASER_QUAT = [
    0.7265684189136249,
    0.008695016088282936,
    0.021493013816529252,
    0.686702832154974,
]

# Pick location: robot-facing side of the table, away from the clutter ring.
PICK_XY = (0.018, -0.12)

# Place location: far side of the table, centred in the ring.
PLACE_XY = (0.018, 0.05)

# Height the eraser is lifted to for the prepick goal.
PREPICK_LIFT = 0.15

# ---------------------------------------------------------------------------
# Clutter block parameters
# ---------------------------------------------------------------------------

# Ring radius from place centre to block centres.
# Must exceed eraser_half_diag + max_block_half_diag + safety:
#   max_block_half_diag ≈ sqrt(0.03² + 0.02²) ≈ 0.036
#   required > 0.064 + 0.036 + 0.025 = 0.125 m
RING_RADIUS = 0.13  # m  (exceeds the 0.125 minimum; keeps blocks within table)

# Block dimension ranges [min, max] in metres.
BLOCK_L_RANGE = (0.03, 0.06)   # x extent
BLOCK_W_RANGE = (0.02, 0.04)   # y extent
BLOCK_H_RANGE = (0.05, 0.10)   # height (visibly tall)

# Default generation parameters.
DEFAULT_NUM_BLOCKS = 6
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Goal computation
# ---------------------------------------------------------------------------

def compute_goals() -> List[List[float]]:
    qx, qy, qz, qw = ERASER_QUAT
    prepick = [PICK_XY[0], PICK_XY[1], ERASER_REST_Z + PREPICK_LIFT, qx, qy, qz, qw]
    place   = [PLACE_XY[0], PLACE_XY[1], ERASER_REST_Z, qx, qy, qz, qw]
    return [prepick, place]


# ---------------------------------------------------------------------------
# Block generation
# ---------------------------------------------------------------------------

def sample_blocks(num_blocks: int, seed: int) -> List[Dict]:
    """Return list of block dicts with world and table-local positions."""
    import random
    rng = random.Random(seed)

    blocks = []
    for k in range(num_blocks):
        # Evenly spaced angular positions plus small random jitter.
        base_angle = 2 * math.pi * k / num_blocks
        angle = base_angle + rng.uniform(-0.25, 0.25)  # radians

        l = rng.uniform(*BLOCK_L_RANGE)
        w = rng.uniform(*BLOCK_W_RANGE)
        h = rng.uniform(*BLOCK_H_RANGE)
        yaw = rng.uniform(0, math.pi)

        world_x = PLACE_XY[0] + RING_RADIUS * math.cos(angle)
        world_y = PLACE_XY[1] + RING_RADIUS * math.sin(angle)
        world_z = TABLE_SURFACE_Z + h / 2   # bottom face flush with table surface

        # Table-local frame: subtract table actor world position (0, 0, TABLE_CENTER_Z).
        urdf_x = world_x
        urdf_y = world_y
        urdf_z = world_z - TABLE_CENTER_Z   # = TABLE_SURFACE_LOCAL_Z + h/2 ≈ 0.154 + h/2

        blocks.append({
            "l": l, "w": w, "h": h, "yaw": yaw,
            "world_x": world_x, "world_y": world_y, "world_z": world_z,
            "urdf_x": urdf_x, "urdf_y": urdf_y, "urdf_z": urdf_z,
        })
    return blocks


def verify_clearance(blocks: List[Dict]) -> None:
    """Print per-block clearance and warn about table-bounds violations."""
    min_clearance = ERASER_HALF_DIAG_XY + 0.025  # minimum face-to-face gap required

    print("Block clearance check (distance from place centre - block half-diagonal):")
    ok = True
    for k, b in enumerate(blocks):
        dx = b["world_x"] - PLACE_XY[0]
        dy = b["world_y"] - PLACE_XY[1]
        dist = math.sqrt(dx**2 + dy**2)
        half_diag = math.sqrt((b["l"] / 2) ** 2 + (b["w"] / 2) ** 2)
        face_gap = dist - half_diag - ERASER_HALF_DIAG_XY
        status = "OK" if face_gap >= 0.025 else "WARNING: too close"
        print(f"  Block {k}: centre dist={dist:.3f}m  face gap={face_gap:.3f}m  [{status}]")

        if abs(b["urdf_x"]) + b["l"] / 2 > TABLE_X_HALF:
            print(f"  WARNING: block {k} extends past table x-edge")
            ok = False
        if abs(b["urdf_y"]) + b["w"] / 2 > TABLE_Y_HALF:
            print(f"  WARNING: block {k} extends past table y-edge")
            ok = False
    if ok:
        print("  All blocks within table bounds.")


# ---------------------------------------------------------------------------
# URDF generation
# ---------------------------------------------------------------------------

_URDF_HEADER = """\
<?xml version="1.0"?>
<robot name="box">
  <link name="box">
    <visual>
      <material name="wood">
        <color rgba="0.82 0.56 0.35 1.0"/>
      </material>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.475 0.4 0.3"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.475 0.4 0.3"/>
      </geometry>
    </collision>
    <!-- Clutter blocks generated by generate_clutter_task.py -->
"""

_URDF_FOOTER = """\
    <inertial>
      <mass value="500"/>
      <friction value="1.0"/>
      <inertia ixx="1000.0" ixy="0.0" ixz="0.0" iyy="1000.0" iyz="0.0" izz="1000.0"/>
    </inertial>
  </link>
</robot>
"""


def _block_xml(b: Dict, k: int) -> str:
    x, y, z = b["urdf_x"], b["urdf_y"], b["urdf_z"]
    l, w, h, yaw = b["l"], b["w"], b["h"], b["yaw"]
    return (
        f"    <!-- Block {k} -->\n"
        f"    <visual>\n"
        f"      <material name=\"block_{k}\"><color rgba=\"0.35 0.35 0.45 1.0\"/></material>\n"
        f"      <origin xyz=\"{x:.4f} {y:.4f} {z:.4f}\" rpy=\"0 0 {yaw:.4f}\"/>\n"
        f"      <geometry><box size=\"{l:.4f} {w:.4f} {h:.4f}\"/></geometry>\n"
        f"    </visual>\n"
        f"    <collision>\n"
        f"      <origin xyz=\"{x:.4f} {y:.4f} {z:.4f}\" rpy=\"0 0 {yaw:.4f}\"/>\n"
        f"      <geometry><box size=\"{l:.4f} {w:.4f} {h:.4f}\"/></geometry>\n"
        f"    </collision>\n"
    )


def write_clutter_urdf(blocks: List[Dict], output_path: Path) -> None:
    parts = [_URDF_HEADER]
    for k, b in enumerate(blocks):
        parts.append(_block_xml(b, k))
    parts.append(_URDF_FOOTER)
    output_path.write_text("".join(parts))
    print(f"Wrote URDF: {output_path}")


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def write_trajectory_json(
    start_pose: List[float], goals: List[List[float]], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"start_pose": start_pose, "goals": goals}, f, indent=4)
    print(f"Wrote JSON: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num_blocks", type=int, default=DEFAULT_NUM_BLOCKS)
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent

    qx, qy, qz, qw = ERASER_QUAT
    start_pose = [PICK_XY[0], PICK_XY[1], ERASER_REST_Z, qx, qy, qz, qw]
    goals = compute_goals()
    blocks = sample_blocks(args.num_blocks, args.seed)

    print(f"\nPick  : xy={PICK_XY}, z={ERASER_REST_Z}")
    print(f"Prepick goal : {goals[0][:3]}")
    print(f"Place goal   : {goals[1][:3]}")
    print(f"Pick → place distance: {abs(PLACE_XY[1] - PICK_XY[1]):.3f} m")
    print(f"Ring radius: {RING_RADIUS} m  (min clearance radius: {ERASER_HALF_DIAG_XY + 0.036 + 0.025:.3f} m)\n")

    verify_clearance(blocks)

    traj_path = repo_root / "dextoolbench/trajectories/eraser/flat_eraser/clutter.json"
    urdf_path = repo_root / "assets/urdf/table_narrow_clutter.urdf"
    # Per-task environment URDF read by the interactive demo (eval_interactive_isaac*.py)
    env_urdf_path = (
        repo_root / "assets/urdf/dextoolbench/environments/eraser/flat_eraser/clutter.urdf"
    )

    write_trajectory_json(start_pose, goals, traj_path)
    write_clutter_urdf(blocks, urdf_path)
    write_clutter_urdf(blocks, env_urdf_path)

    print("\nNext steps:")
    print("  metadata.py  : 'clutter' is already added to flat_eraser task list")
    print("  eval_isaacgym: run with --force_table_urdf False --task_name clutter")


if __name__ == "__main__":
    main()
