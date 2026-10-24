"""Apply a DexToolBench task (object + trajectory + obstacles) to a SimToolReal env cfg.

Shared by ``dextoolbench/eval_isaacsim.py`` and training (``SimToolRealEnvCfg.
dextoolbench_task``). Only sets what defines the task; eval-only overrides
(zero noise, DR off, pinned success tolerance) stay with the caller.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

TABLE_URDF = "assets/urdf/table_narrow.urdf"
TABLE_WHITEBOARD_URDF = "assets/urdf/table_narrow_whiteboard.urdf"
TABLE_NAIL_URDF = "assets/urdf/table_narrow_nail.urdf"
TABLE_BOWL_PLATE_URDF = "assets/urdf/table_narrow_bowl_plate.urdf"

OBJECT_CATEGORY_TO_TABLE_URDF = {
    "hammer": TABLE_NAIL_URDF,
    "spatula": TABLE_BOWL_PLATE_URDF,
    "eraser": TABLE_WHITEBOARD_URDF,
    "screwdriver": TABLE_URDF,
    "marker": TABLE_WHITEBOARD_URDF,
    "brush": TABLE_URDF,
}


def load_trajectory(object_category: str, object_name: str, task_name: str) -> dict:
    path = REPO_ROOT / "dextoolbench/trajectories" / object_category / object_name / f"{task_name}.json"
    assert path.exists(), f"Trajectory file not found: {path}"
    with open(path) as f:
        return json.load(f)


def write_isaac_trajectory(goals_xyzw: list[list[float]], out_dir: Path) -> str:
    """Convert gym-format goals ([x,y,z,qx,qy,qz,qw] list) into the
    fixed_trajectory_file format (pos (1,K,3), quat wxyz (1,K,4))."""
    pos = [[g[:3] for g in goals_xyzw]]
    quat_wxyz = [[[g[6], g[3], g[4], g[5]] for g in goals_xyzw]]
    path = out_dir / "trajectory_isaac_format.json"
    with open(path, "w") as f:
        json.dump({"pos": pos, "quat_wxyz": quat_wxyz}, f)
    return str(path)


def apply_dextoolbench_task(
    cfg,
    object_category: str,
    object_name: str,
    task_name: str,
    *,
    z_offset: float = 0.03,
    downsample_factor: int = 1,
    force_table_urdf: bool = True,
    obstacles: bool = True,
) -> dict:
    """Set object/table/obstacles/start pose/goals on ``cfg``; returns the loaded trajectory."""
    from dextoolbench.objects import NAME_TO_OBJECT
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import (
        obstacle_cfgs_from_trajectory,
    )

    obj = NAME_TO_OBJECT[object_name]
    traj_data = load_trajectory(object_category, object_name, task_name)
    traj_data["start_pose"][2] += z_offset
    traj_data["goals"] = traj_data["goals"][::downsample_factor]

    cfg.assets.object_urdf = str(obj.decomposed_urdf_path)
    cfg.assets.object_scale = tuple(obj.scale)
    cfg.assets.table_urdf = TABLE_URDF if force_table_urdf else (
        OBJECT_CATEGORY_TO_TABLE_URDF[object_category]
    )
    cfg.assets.obstacles = obstacle_cfgs_from_trajectory(traj_data) if obstacles else []

    rs = cfg.reset
    rs.table_reset_z = 0.38
    rs.table_reset_z_range = 0.0
    rs.start_arm_higher = True
    # gym start_pose is (x,y,z, qx,qy,qz,qw); cfg wants (x,y,z, qw,qx,qy,qz).
    sp = traj_data["start_pose"]
    rs.fixed_start_pose = (sp[0], sp[1], sp[2], sp[6], sp[3], sp[4], sp[5])
    tmp_dir = Path(tempfile.mkdtemp(prefix="dextoolbench_task_"))
    rs.fixed_trajectory_file = write_isaac_trajectory(traj_data["goals"], tmp_dir)

    cfg.termination.max_consecutive_successes = len(traj_data["goals"])
    return traj_data
