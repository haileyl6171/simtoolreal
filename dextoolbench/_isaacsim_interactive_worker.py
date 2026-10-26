"""Isaac Sim child-process worker for the DexToolBench interactive demo.

Lives in its own module with stdlib-only top-level imports: multiprocessing
``spawn`` re-imports the target function's module in the child, and Kit
segfaults at boot if heavy C extensions (viser's trimesh/embreex stack from
eval_interactive) are already loaded. Keep this module import-clean.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLE_Z = 0.38
Z_OFFSET = 0.03
CONTROL_DT = 1.0 / 60.0


def _track_arm_limits(inner, stats):
    """Accumulate per-arm-joint closest approach (rad) to the joint limits, for
    the measured position and the commanded target, plus the deepest object y."""
    q = inner.robot.data.joint_pos[0, inner._arm_joint_ids]
    tgt = inner._cur_targets[0, inner._arm_joint_ids]
    lo, hi = inner._arm_lower[0], inner._arm_upper[0]
    for key, v in (("pos", q), ("target", tgt)):
        m = torch_min(v - lo, hi - v)
        prev = stats.get(key)
        stats[key] = m if prev is None else torch_min(prev, m)
    stats["near_steps"] = stats.get("near_steps", 0) + int((torch_min(q - lo, hi - q) < 0.05).any())
    y = float(inner.object.data.root_pos_w[0, 1] - inner.scene.env_origins[0, 1])
    stats["min_obj_y"] = min(stats.get("min_obj_y", y), y)


def torch_min(a, b):
    import torch

    return torch.minimum(a, b)


def _report_arm_limits(stats, steps):
    import math

    if "pos" not in stats:
        return
    fmt = lambda m: " ".join(f"j{i + 1}={math.degrees(v):.0f}" for i, v in enumerate(m.tolist()))  # noqa: E731
    print(f"[limits] closest approach to arm joint limits (deg) over {steps} steps: "
          f"pos [{fmt(stats['pos'])}] target [{fmt(stats['target'])}]; "
          f"steps with a joint <3deg from its limit: {stats['near_steps']}; "
          f"deepest object y {stats['min_obj_y']:+.3f}", flush=True)


def _write_isaac_trajectory(goals_xyzw, out_dir: Path) -> str:
    """gym-format goals ([x,y,z,qx,qy,qz,qw] list) -> fixed_trajectory_file
    format (pos (1,K,3), quat wxyz (1,K,4))."""
    pos = [[g[:3] for g in goals_xyzw]]
    quat_wxyz = [[[g[6], g[3], g[4], g[5]] for g in goals_xyzw]]
    path = out_dir / "trajectory_isaac_format.json"
    with open(path, "w") as f:
        json.dump({"pos": pos, "quat_wxyz": quat_wxyz}, f)
    return str(path)


def _sim_get_state(inner, obs):
    """Visualisation state (joint_pos, object pose, goal pose) in the format
    the gym worker sends: poses env-local, quats xyzw."""
    import numpy as np

    obs_np = obs["policy"][0].detach().cpu().numpy()
    lower = inner._joint_lower_canon.detach().cpu().numpy()
    upper = inner._joint_upper_canon.detach().cpu().numpy()
    joint_pos = 0.5 * (obs_np[:29] + 1.0) * (upper - lower) + lower

    origin = inner.scene.env_origins[0].detach().cpu().numpy()

    def _pose7(rigid):
        pos = rigid.data.root_pos_w[0].detach().cpu().numpy() - origin
        wxyz = rigid.data.root_quat_w[0].detach().cpu().numpy()
        xyzw = wxyz[[1, 2, 3, 0]]
        return np.concatenate([pos, xyzw]).astype(np.float32)

    obstacle_poses = [_pose7(o) for o in getattr(inner, "obstacles", [])]
    return joint_pos, _pose7(inner.object), _pose7(inner.goal_viz), obstacle_poses


def _sim_episode(conn, env, inner, player, n_act, n_goals, rl_device):
    """Run one episode, streaming state to the parent via *conn*."""
    import time

    import torch

    limit_stats = {}
    player.player.init_rnn()
    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(torch.zeros((1, n_act), device=inner.device))

    step, done, paused = 0, False, False
    goals_reached = 0

    while not done:
        while conn.poll(0):
            cmd = conn.recv()
            if cmd == "pause":
                paused = True
            elif cmd == "resume":
                paused = False
            elif cmd == "stop":
                conn.send(("stopped",))
                return

        if paused:
            time.sleep(0.05)
            continue

        t0 = time.time()

        state = _sim_get_state(inner, obs)
        _track_arm_limits(inner, limit_stats)
        policy_obs = obs["policy"].to(rl_device)
        action = player.get_normalized_action(policy_obs, deterministic_actions=True)
        obs, _, terminated, truncated, _ = env.step(action.to(inner.device))
        done = bool(terminated[0].item() or truncated[0].item())
        if done:
            goals_reached = max(goals_reached, int(inner._prev_episode_successes[0].item()))
        else:
            goals_reached = max(goals_reached, int(inner._successes[0].item()))
        step += 1

        conn.send(("state", state, goals_reached, n_goals, step))

        elapsed = time.time() - t0
        if (sleep := CONTROL_DT - elapsed) > 0:
            time.sleep(sleep)

    _report_arm_limits(limit_stats, step)
    goal_pct = 100 * goals_reached / n_goals
    conn.send(("done", goal_pct, step))


def sim_worker_isaacsim(conn, category, object_name, task_name, table_urdf,
                        config_path, checkpoint_path):
    """Child process entry-point: boots Kit, creates the env, waits for commands."""
    try:
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(parser)
        launcher_args = parser.parse_args([])
        launcher_args.headless = True
        app = AppLauncher(launcher_args).app  # noqa: F841

        import gymnasium as gym
        import torch

        import isaacsimenvs  # noqa: F401  registers gym envs
        from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import (
            SimToolRealEnvCfg,
            cabinet_cfg_from_trajectory,
            obstacle_cfgs_from_trajectory,
        )
        from deployment.rl_player import RlPlayer
        from dextoolbench.objects import NAME_TO_OBJECT

        rl_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Trajectory (gym format) -> fixed_trajectory_file (isaac format).
        traj_path = (
            REPO_ROOT / "dextoolbench" / "trajectories"
            / category / object_name / f"{task_name}.json"
        )
        with open(traj_path) as f:
            traj_data = json.load(f)
        if "clutter" in traj_data:
            # Cabinet task: fresh clutter layout + ghost goal per load; the
            # placed items become free-standing obstacles, its waypoints the goals.
            import random

            from dextoolbench.clutter import sample_task_layout

            seed = int.from_bytes(os.urandom(4), "little")
            layout, clutter_obstacles = sample_task_layout(traj_data, random.Random(seed))
            traj_data["goals"] = layout["waypoints"]
            traj_data["obstacles"] = traj_data.get("obstacles", []) + clutter_obstacles
            print(f"[clutter] seed {seed}: goal level {layout['level']}, "
                  f"{len(clutter_obstacles)} items ({layout['n_front']} in front), "
                  f"carry z {layout['waypoints'][1][2]:.3f}", flush=True)
        traj_data["start_pose"][2] += Z_OFFSET
        n_goals = len(traj_data["goals"])
        tmp_dir = Path(tempfile.mkdtemp(prefix="dextoolbench_interactive_"))
        traj_file = _write_isaac_trajectory(traj_data["goals"], tmp_dir)

        obj = NAME_TO_OBJECT[object_name]

        cfg = SimToolRealEnvCfg()
        cfg.scene.num_envs = 1
        cfg.assets.object_urdf = str(obj.decomposed_urdf_path)
        cfg.assets.object_scale = tuple(obj.scale)
        # table_urdf arrives relative to assets/ (gym convention).
        cfg.assets.table_urdf = str(REPO_ROOT / "assets" / table_urdf)
        cfg.assets.obstacles = obstacle_cfgs_from_trajectory(traj_data)
        cfg.assets.cabinet = cabinet_cfg_from_trajectory(traj_data)

        rs = cfg.reset
        rs.reset_position_noise_x = 0.0
        rs.reset_position_noise_y = 0.0
        rs.reset_position_noise_z = 0.0
        rs.reset_dof_pos_random_interval_arm = 0.0
        rs.reset_dof_pos_random_interval_fingers = 0.0
        rs.reset_dof_vel_random_interval = 0.0
        rs.table_reset_z = TABLE_Z
        rs.table_reset_z_range = 0.0
        rs.start_arm_higher = True
        sp = traj_data["start_pose"]
        rs.fixed_start_pose = (sp[0], sp[1], sp[2], sp[6], sp[3], sp[4], sp[5])
        rs.fixed_trajectory_file = traj_file

        dr = cfg.domain_randomization
        dr.use_obs_delay = False
        dr.use_action_delay = False
        dr.use_object_state_delay_noise = False
        dr.object_scale_noise_multiplier_range = (1.0, 1.0)
        dr.joint_velocity_obs_noise_std = 0.0
        dr.force_scale = 0.0
        dr.torque_scale = 0.0
        dr.force_prob_range = (0.0001, 0.0001)
        dr.torque_prob_range = (0.0001, 0.0001)

        term = cfg.termination
        term.eval_success_tolerance = 0.01
        term.success_steps = 1
        term.max_consecutive_successes = n_goals
        # Quadruple the per-goal timeout (the counter resets on each goal hit).
        cfg.episode_length_s *= 4
        term.episode_length *= 4

        env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
        inner = env.unwrapped
        inner._replay_target_lab_order = None

        n_act = cfg.action_space
        player = RlPlayer(
            num_observations=inner.cfg.observation_space,
            num_actions=n_act,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=rl_device,
            num_envs=1,
        )

        player.player.init_rnn()
        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(torch.zeros((1, n_act), device=inner.device))
        init_state = _sim_get_state(inner, obs)

        # Obstacle specs too, since cabinet tasks sample them here, not in the JSON.
        conn.send(("ready", init_state, traj_data.get("obstacles", [])))

        while True:
            cmd = conn.recv()
            if cmd == "run":
                _sim_episode(conn, env, inner, player, n_act, n_goals, rl_device)
            elif cmd == "quit":
                break

    except Exception as exc:
        try:
            conn.send(("error", f"{exc}\n{traceback.format_exc()}"))
        except (BrokenPipeError, OSError):
            pass

    try:
        conn.close()
    except OSError:
        pass
    # Skip Kit teardown (it hangs); the parent already has everything it needs.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    # Standalone entry point: launched via subprocess by
    # eval_interactive_isaacsim.py (Kit segfaults when booted inside a
    # multiprocessing-spawned child, so we use a plain process + socket).
    from multiprocessing.connection import Client

    parser = argparse.ArgumentParser()
    parser.add_argument("--address", required=True, help="host:port of the parent Listener")
    parser.add_argument("--authkey", required=True, help="hex auth key")
    parser.add_argument("--category", required=True)
    parser.add_argument("--object_name", required=True)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--table_urdf", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    cli = parser.parse_args()

    host, port_str = cli.address.rsplit(":", 1)
    conn = Client((host, int(port_str)), authkey=bytes.fromhex(cli.authkey))
    sim_worker_isaacsim(
        conn, cli.category, cli.object_name, cli.task_name, cli.table_urdf,
        cli.config_path, cli.checkpoint_path,
    )
