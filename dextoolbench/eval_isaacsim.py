"""DexToolBench policy evaluation — Isaac Sim (Isaac Lab) backend.

Clone of ``dextoolbench/eval_isaacgym.py`` (Isaac Gym) running on
``Isaacsimenvs-SimToolReal-Direct-v0``. Loads a specific DexToolBench object
URDF + task trajectory, rolls the policy for N episodes, and writes the same
``eval.json`` schema (avg_goal_pct / avg_time_sec / per-episode lists).

Run inside the Isaac Sim venv:

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python dextoolbench/eval_isaacsim.py \\
        --object_category hammer \\
        --object_name claw_hammer \\
        --task_name swing_down \\
        --config_path pretrained_policy/config.yaml \\
        --checkpoint_path pretrained_policy/model.pth \\
        --num_episodes 10 \\
        --output_dir evals_isaacsim/hammer/claw_hammer/swing_down
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]

CONTROL_HZ = 60.0

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_category", required=True)
    parser.add_argument("--object_name", required=True)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--config_path", default=str(REPO_ROOT / "pretrained_policy/config.yaml"))
    parser.add_argument("--checkpoint_path", default=str(REPO_ROOT / "pretrained_policy/model.pth"))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--downsample_factor", type=int, default=1,
                        help="Downsample factor for trajectory goals.")
    parser.add_argument("--z_offset", type=float, default=0.03,
                        help="Z offset added to start pose to avoid the table.")
    parser.add_argument("--force_table_urdf", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Always use the default table URDF regardless of category.")
    parser.add_argument("--robot_reset_noise", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Keep the training-time robot joint pos/vel reset noise so "
                             "episodes differ (object start pose and goals stay fixed).")
    parser.add_argument("--episode_length_s", type=float, default=None,
                        help="Per-goal timeout (s); resets on each goal reached. Default: env cfg (10 s).")
    parser.add_argument("--no_obstacles", action="store_true",
                        help="Drop the task's obstacles (control run with the same goals).")
    parser.add_argument("--obstacle_tilt_threshold_deg", type=float, default=45.0,
                        help="An obstacle counts as knocked down once its z-axis tilts past this.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rl_device", default="cuda")
    return parser


def _launch_app():
    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    app = AppLauncher(args).app
    return app, args


_app, _args = _launch_app()


def main() -> None:
    args = _args

    import gymnasium as gym
    import numpy as np
    import torch

    import isaacsimenvs  # noqa: F401  registers gym envs
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from deployment.rl_player import RlPlayer
    from dextoolbench.task_cfg import apply_dextoolbench_task

    # --- Env config (mirrors dextoolbench/eval_isaacgym.py overrides) ---
    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = 1
    traj_data = apply_dextoolbench_task(
        cfg, args.object_category, args.object_name, args.task_name,
        z_offset=args.z_offset,
        downsample_factor=args.downsample_factor,
        force_table_urdf=args.force_table_urdf,
        obstacles=not args.no_obstacles,
    )
    n_goals = len(traj_data["goals"])

    rs = cfg.reset
    rs.reset_position_noise_x = 0.0
    rs.reset_position_noise_y = 0.0
    rs.reset_position_noise_z = 0.0
    if not args.robot_reset_noise:
        rs.reset_dof_pos_random_interval_arm = 0.0
        rs.reset_dof_pos_random_interval_fingers = 0.0
        rs.reset_dof_vel_random_interval = 0.0
    cfg.seed = args.seed
    # Same knock-down definition as the per-episode obstacle stats below.
    cfg.reward.obstacle_knock_tilt_deg = args.obstacle_tilt_threshold_deg

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
    if args.episode_length_s is not None:
        term.episode_length = round(term.episode_length * args.episode_length_s / cfg.episode_length_s)
        cfg.episode_length_s = args.episode_length_s

    env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    inner._replay_target_lab_order = None

    n_act = cfg.action_space
    player = RlPlayer(
        num_observations=inner.cfg.observation_space,
        num_actions=n_act,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.rl_device,
        num_envs=1,
    )

    episode_goal_pcts: list[float] = []
    episode_lengths: list[int] = []
    # Per-episode obstacle stats (obstacle 0 only; empty lists when the task has none).
    obstacle = inner.obstacles[0] if getattr(inner, "obstacles", None) else None
    obstacle_max_tilts: list[float] = []
    obstacle_knocked_down: list[bool] = []
    obstacle_knocked_at_goals: list[int | None] = []
    obstacle_final_disp_xy: list[list[float]] = []
    # Per-episode stall diagnostics for the first unreached goal.
    stall_min_kp_dists: list[float | None] = []
    final_object_pos: list[list[float]] = []
    final_goal_pos: list[list[float]] = []

    def local_pos(asset) -> np.ndarray:
        return (asset.data.root_pos_w[0] - inner.scene.env_origins[0]).cpu().numpy()

    def obstacle_tilt_deg() -> float:
        # Angle between the body z-axis and world z: cos = 1 - 2 (qx^2 + qy^2).
        q = obstacle.data.root_quat_w[0]
        cos = (1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)).clamp(-1.0, 1.0)
        return float(torch.rad2deg(torch.arccos(cos)).item())

    def obstacle_xy_local() -> np.ndarray:
        return (obstacle.data.root_pos_w[0, :2] - inner.scene.env_origins[0, :2]).cpu().numpy()

    for ep in range(args.num_episodes):
        player.player.init_rnn()
        obs, _ = env.reset()
        # Match gym driver's reset timing: one physics tick before the first
        # policy action (gym's first step doubles as the reset trigger).
        obs, _, _, _, _ = env.step(torch.zeros((1, n_act), device=inner.device))

        step, done, goals_reached = 0, False, 0
        max_tilt, knocked_at, last_xy = 0.0, None, None
        cur_goal, min_kp_dist, obj_pos, goal_pos = 0, float("inf"), None, None
        start_xy = obstacle_xy_local() if obstacle is not None else None
        # Loop on the env's own done signal only (matches eval_isaacgym.py and the
        # interactive worker). episode_length_s is a PER-GOAL timeout that the env
        # resets on each goal reached, so a multi-goal trajectory legitimately runs
        # far longer than episode_length_s; a fixed max-step cap here would guillotine
        # slow multi-goal tasks (e.g. flip_over) after the first goal's budget.
        while not done:
            policy_obs = obs["policy"].to(args.rl_device)
            action = player.get_normalized_action(policy_obs, deterministic_actions=True)
            obs, _, terminated, truncated, _ = env.step(action.to(inner.device))
            done = bool(terminated[0].item() or truncated[0].item())
            if done:
                # step() already reset the env: the episode's final count was
                # copied into _prev_episode_successes before _successes zeroed.
                goals_reached = max(goals_reached, int(inner._prev_episode_successes[0].item()))
            else:
                goals_reached = max(goals_reached, int(inner._successes[0].item()))
                # Closest approach (max keypoint distance) to the current goal.
                if goals_reached != cur_goal:
                    cur_goal, min_kp_dist = goals_reached, float("inf")
                min_kp_dist = min(min_kp_dist, float(inner._keypoints_max_dist[0].item()))
                obj_pos, goal_pos = local_pos(inner.object), local_pos(inner.goal_viz)
                # Only sample the obstacle before the env auto-resets it.
                if obstacle is not None:
                    tilt = obstacle_tilt_deg()
                    max_tilt = max(max_tilt, tilt)
                    if knocked_at is None and tilt > args.obstacle_tilt_threshold_deg:
                        knocked_at = goals_reached
                    last_xy = obstacle_xy_local()
            step += 1

        goal_pct = 100.0 * goals_reached / n_goals
        episode_goal_pcts.append(goal_pct)
        episode_lengths.append(step)
        msg = (f"[eval] episode {ep + 1}/{args.num_episodes}: "
               f"{goals_reached}/{n_goals} goals ({goal_pct:.0f}%), "
               f"{step / CONTROL_HZ:.1f}s")
        stalled = goals_reached < n_goals and obj_pos is not None
        stall_min_kp_dists.append(min_kp_dist if stalled else None)
        final_object_pos.append([float(v) for v in obj_pos] if obj_pos is not None else [])
        final_goal_pos.append([float(v) for v in goal_pos] if goal_pos is not None else [])
        if stalled:
            msg += (f", stalled on goal {goals_reached + 1}: closest kp dist {min_kp_dist:.3f}, "
                    f"object-goal {np.round(obj_pos - goal_pos, 3).tolist()}")
        if obstacle is not None:
            disp = (last_xy - start_xy) if last_xy is not None else np.zeros(2)
            obstacle_max_tilts.append(max_tilt)
            obstacle_knocked_down.append(knocked_at is not None)
            obstacle_knocked_at_goals.append(knocked_at)
            obstacle_final_disp_xy.append([float(d) for d in disp])
            msg += (f", obstacle max tilt {max_tilt:.0f}deg"
                    + (f" (knocked down after {knocked_at} goals)" if knocked_at is not None else "")
                    + f", moved dx={disp[0]:+.3f} dy={disp[1]:+.3f}")
        print(msg)

    avg_goal_pct = float(np.mean(episode_goal_pcts))
    avg_time_sec = float(np.mean(episode_lengths) / CONTROL_HZ)
    print(f"[eval] DONE: avg_goal_pct={avg_goal_pct:.1f}, avg_time_sec={avg_time_sec:.1f}")
    if obstacle_knocked_down:
        knocked = np.array(obstacle_knocked_down)
        pcts = np.array(episode_goal_pcts)
        print(f"[eval] obstacle knocked down in {knocked.sum()}/{knocked.size} episodes; avg_goal_pct "
              f"knocked={pcts[knocked].mean() if knocked.any() else float('nan'):.1f}, "
              f"upright={pcts[~knocked].mean() if (~knocked).any() else float('nan'):.1f}")

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "eval.json", "w") as f:
            json.dump(
                {
                    "avg_goal_pct": avg_goal_pct,
                    "avg_time_sec": avg_time_sec,
                    "episode_goal_pcts": episode_goal_pcts,
                    "episode_lengths": episode_lengths,
                    "obstacle_tilt_threshold_deg": args.obstacle_tilt_threshold_deg,
                    "episode_obstacle_max_tilt_deg": obstacle_max_tilts,
                    "episode_obstacle_knocked_down": obstacle_knocked_down,
                    "episode_obstacle_knocked_at_goals": obstacle_knocked_at_goals,
                    "episode_obstacle_final_disp_xy": obstacle_final_disp_xy,
                    "episode_stall_min_kp_dist": stall_min_kp_dists,
                    "episode_final_object_pos": final_object_pos,
                    "episode_final_goal_pos": final_goal_pos,
                },
                f,
                indent=4,
            )
        print(f"[eval] wrote {output_dir / 'eval.json'}")

    import os
    import sys
    # Skip Kit teardown (it hangs); os._exit makes cleanup moot.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
