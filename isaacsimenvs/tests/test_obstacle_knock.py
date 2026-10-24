"""Pure-torch checks for obstacle knock-down reward gating and critic widening.

Uses a fake env namespace (no scene is built); the headless app is only
needed because importing the task package pulls in isaaclab.envs.

    .venv_isaacsim/bin/python isaacsimenvs/tests/test_obstacle_knock.py
"""

from __future__ import annotations

from types import SimpleNamespace

from isaaclab.app import AppLauncher

_app = AppLauncher(headless=True).app

import torch  # noqa: E402

from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import RewardCfg, TerminationCfg  # noqa: E402
from isaacsimenvs.tasks.simtoolreal.utils.reward_utils import compute_rewards  # noqa: E402
from rl_games.torch_runner import _pad_input_dim  # noqa: E402


def _fake_env(knocked: list[bool], just_knocked: list[bool], penalty: float = 0.0):
    n = len(knocked)
    reward = RewardCfg()
    reward.obstacle_knock_penalty = penalty
    term = TerminationCfg()
    term.force_consecutive_near_goal_steps = True  # lump-sum bonus on _is_success
    return SimpleNamespace(
        cfg=SimpleNamespace(reward=reward, termination=term),
        scene=SimpleNamespace(env_origins=torch.zeros(n, 3)),
        object=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(n, 3))),
        robot=SimpleNamespace(data=SimpleNamespace(joint_vel=torch.zeros(n, 29))),
        _arm_joint_ids=list(range(7)),
        _hand_joint_ids=list(range(7, 29)),
        _object_init_z=torch.zeros(n),
        _lifted_object=torch.ones(n, dtype=torch.bool),
        _curr_fingertip_distances=torch.zeros(n, 5),
        _closest_fingertip_dist=torch.zeros(n, 5),
        _keypoints_max_dist=torch.zeros(n),
        _closest_keypoint_max_dist=torch.zeros(n),
        _near_goal=torch.ones(n, dtype=torch.bool),
        _is_success=torch.ones(n, dtype=torch.bool),
        _obstacle_knocked=torch.tensor(knocked),
        _obstacle_just_knocked=torch.tensor(just_knocked),
    )


def test_bonus_gated_when_knocked() -> None:
    env = _fake_env(knocked=[False, True, True], just_knocked=[False, True, False])
    compute_rewards(env)
    bonus = env._reward_terms["bonus_rew"]
    assert torch.equal(bonus, torch.tensor([RewardCfg().reach_goal_bonus, 0.0, 0.0])), bonus
    assert torch.all(env._reward_terms["obstacle_knock_penalty"] == 0.0)


def test_knock_penalty_once() -> None:
    env = _fake_env(knocked=[True, True], just_knocked=[True, False], penalty=300.0)
    compute_rewards(env)
    assert torch.equal(env._reward_terms["obstacle_knock_penalty"], torch.tensor([-300.0, 0.0]))


def test_pad_input_dim() -> None:
    # SAPG critic layout: 3 obs dims + 2 extra-param dims into the first layer;
    # running stats cover the obs only. Two obs dims are appended.
    old = {
        "net.0.weight": torch.randn(4, 3 + 2),
        "net.0.bias": torch.randn(4),
        "running_mean_std.running_mean": torch.randn(3),
        "running_mean_std.running_var": torch.rand(3) + 0.5,
        "running_mean_std.count": torch.tensor(1e6),
    }
    new = {
        "net.0.weight": torch.zeros(4, 5 + 2),
        "net.0.bias": torch.zeros(4),
        "running_mean_std.running_mean": torch.zeros(5),
        "running_mean_std.running_var": torch.zeros(5),
        "running_mean_std.count": torch.tensor(1.0),
    }
    out = _pad_input_dim(old, new)
    w = out["net.0.weight"]
    assert torch.equal(w[:, :3], old["net.0.weight"][:, :3])
    assert torch.all(w[:, 3:5] == 0.0)
    assert torch.equal(w[:, 5:], old["net.0.weight"][:, 3:])  # extra-param columns kept last
    assert torch.equal(out["net.0.bias"], old["net.0.bias"])
    assert torch.all(out["running_mean_std.running_mean"][3:] == 0.0)
    assert torch.all(out["running_mean_std.running_var"][3:] == 1.0)
    assert out["running_mean_std.count"] == old["running_mean_std.count"]
    for key, tensor in out.items():
        assert tensor.shape == new[key].shape, key


if __name__ == "__main__":
    test_bonus_gated_when_knocked()
    test_knock_penalty_once()
    test_pad_input_dim()
    print("test_obstacle_knock: OK")
    _app.close()
