# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the vision DAgger stack: StudentTeacherVision + DistillationDAgger(Weighted) + DistillationRunnerSplit."""

from __future__ import annotations

import os
import tempfile
import torch
import torch.nn as nn
from tensordict import TensorDict

import pytest

from rsl_rl.env import VecEnv
from rsl_rl.modules import StudentTeacherVision
from rsl_rl.runners import DistillationRunnerSplit

NUM_ENVS = 10
PROPRIO_DIM = 6
TEACHER_DIM = 9
NUM_ACTIONS = 4
IMG_SHAPE = (1, 3, 32, 32)  # (T, C, H, W) as produced by ObsGroup(flatten_history_dim=False, history_length=1)
MAX_EP_LEN = 20


class _Teacher(nn.Module):
    """Mirrors the UWLab exporter contract: ``forward -> mean`` and ``compute_distribution -> (mean, std)``."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.num_actions: int = NUM_ACTIONS
        self.net = nn.Sequential(nn.Linear(TEACHER_DIM, 16), nn.ELU(), nn.Linear(16, 2 * NUM_ACTIONS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x)[..., : self.num_actions])

    @torch.jit.export
    def compute_distribution(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        mean = torch.tanh(out[..., : self.num_actions])
        return mean, torch.exp(out[..., self.num_actions :].clamp(-3.0, 0.5))


class _MeanStdTupleTeacher(nn.Module):
    """Yanda's ``convert_state_expert_to_jit.py --std`` contract: ``forward -> (mean, std)``."""

    def __init__(self) -> None:
        super().__init__()
        self.num_actions: int = NUM_ACTIONS

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = x[..., : self.num_actions]
        return mean, torch.ones_like(mean) * 0.5


def _save_jit(module: nn.Module, directory: str, name: str) -> str:
    path = os.path.join(directory, name)
    torch.jit.script(module).save(path)
    return path


def _make_obs(num_envs: int, generator: torch.Generator | None = None) -> TensorDict:
    return TensorDict(
        {
            "proprio": torch.randn(num_envs, PROPRIO_DIM, generator=generator),
            "side_rgb": torch.rand(num_envs, *IMG_SHAPE, generator=generator),
            "wrist_rgb": torch.rand(num_envs, *IMG_SHAPE, generator=generator),
            "teacher": torch.randn(num_envs, TEACHER_DIM, generator=generator),
        },
        batch_size=[num_envs],
    )


class _DummyEnv(VecEnv):
    def __init__(self, fixed_obs: bool = False) -> None:  # noqa: D107
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = MAX_EP_LEN
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long)
        self.device = "cpu"
        self.cfg = {}
        self._fixed = _make_obs(NUM_ENVS, torch.Generator().manual_seed(1)) if fixed_obs else None

    @property
    def unwrapped(self) -> _DummyEnv:
        return self

    def get_observations(self) -> TensorDict:  # noqa: D102
        return self._fixed.clone() if self._fixed is not None else _make_obs(self.num_envs)

    def step(self, actions: torch.Tensor):  # noqa: D102
        self.episode_length_buf += 1
        dones = (self.episode_length_buf >= self.max_episode_length).long()
        self.episode_length_buf[dones.bool()] = 0
        return self.get_observations(), torch.zeros(self.num_envs), dones, {"time_outs": torch.zeros(self.num_envs)}


def _train_cfg(teacher_path: str, weighted: bool = True, **overrides) -> dict:
    cfg = {
        "class_name": "DistillationRunnerSplit",
        "num_steps_per_env": 1,
        "save_interval": 1000,
        "logger": "tensorboard",
        "obs_groups": {"policy": ["proprio"], "teacher": ["teacher"]},
        "student_fraction": 0.5,
        "eval_fraction": 0.2,
        "teacher_eval_fraction": 0.5,
        "algorithm": {
            "class_name": "DistillationDAggerWeighted" if weighted else "DistillationDAgger",
            "num_learning_epochs": 1,
            "gradient_length": 1,
            "learning_rate": 1e-3,
            "max_grad_norm": 1.0,
        },
        "vision_policy": {
            "class_name": "StudentTeacherVision",
            "teacher_jit_path": teacher_path,
            "vision_groups": ["side_rgb", "wrist_rgb"],
            "embed_dim": 16,
            "student_hidden_dims": [32, 32],
            "encoder_type": "resnet18",
            "predict_std": weighted,
            "teacher_returns_std": weighted,
        },
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def teacher_path():
    with tempfile.TemporaryDirectory() as d:
        yield _save_jit(_Teacher(), d, "teacher.pt")


def test_pool_split_masks(teacher_path: str) -> None:
    """10 envs, 20% eval (half teacher-driven), remaining 8 split 50/50 student/teacher."""
    runner = DistillationRunnerSplit(_DummyEnv(), _train_cfg(teacher_path))
    assert (runner.num_student_eval, runner.num_teacher_eval) == (1, 1)
    assert (runner.num_student_train, runner.num_teacher_train) == (4, 4)
    assert runner.eval_mask.tolist() == [True, True] + [False] * 8
    assert runner.student_mask.tolist() == [True, False] + [True] * 4 + [False] * 4
    assert isinstance(runner.alg.policy, StudentTeacherVision)
    assert runner.alg.policy._teacher_has_compute_distribution


def test_act_routes_student_and_teacher_actions(teacher_path: str) -> None:
    runner = DistillationRunnerSplit(_DummyEnv(), _train_cfg(teacher_path))
    alg, policy = runner.alg, runner.alg.policy
    policy.eval()
    obs = _make_obs(NUM_ENVS)
    with torch.inference_mode():
        actions = alg.act(obs)
        teacher_mean = policy.evaluate(obs)
    student = runner.student_mask
    # teacher-driven envs execute the deterministic teacher mean
    torch.testing.assert_close(actions[~student], teacher_mean[~student])
    # student-driven envs execute student samples, never the teacher action
    assert not torch.allclose(actions[student], teacher_mean[student])
    # every env stores the teacher mean as its label
    torch.testing.assert_close(alg.transition.privileged_actions, teacher_mean)


def test_teacher_contracts(teacher_path: str) -> None:
    """compute_distribution teachers and tuple-forward teachers both give (mean, std)."""
    obs = _make_obs(NUM_ENVS)
    policy = StudentTeacherVision(obs, {"policy": ["proprio"], "teacher": ["teacher"]}, NUM_ACTIONS,
                                  teacher_jit_path=teacher_path, vision_groups=["side_rgb"], embed_dim=8,
                                  student_hidden_dims=[16], teacher_returns_std=True, predict_std=True)
    ref = _Teacher()
    mean, std = policy.evaluate_with_std(obs)
    ref_mean, ref_std = ref.compute_distribution(obs["teacher"])
    torch.testing.assert_close(mean, ref_mean)
    torch.testing.assert_close(std, ref_std)
    torch.testing.assert_close(policy.evaluate(obs), ref(obs["teacher"]))
    with tempfile.TemporaryDirectory() as d:
        tuple_path = _save_jit(_MeanStdTupleTeacher(), d, "tuple_teacher.pt")
        policy = StudentTeacherVision(obs, {"policy": ["proprio"], "teacher": ["teacher"]}, NUM_ACTIONS,
                                      teacher_jit_path=tuple_path, vision_groups=[], student_hidden_dims=[16],
                                      teacher_returns_std=True, predict_std=True)
        mean, std = policy.evaluate_with_std(obs)
        torch.testing.assert_close(mean, obs["teacher"][:, :NUM_ACTIONS])
        torch.testing.assert_close(policy.evaluate(obs), mean)


@pytest.mark.parametrize("weighted", [True, False])
def test_learn_reduces_behavior_loss_and_checkpoints(teacher_path: str, weighted: bool) -> None:
    """On a fixed observation the student should fit the teacher; save/load must round-trip."""
    torch.manual_seed(0)
    env = _DummyEnv(fixed_obs=True)
    with tempfile.TemporaryDirectory() as log_dir:
        runner = DistillationRunnerSplit(env, _train_cfg(teacher_path, weighted=weighted), log_dir=log_dir)
        alg = runner.alg

        def one_iteration() -> dict:
            obs = env.get_observations()
            alg.policy.eval()
            with torch.inference_mode():
                alg.act(obs)
                alg.process_env_step(obs, torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS, dtype=torch.long), {})
            alg.policy.train()
            return alg.update()

        first = one_iteration()
        assert "behavior" in first and ("student_fraction" in first)
        for _ in range(60):
            last = one_iteration()
        assert last["behavior"] < 0.5 * first["behavior"], (first, last)

        # full learn() loop runs (logging, pool metrics, final save)
        runner.learn(num_learning_iterations=3)
        ckpts = [f for f in os.listdir(log_dir) if f.startswith("model_")]
        assert ckpts, os.listdir(log_dir)

        reloaded = DistillationRunnerSplit(_DummyEnv(fixed_obs=True), _train_cfg(teacher_path, weighted=weighted))
        reloaded.load(os.path.join(log_dir, sorted(ckpts)[-1]))
        obs = env.get_observations()
        alg.policy.eval()
        reloaded.alg.policy.eval()
        with torch.inference_mode():
            torch.testing.assert_close(reloaded.alg.policy.act_inference(obs), alg.policy.act_inference(obs))
        assert reloaded.alg.num_updates == alg.num_updates
