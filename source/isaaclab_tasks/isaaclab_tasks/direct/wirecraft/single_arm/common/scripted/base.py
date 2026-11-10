# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Per-env phase machine and bounded Cartesian commands.

Every env runs its own rollout from its own reset, so ``num_envs`` envs give
``num_envs`` independent rollouts. Nothing here writes object or joint state.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import (
    compute_pose_error,
    quat_apply,
    quat_from_euler_xyz,
    quat_mul,
    subtract_frame_transforms,
)


def bound(vector: torch.Tensor, limit: float) -> torch.Tensor:
    """Scale each row so its largest component stays within ``limit``."""
    return vector / torch.clamp(vector.abs().amax(-1, keepdim=True) / limit, min=1.0)


class ScriptedPolicy:
    PHASES: tuple[str, ...] = ()
    max_steps = 3000

    def __init__(self, env):
        self.env = env
        self.device = env.device
        count = env.num_envs
        self.phase = torch.zeros(count, dtype=torch.long, device=self.device)
        self.wait = torch.zeros(count, dtype=torch.long, device=self.device)
        self.steps = 0
        self.success = torch.zeros(count, dtype=torch.bool, device=self.device)
        self.success_step = torch.full((count,), -1, dtype=torch.long, device=self.device)
        self.faulted = torch.zeros(count, dtype=torch.bool, device=self.device)
        self.rest = torch.tensor(env.cfg.robot.grasp_rest_quat, device=self.device).repeat(count, 1)
        self.done_phase = len(self.PHASES) - 1
        self.phase_log: list[list[tuple[int, str]]] = [[(0, self.PHASES[0])] for _ in range(count)]

    def index(self, name: str) -> int:
        return self.PHASES.index(name)

    def at(self, name: str) -> torch.Tensor:
        return self.phase == self.index(name)

    def advance(self, mask: torch.Tensor, to: str | None = None) -> None:
        if not mask.any():
            return
        self.phase = torch.where(
            mask, torch.full_like(self.phase, self.index(to)) if to else (self.phase + 1).clamp(max=self.done_phase), self.phase
        )
        self.wait = torch.where(mask, torch.zeros_like(self.wait), self.wait)
        for env_id in mask.nonzero().flatten().tolist():
            self.phase_log[env_id].append((self.steps, self.PHASES[int(self.phase[env_id])]))

    def top_down(self, yaw: torch.Tensor) -> torch.Tensor:
        """Tool orientation for a vertical pinch whose fingers close across ``yaw``."""
        zero = torch.zeros_like(yaw)
        return quat_mul(quat_from_euler_xyz(zero, zero, yaw), self.rest)

    @staticmethod
    def yaw_of(quat: torch.Tensor) -> torch.Tensor:
        """Yaw of a body's +X axis, folded to +/-90 degrees for a two-finger pinch."""
        axis = torch.zeros(quat.shape[0], 3, device=quat.device)
        axis[:, 0] = 1.0
        x = quat_apply(quat, axis)
        yaw = torch.atan2(x[:, 1], x[:, 0])
        return torch.remainder(yaw + math.pi / 2, math.pi) - math.pi / 2

    def command(self, position, quat, grip, step=None, angle_step=0.012) -> torch.Tensor:
        """World target to one bounded delta action in the robot base frame."""
        env, arm = self.env, self.env.arm
        root = arm.robot.data.root_pose_w
        ee_b, ee_q = subtract_frame_transforms(root[:, :3], root[:, 3:7], arm.ee_pos, arm.ee_quat)
        target_b, target_q = subtract_frame_transforms(root[:, :3], root[:, 3:7], position, quat)
        delta, angle = compute_pose_error(ee_b, ee_q, target_b, target_q, rot_error_type="axis_angle")
        step = env.cfg.robot.ik_tracking_step if step is None else step
        action = torch.cat((bound(delta, step), bound(angle, angle_step)), dim=-1) / env.cfg.action_scale
        return torch.cat((action, grip.unsqueeze(-1)), dim=-1).clamp(-1.0, 1.0)

    # ---- rollout bookkeeping ------------------------------------------------
    @property
    def finished(self) -> bool:
        return bool(((self.phase == self.done_phase) | self.faulted).all())

    def observe(self, extras: dict) -> None:
        self.faulted |= extras["faulted"]
        # The env was reset under the rollout, so it can't continue.
        self.phase = torch.where(self.faulted, torch.full_like(self.phase, self.done_phase), self.phase)

    def mark_success(self, mask: torch.Tensor) -> None:
        new = mask & ~self.success
        self.success_step = torch.where(new, torch.full_like(self.success_step, self.steps), self.success_step)
        self.success |= mask

    def details(self, env_id: int) -> dict:
        return {}

    def summary(self) -> list[dict]:
        return [
            {
                "env": i,
                "success": bool(self.success[i]) and not bool(self.faulted[i]),
                "success_step": int(self.success_step[i]),
                "final_phase": self.PHASES[int(self.phase[i])],
                "faulted": bool(self.faulted[i]),
                "phases": self.phase_log[i],
                **self.details(i),
            }
            for i in range(self.env.num_envs)
        ]
