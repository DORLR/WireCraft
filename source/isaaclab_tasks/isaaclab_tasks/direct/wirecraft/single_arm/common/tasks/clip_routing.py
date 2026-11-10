# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Route the wire through one to three clips, then let go."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim.spawners.ports import ProceduralClipCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_inv, quat_mul

from ..env_cfg import PRIM_PATH


class ClipRoutingTask:
    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg.task

    def spawn(self) -> None:
        self.clips = [RigidObject(c) for c in self.cfg.clip_cfgs]
        for index, clip in enumerate(self.clips):
            self.env.scene.rigid_objects[f"clip_{index}"] = clip

    def post_clone(self) -> None:
        pass

    def initialize(self) -> None:
        pass

    def update(self) -> None:
        self.clip_pos = torch.stack([c.data.root_pos_w for c in self.clips], dim=1)
        self.clip_quat = torch.stack([c.data.root_quat_w for c in self.clips], dim=1)

    def points_in_clip(self, index: int) -> torch.Tensor:
        """Centerline points in clip ``index``'s local frame, (num_envs, points, 3)."""
        points = self.env.wire_state.centerline
        count = points.shape[1]
        relative = points - self.clip_pos[:, index].unsqueeze(1)
        inverse = quat_inv(self.clip_quat[:, index]).unsqueeze(1).expand(-1, count, -1)
        return quat_apply(inverse.reshape(-1, 4), relative.reshape(-1, 3)).view_as(points)

    def routed(self) -> torch.Tensor:
        """(num_envs, num_clips) bool: the wire passes through each clip's window."""
        cfg, hits = self.cfg, []
        for index in range(len(self.clips)):
            local = self.points_in_clip(index)
            inside = (
                (local[..., 0].abs() <= cfg.clip_depth * 1.5)
                & (local[..., 1].abs() <= cfg.clip_opening_width * 1.5)
                & (local[..., 2] >= cfg.clip_base_height - 0.004)
                & (local[..., 2] <= cfg.clip_wall_height + 0.01)
            )
            hits.append(inside.any(dim=1))
        return torch.stack(hits, dim=1)

    def seated(self) -> torch.Tensor:
        """(num_envs, num_clips) bool: the wire sits below the lips. Reported, not rewarded."""
        cfg, hits = self.cfg, []
        for index in range(len(self.clips)):
            local = self.points_in_clip(index)
            inside = (
                (local[..., 0].abs() <= cfg.clip_depth / 2)
                & (local[..., 1].abs() <= cfg.clip_opening_width / 2)
                & (local[..., 2] <= cfg.clip_base_height + cfg.clip_wall_height)
            )
            hits.append(inside.any(dim=1))
        return torch.stack(hits, dim=1)

    def slot_fit(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Lowest wire height and smallest |y| over the slot's X span, for diagnostics."""
        local = self.points_in_clip(index)
        span = local[..., 0].abs() <= self.cfg.clip_depth
        z = torch.where(span, local[..., 2], torch.full_like(local[..., 2], 9.0)).amin(dim=1)
        y = torch.where(span, local[..., 1].abs(), torch.full_like(local[..., 1], 9.0)).amin(dim=1)
        return z, y

    def wire_offset(self, index: int) -> torch.Tensor:
        """Wire's lateral offset from the slot where it crosses the clip's mid plane, (num_envs,)."""
        local = self.points_in_clip(index)
        x = local[..., 0]
        crossing = (x[:, :-1] * x[:, 1:] <= 0).float().argmax(dim=1, keepdim=True)
        x0, x1 = x.gather(1, crossing), x.gather(1, crossing + 1)
        y0, y1 = local[..., 1].gather(1, crossing), local[..., 1].gather(1, crossing + 1)
        share = x0 / (x0 - x1).where(x0 != x1, torch.ones_like(x0))
        return (y0 + share * (y1 - y0)).squeeze(1)

    def observations(self) -> torch.Tensor:
        env, arm = self.env, self.env.arm
        to_target = env.port_local_pos - env.ee_local_pos
        return torch.cat((arm.joint_pos, arm.joint_vel, env.ee_local_pos, env.ee_local_quat, to_target), dim=-1)

    def success(self) -> torch.Tensor:
        """Every clip holds the wire and the gripper has released it."""
        routed = self.routed()
        self.env.extras["clips_routed"] = routed.sum(dim=1)
        self.env.extras["clips_seated"] = self.seated().sum(dim=1)
        return routed.all(dim=1) & self.env.arm.is_released

    def rewards(self) -> torch.Tensor:
        env = self.env
        return -self.cfg.distance_reward_scale * torch.norm(env.port_local_pos - env.ee_local_pos, p=2, dim=-1)

    def reset(self, env_ids: torch.Tensor) -> None:
        env, cfg = self.env, self.cfg
        count = len(env_ids)
        for clip in self.clips:
            state = clip.data.default_root_state[env_ids].clone()
            state[:, :3] += env.scene.env_origins[env_ids]
            if cfg.randomize_clips:
                state[:, 1] += (torch.rand(count, device=env.device) * 2.0 - 1.0) * cfg.clip_y_range
                zeros = torch.zeros(count, device=env.device)
                yaw = (torch.rand(count, device=env.device) * 2.0 - 1.0) * cfg.clip_yaw_range
                state[:, 3:7] = quat_mul(quat_from_euler_xyz(zeros, zeros, yaw), state[:, 3:7])
            clip.write_root_state_to_sim(state, env_ids=env_ids)
            clip.reset(env_ids)


@configclass
class ClipRoutingCfg:
    task_class: type = ClipRoutingTask
    name: str = "clip_routing"

    num_clips: int = 1
    clip_x_positions: tuple[float, float, float] = (0.50, 0.65, 0.35)
    """Index 0 is first so a single clip sits within reach of the wire."""
    distance_reward_scale: float = 1.0

    # Deep and long enough to wrap the wire over a long hold; the top gap is the difficulty knob.
    clip_opening_width: float = 0.015
    clip_wall_width: float = 0.004
    clip_wall_height: float = 0.016
    clip_depth: float = 0.03
    clip_base_height: float = 0.004
    clip_lip_height: float = 0.004
    clip_top_gap: float = 0.009
    """Y-gap between the lips, meters. Must exceed the wire diameter to press it in."""
    clip_color: tuple[float, float, float] = (0.1, 0.1, 0.8)

    randomize_clips: bool = True
    clip_y_range: float = 0.06
    clip_yaw_range: float = 0.78539816

    # Derived by resolve.
    clip_cfgs: list = []

    def observation_dim(self, joints: int) -> int:
        return 2 * joints + 3 + 4 + 3

    def resolve(self, env_cfg) -> None:
        if not 1 <= self.num_clips <= len(self.clip_x_positions):
            raise ValueError(f"task.num_clips must be in 1..{len(self.clip_x_positions)}.")
        wire_diameter = 2 * (env_cfg.wire.radius if env_cfg.wire.kind == "articulation" else env_cfg.wire.fem_radius)
        if self.clip_top_gap <= wire_diameter:
            raise ValueError(f"task.clip_top_gap {self.clip_top_gap} leaves no way past a {wire_diameter} m wire.")
        self.clip_cfgs = [
            RigidObjectCfg(
                prim_path=f"{PRIM_PATH}/clip_{i}",
                spawn=ProceduralClipCfg(
                    opening_width=self.clip_opening_width,
                    wall_width=self.clip_wall_width,
                    wall_height=self.clip_wall_height,
                    clip_depth=self.clip_depth,
                    base_height=self.clip_base_height,
                    lip_height=self.clip_lip_height,
                    lip_overhang=(self.clip_opening_width - self.clip_top_gap) / 2.0,
                    color=tuple(self.clip_color),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(self.clip_x_positions[i], 0.0, env_cfg.table_top_z), rot=(1.0, 0.0, 0.0, 0.0)
                ),
            )
            for i in range(self.num_clips)
        ]
