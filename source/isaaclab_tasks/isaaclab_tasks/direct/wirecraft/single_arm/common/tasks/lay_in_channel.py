# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Lay the wire into a chain of straight or curved channel modules."""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim.spawners.channels import ChannelCfg
from isaaclab.sim.spawners.channels.channel_spawner import _channel_centerline
from isaaclab.utils import configclass

from ....wire_base import FarAnchor
from ..env_cfg import PRIM_PATH

PRINTED_MODULE_LENGTH = {"straight": 0.110, "curved": 0.066}
"""The channel spawner's default modules, in metres."""


def chain_layout(cfg: ChannelCfg, count: int) -> tuple[list[tuple[float, float, float]], list[tuple[float, float]]]:
    """Join modules at their flat end faces, not at nominal arc tangents.

    An end cap is square to its last chord, so consecutive roots differ by the
    heading between the first and last chords: 9.75 degrees for the
    66 mm / 10 degree / 40 chord module, not 10.

    Returns root poses ``(x, y, yaw)`` and the chain centerline, in a frame that
    starts at the first centerline point heading along +X.
    """
    local = _channel_centerline(cfg)
    first = math.atan2(local[1][1] - local[0][1], local[1][0] - local[0][0])
    last = math.atan2(local[-1][1] - local[-2][1], local[-1][0] - local[-2][0])
    turn = last - first
    poses, line = [], []
    yaw, x, y = -first, 0.0, 0.0
    for index in range(count):
        cos, sin = math.cos(yaw), math.sin(yaw)
        root = (x - cos * local[0][0] + sin * local[0][1], y - sin * local[0][0] - cos * local[0][1])
        poses.append((*root, yaw))
        world = [(root[0] + cos * px - sin * py, root[1] + sin * px + cos * py) for px, py in local]
        line.extend(world if index == 0 else world[1:])
        x, y = world[-1]
        yaw += turn
    return poses, line


class LayInChannelTask:
    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg.task

    def spawn(self) -> None:
        self.modules = [RigidObject(c) for c in self.cfg.module_cfgs]
        for index, module in enumerate(self.modules):
            self.env.scene.rigid_objects[f"channel_{index}"] = module

    def post_clone(self) -> None:
        if self.cfg.anchor_far_port:
            self.env.wire_backend.pin_far_end()

    def initialize(self) -> None:
        env, cfg = self.env, self.cfg
        self.base_line = torch.tensor(cfg.line_xy, device=env.device)
        self.offset = torch.zeros(env.num_envs, 2, device=env.device)
        self.headings = torch.tensor([p[2] for p in cfg.module_poses], device=env.device).repeat(env.num_envs, 1)
        self.floor_top = env.cfg.table_top_z + cfg.wall_thickness
        self.wall_top = self.floor_top + cfg.depth

    def line_w(self) -> torch.Tensor:
        """World-frame channel centerline, (num_envs, points, 2)."""
        start = torch.tensor(self.cfg.start_xy, device=self.env.device)
        return self.base_line + start + (self.offset + self.env.scene.env_origins[:, :2]).unsqueeze(1)

    def update(self) -> None:
        points = self.env.wire_state.centerline
        line = self.line_w()
        start, delta = line[:, :-1], line[:, 1:] - line[:, :-1]
        relative = points[:, :, None, :2] - start[:, None]
        t = ((relative * delta[:, None]).sum(-1) / (delta * delta).sum(-1)[:, None]).clamp(0.0, 1.0)
        lateral = (relative - t.unsqueeze(-1) * delta[:, None]).norm(dim=-1).amin(dim=-1)
        z = points[..., 2] - self.env.scene.env_origins[:, None, 2]
        inside = (lateral <= self.cfg.width / 2) & (z >= self.env.cfg.table_top_z) & (z <= self.wall_top)
        self.inside = inside
        self.lateral, self.height = lateral, z - self.wall_top
        self.in_channel_fraction = inside.float().mean(dim=1)
        self.in_channel = self.in_channel_fraction > self.cfg.success_fraction

    def observations(self) -> torch.Tensor:
        env, arm = self.env, self.env.arm
        to_target = env.port_local_pos - env.ee_local_pos
        return torch.cat((
            arm.joint_pos, arm.joint_vel, env.ee_local_pos, env.ee_local_quat, to_target,
            self.in_channel_fraction.unsqueeze(-1), self.offset, self.headings,
        ), dim=-1)

    def success(self) -> torch.Tensor:
        self.env.extras["in_channel_fraction"] = self.in_channel_fraction
        return self.in_channel

    def rewards(self) -> torch.Tensor:
        env, cfg = self.env, self.cfg
        distance = -torch.norm(env.port_local_pos - env.ee_local_pos, p=2, dim=-1)
        return cfg.distance_reward_scale * distance + cfg.channel_reward_scale * self.in_channel_fraction

    def reset(self, env_ids: torch.Tensor) -> None:
        env, cfg = self.env, self.cfg
        if cfg.randomize_position:
            half = torch.tensor((cfg.x_range, cfg.y_range), device=env.device)
            self.offset[env_ids] = (torch.rand(len(env_ids), 2, device=env.device) * 2.0 - 1.0) * half
        for module in self.modules:
            state = module.data.default_root_state[env_ids].clone()
            state[:, :2] += self.offset[env_ids]
            state[:, :3] += env.scene.env_origins[env_ids]
            module.write_root_state_to_sim(state, env_ids=env_ids)
            module.reset(env_ids)
        return self._far_port_pose(env_ids) if cfg.anchor_far_port else None

    def _far_port_pose(self, env_ids: torch.Tensor) -> FarAnchor:
        """Far port just outside the chain's first end, along the chain, with the rest of the wire swung off the slot.

        As in the original lay-in-channel task, the far port keeps the chain's axis, so the
        wire's first stretch points down the slot; only the rest of the wire is turned.
        """
        env, cfg = self.env, self.cfg
        count = len(env_ids)
        line = self.line_w()[env_ids]
        along = line[:, 1] - line[:, 0]
        along = along / along.norm(dim=-1, keepdim=True)
        across = torch.stack((-along[:, 1], along[:, 0]), dim=-1)
        entry = torch.empty(count, 3, device=env.device)
        entry[:, :2] = line[:, 0] + along * cfg.far_entry_offset[0] + across * cfg.far_entry_offset[1]
        entry[:, 2] = env.cfg.table_top_z + cfg.far_entry_height[env.cfg.wire.kind] + env.scene.env_origins[env_ids, 2]
        low, high = cfg.wire_yaw_range
        swing = low + (high - low) * torch.rand(count, device=env.device)
        sign = torch.where(torch.rand(count, device=env.device) < 0.5, -1.0, 1.0)
        heading = torch.atan2(along[:, 1], along[:, 0])
        return FarAnchor(entry, heading, sign * swing, cfg.far_straight, cfg.far_bend)


@configclass
class LayInChannelCfg:
    task_class: type = LayInChannelTask
    name: str = "lay_in_channel"

    channel_type: str = "straight"
    num_modules: int = 0
    """Zero keeps the chain near the 0.33 m legacy length: 3 straight or 5 curved modules."""
    start_xy: tuple[float, float] = (0.255, 0.0)
    """First centerline point; the chain leaves it along +X, the way the wire lies at reset."""
    module_length: float | None = None
    """None uses the printable 0.110/0.066 m modules, times ``module_scale``."""
    module_scale: float = 1.0
    """Shrinks the default modules for a smaller arm; the slot cross-section keeps the connector's size."""
    width: float = 0.010
    depth: float = 0.010
    wall_thickness: float = 0.005
    bend_angle_deg: float = 10.0
    curve_segments: int = 40
    success_fraction: float = 0.8

    anchor_far_port: bool = True
    """Start the far port at the chain's entry, clamped where the wire backend can hold it."""
    far_entry_offset: tuple[float, float] = (-0.025, 0.0)
    """Far port from the first centerline point, along the chain and across it.

    Close to the entry leaves little wire outside the chain; closer still, the port itself sits on the entry.
    """
    far_entry_height: dict[str, float] = {"articulation": 0.017, "deformable": 0.020}
    """Far port above the table, per wire. Set any higher, the wire's first stretch from the held port stays
    above the wall tops (the trained articulation cable is soft: at 24 mm its first 5-14 points did)."""
    far_straight: float = 0.06
    """Wire from the far port laid along the chain before it turns."""
    far_bend: float = 0.10
    """Wire over which it turns by the swing."""
    scripted_swing: dict[str, dict[str, float]] = {}
    """Per wire kind, fields of the scripted policy's ``Swing`` to replace. Its defaults draw a 0.4 m
    wire about 20 % past its length, which a smaller arm can't pull a shorter wire to."""
    wire_yaw_range: tuple[float, float] = (0.20, 0.45)
    """Size of the swing at reset, either way. The original task drew it from +-0.45; below 0.2
    the wire starts in the slot and the task is already done."""

    randomize_position: bool = True
    x_range: float = 0.1
    y_range: float = 0.1
    distance_reward_scale: float = 1.0
    channel_reward_scale: float = 1.0

    # Derived by resolve.
    module_cfgs: list = []
    module_poses: list = []
    line_xy: list = []

    def observation_dim(self, joints: int) -> int:
        return 2 * joints + 3 + 4 + 3 + 1 + 2 + len(self.module_poses)

    def resolve(self, env_cfg) -> None:
        count = self.num_modules or (3 if self.channel_type == "straight" else 5)
        spawn = ChannelCfg(
            channel_type=self.channel_type,
            length=self.module_length or PRINTED_MODULE_LENGTH[self.channel_type] * self.module_scale,
            width=self.width,
            depth=self.depth,
            wall_thickness=self.wall_thickness,
            bend_angle_deg=self.bend_angle_deg,
            curve_segments=self.curve_segments,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        )
        self.module_poses, self.line_xy = chain_layout(spawn, count)
        cx, cy = self.start_xy
        self.module_cfgs = [
            RigidObjectCfg(
                prim_path=f"{PRIM_PATH}/channel_{i}",
                spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(cx + x, cy + y, env_cfg.table_top_z),
                    rot=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
                ),
            )
            for i, (x, y, yaw) in enumerate(self.module_poses)
        ]
