# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Capsule-chain wire backend."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.sim.spawners.wires import build_wire_articulation_cfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, quat_mul

from .wire_state import FarAnchor, WireState, resample_polyline, tip_from_root

HOLD_STIFFNESS = 0.5
HOLD_DAMPING = 0.5
"""Fractions of the deadbeat gains, which stay stable for any port mass."""
GRAVITY = 9.81


class ArticulationWire:
    """Ports and segments are links of one articulation joined by fixed end joints."""

    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg.wire
        self.tip_length = env.connector.tip_length
        self._pinned = False

    def spawn(self) -> None:
        assembly = self.env.cfg.wire_spawn
        cfg = build_wire_articulation_cfg(assembly.wire, self.env.cfg.wire_prim_path)
        cfg.spawn = None
        cfg.init_state.pos = tuple(self.cfg.root_pos)
        assembly.func(self.env.cfg.wire_prim_path, assembly, translation=tuple(self.cfg.root_pos))
        self.wire = Articulation(cfg)
        self.env.scene.articulations["wire"] = self.wire

    def pin_far_end(self) -> None:
        self._pinned = True

    def hold_far_end(self) -> None:
        """Pull the far port to its reset pose with a mass-scaled spring.

        A rigid clamp plus a pinch on the other end holds the chain at both ends, and joint
        velocity then doubles every step until the solve turns NaN. A spring can give.
        """
        if not self._pinned:
            return
        data = self.wire.data
        dt = self.env.physics_dt
        error = self._held_far - data.body_pos_w[:, self._start]
        velocity = data.body_lin_vel_w[:, self._start]
        force = self._far_mass * (HOLD_STIFFNESS * error / dt**2 - HOLD_DAMPING * velocity / dt)
        force[:, 2] += self._far_mass[:, 0] * GRAVITY
        # External forces are given in the body frame.
        force = quat_apply_inverse(data.body_quat_w[:, self._start], force).unsqueeze(1)
        self.wire.set_external_force_and_torque(force, torch.zeros_like(force), body_ids=[self._start])

    def initialize(self) -> None:
        names = self.wire.body_names
        self._start = self.wire.find_bodies("start_port")[0][0]
        self._end = self.wire.find_bodies("end_port")[0][0]
        count = self.env.cfg.wire_spawn.wire.num_segments
        # Order by the actual names; PhysX link order is not chain order.
        self._chain = [self._start, *[names.index(f"seg_{i}") for i in range(count)], self._end]
        # PhysX picks the root link, so read its spawned pose; init_state is the prim origin only.
        self._rest_root = self.wire.data.root_state_w[:, :7].clone()
        self._rest_root[:, :3] -= self.env.scene.env_origins
        self._rest_far = self.wire.data.body_pos_w[:, self._start] - self.env.scene.env_origins
        self._held_far = self.wire.data.body_pos_w[:, self._start].clone()
        self._far_mass = self.wire.root_physx_view.get_masses()[:, self._start : self._start + 1].to(self.env.device)
        # Cable coordinates of the links, for laying the chain along a bent path at reset.
        rest = self.wire.data.body_pos_w[:, self._chain] - self.env.scene.env_origins.unsqueeze(1)
        axis = rest[:, -1, :2] - rest[:, 0, :2]
        axis = axis / axis.norm(dim=-1, keepdim=True)
        self._rest_heading = torch.atan2(axis[:, 1], axis[:, 0])
        self._cable_x = ((rest[..., :2] - rest[:, :1, :2]) * axis.unsqueeze(1)).sum(-1)
        self._cable_z = rest[..., 2] - rest[:, :1, 2]
        root = (self.wire.data.body_pos_w - self.wire.data.root_pos_w.unsqueeze(1)).norm(dim=-1)[0].argmin()
        self._root_link = self._chain.index(int(root))
        # The bending DOF about the vertical of the joint between link k-1 and link k (k counted along the chain).
        names = self.wire.joint_names
        self._yaw_dofs = {k + 1: names.index(f"joint_seg_{k}_to_parent:2") for k in range(1, count)}

    def state(self) -> WireState:
        data = self.wire.data
        body_pos, body_quat = data.body_pos_w, data.body_quat_w
        insertion_pos, insertion_quat = body_pos[:, self._end], body_quat[:, self._end]
        return WireState(
            insertion_end_pos=insertion_pos,
            insertion_end_quat=insertion_quat,
            far_end_pos=body_pos[:, self._start],
            far_end_quat=body_quat[:, self._start],
            tip_pos=tip_from_root(insertion_pos, insertion_quat, self.tip_length),
            centerline=resample_polyline(body_pos[:, self._chain[1:-1]], self.cfg.centerline_points),
        )

    def cable_contact_paths(self) -> list[str]:
        """Cable links a finger can touch besides the insertion port, for contact sensor filters."""
        root = self.env.cfg.wire_prim_path
        return [f"{root}/seg_{i}" for i in range(self.env.cfg.wire_spawn.wire.num_segments)]

    def port_speed(self) -> torch.Tensor:
        """Faster port's angular speed in rad/s; NaN once a port's state is lost."""
        data = self.wire.data
        ports = [self._start, self._end]
        speed = data.body_ang_vel_w[:, ports].norm(dim=-1).amax(dim=-1)
        return torch.where(torch.isfinite(data.body_pos_w[:, ports]).all(-1).all(-1), speed, torch.nan)

    def reset(self, env_ids: torch.Tensor, anchor: FarAnchor | None = None) -> None:
        """Place the wire: randomly on the table, or bent away from a far port at ``anchor``."""
        if anchor is not None:
            self._reset_anchored(env_ids, anchor)
            return
        root = torch.zeros(len(env_ids), 13, device=self.env.device)
        rest = self._rest_root[env_ids]
        xy, turn, lift = self.env.sample_wire_pose(len(env_ids))
        # Rotate about the cable centre so the wire stays over the same table region.
        pivot = torch.tensor(self.cfg.root_pos, device=self.env.device)
        pivot = pivot + torch.tensor((self.cfg.length / 2, 0.0, 0.0), device=self.env.device)
        target = pivot + self.env.scene.env_origins[env_ids]
        target[:, :2] += xy
        target[:, 2] += lift
        root[:, :3] = target + quat_apply(turn, rest[:, :3] - pivot)
        root[:, 3:7] = quat_mul(turn, rest[:, 3:7])
        self.wire.write_root_state_to_sim(root, env_ids=env_ids)
        joint_pos = self.wire.data.default_joint_pos[env_ids]
        self.wire.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        self.wire.reset(env_ids)

    def _reset_anchored(self, env_ids: torch.Tensor, anchor: FarAnchor) -> None:
        """Walk the chain out from the far port, each link turned to the path's heading at its centre.

        Only the root pose and the joint angles can be written, so the root is placed where
        this walk puts it and the bend goes into the joints between the links.
        """
        x = self._cable_x[env_ids]
        heading = anchor.heading_at(x)
        # Ports are fixed to their segments.
        heading[:, 0], heading[:, -1] = heading[:, 1], heading[:, -2]
        points = [anchor.pos[:, :2]]
        for i in range(1, x.shape[1]):
            joint = (x[:, i - 1] + x[:, i]) / 2
            before = torch.stack((heading[:, i - 1].cos(), heading[:, i - 1].sin()), dim=-1)
            after = torch.stack((heading[:, i].cos(), heading[:, i].sin()), dim=-1)
            points.append(points[-1] + before * (joint - x[:, i - 1]).unsqueeze(-1) + after * (x[:, i] - joint).unsqueeze(-1))
        link = self._root_link
        root = torch.zeros(len(env_ids), 13, device=self.env.device)
        root[:, :2] = points[link]
        root[:, 2] = anchor.pos[:, 2] + self._cable_z[env_ids, link]
        zeros = torch.zeros_like(anchor.heading)
        turn = quat_from_euler_xyz(zeros, zeros, heading[:, link] - self._rest_heading[env_ids])
        root[:, 3:7] = quat_mul(turn, self._rest_root[env_ids, 3:7])
        self.wire.write_root_state_to_sim(root, env_ids=env_ids)
        joint_pos = self.wire.data.default_joint_pos[env_ids].clone()
        for k, dof in self._yaw_dofs.items():
            joint_pos[:, dof] = heading[:, k] - heading[:, k - 1]
        self.wire.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        self.wire.reset(env_ids)
        self._held_far[env_ids] = anchor.pos
