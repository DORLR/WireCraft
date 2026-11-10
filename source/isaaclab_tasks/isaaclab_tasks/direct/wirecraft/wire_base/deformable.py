# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""FEM wire backend with two rigid ports held by vertex attachments."""

from __future__ import annotations

import torch

from isaaclab.assets import DeformableObject, DeformableObjectCfg, RigidObject, RigidObjectCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul

from .wire_state import FarAnchor, WireState, resample_polyline, tip_from_root


class DeformableWire:
    """``start_port`` is the far end (cable X=0), ``end_port`` the insertion end (X=length)."""

    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg.wire
        self.tip_length = env.connector.tip_length
        self._pinned = False

    def spawn(self) -> None:
        root = self.env.cfg.wire_prim_path
        assembly = self.env.cfg.wire_spawn
        assembly.func(root, assembly, translation=tuple(self.cfg.root_pos))
        self.wire = DeformableObject(DeformableObjectCfg(prim_path=f"{root}/wire", spawn=None))
        self.start_port = RigidObject(RigidObjectCfg(prim_path=f"{root}/start_port", spawn=None))
        self.end_port = RigidObject(RigidObjectCfg(prim_path=f"{root}/end_port", spawn=None))
        scene = self.env.scene
        scene.deformable_objects["wire"] = self.wire
        scene.rigid_objects["wire_start_port"] = self.start_port
        scene.rigid_objects["wire_end_port"] = self.end_port

    def pin_far_end(self) -> None:
        self._pinned = True

    def hold_far_end(self) -> None:
        """Keep the far port at its reset position; it turns freely, as a hinge, with the wire.

        It stays dynamic: the wire's attachment doesn't follow a kinematic port on the GPU.
        Writing back the reset position, not the current one, stops wire tension ratcheting it away.
        """
        if not self._pinned:
            return
        state = self.start_port.data.root_state_w.clone()
        state[:, :3] = self._held_far
        state[:, 7:10] = 0.0
        self.start_port.write_root_state_to_sim(state)

    def initialize(self) -> None:
        # Rest-pose nodes sorted along the cable give a stable centerline order.
        rest = self.wire.data.default_nodal_state_w[0, :, 0]
        bins = torch.linspace(rest.min(), rest.max(), self.env.cfg.wire_spawn.wire.num_segments + 1, device=rest.device)
        index = torch.bucketize(rest, bins[1:-1])
        self._bin_weights = torch.nn.functional.one_hot(index, bins.numel() - 1).float()
        self._bin_weights = self._bin_weights[:, self._bin_weights.sum(0) > 0]
        self._bin_weights = self._bin_weights / self._bin_weights.sum(0, keepdim=True)
        # The ports are spawned with the assembly, so their init_state holds no pose.
        self._rest_ports = [port.data.root_state_w[:, :7].clone() for port in (self.start_port, self.end_port)]
        self._held_far = self._rest_ports[0][:, :3].clone()
        # Rest frame of the cable, for laying it along a bent path at reset.
        far, end = self._rest_ports[0][:, :3], self._rest_ports[1][:, :3]
        axis = end[:, :2] - far[:, :2]
        self._rest_axis = axis / axis.norm(dim=-1, keepdim=True)
        self._rest_heading = torch.atan2(self._rest_axis[:, 1], self._rest_axis[:, 0])

    def _cable_coordinate(self, env_ids: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Distance along the rest cable from the far port, for rest-pose ``points`` (N, M, 3)."""
        far = self._rest_ports[0][env_ids, :2].unsqueeze(1)
        return ((points[..., :2] - far) * self._rest_axis[env_ids].unsqueeze(1)).sum(-1)

    def _bend(self, env_ids: torch.Tensor, points: torch.Tensor, anchor: FarAnchor) -> torch.Tensor:
        """Carry rest-pose ``points`` (N, M, 3) onto the anchor's bent path, keeping their offsets from the axis."""
        far = self._rest_ports[0][env_ids, :3].unsqueeze(1)
        axis = self._rest_axis[env_ids].unsqueeze(1)
        relative = points - far
        x = (relative[..., :2] * axis).sum(-1)
        side = relative[..., 1] * axis[..., 0] - relative[..., 0] * axis[..., 1]
        heading = anchor.heading_at(x)
        out = torch.empty_like(points)
        out[..., :2] = anchor.point_at(x) + side.unsqueeze(-1) * torch.stack((-heading.sin(), heading.cos()), dim=-1)
        out[..., 2] = anchor.pos[:, 2:3] + relative[..., 2]
        return out

    def state(self) -> WireState:
        end, start = self.end_port.data, self.start_port.data
        bins = torch.einsum("nkd,kb->nbd", self.wire.data.nodal_pos_w, self._bin_weights)
        return WireState(
            insertion_end_pos=end.root_pos_w,
            insertion_end_quat=end.root_quat_w,
            far_end_pos=start.root_pos_w,
            far_end_quat=start.root_quat_w,
            tip_pos=tip_from_root(end.root_pos_w, end.root_quat_w, self.tip_length),
            centerline=resample_polyline(bins, self.cfg.centerline_points),
        )

    def cable_contact_paths(self) -> list[str]:
        """None: PhysX filtered contact reports cover rigid bodies only, not the FEM cable."""
        return []

    def port_speed(self) -> torch.Tensor:
        """Faster port's angular speed in rad/s; NaN once a port's state is lost."""
        end, start = self.end_port.data, self.start_port.data
        speed = torch.maximum(end.root_ang_vel_w.norm(dim=-1), start.root_ang_vel_w.norm(dim=-1))
        finite = torch.isfinite(end.root_pos_w).all(-1) & torch.isfinite(start.root_pos_w).all(-1)
        return torch.where(finite, speed, torch.nan)

    def reset(self, env_ids: torch.Tensor, anchor: FarAnchor | None = None) -> None:
        """Place the wire: randomly on the table, or bent away from a far port at ``anchor``."""
        nodal = self.wire.data.default_nodal_state_w[env_ids].clone()
        ports = []
        for rest in self._rest_ports:
            state = torch.zeros(len(env_ids), 13, device=self.env.device)
            state[:, :7] = rest[env_ids]
            ports.append(state)
        if anchor is None:
            xy, turn, lift = self.env.sample_wire_pose(len(env_ids))
            pivot = nodal[..., :3].mean(dim=1)
            target = pivot.clone()
            target[:, :2] += xy
            target[:, 2] += lift
            count = nodal.shape[1]
            offsets = nodal[..., :3] - pivot.unsqueeze(1)
            turned = quat_apply(turn.unsqueeze(1).expand(-1, count, -1).reshape(-1, 4), offsets.reshape(-1, 3))
            nodal[..., :3] = turned.view(len(env_ids), count, 3) + target.unsqueeze(1)
            for state in ports:
                state[:, :3] = quat_apply(turn, state[:, :3] - pivot) + target
                state[:, 3:7] = quat_mul(turn, state[:, 3:7])
        else:
            nodal[..., :3] = self._bend(env_ids, nodal[..., :3], anchor)
            for state in ports:
                position = state[:, :3].clone()
                state[:, :3] = self._bend(env_ids, position.unsqueeze(1), anchor)[:, 0]
                x = self._cable_coordinate(env_ids, position.unsqueeze(1))
                zeros = torch.zeros_like(anchor.heading)
                turn = quat_from_euler_xyz(zeros, zeros, anchor.heading_at(x)[:, 0] - self._rest_heading[env_ids])
                state[:, 3:7] = quat_mul(turn, state[:, 3:7])
        nodal[..., 3:] = 0.0
        self._held_far[env_ids] = ports[0][:, :3]
        self.wire.write_nodal_state_to_sim(nodal, env_ids)
        self.wire.reset(env_ids)
        for port, state in zip((self.start_port, self.end_port), ports):
            port.write_root_state_to_sim(state, env_ids=env_ids)
            port.reset(env_ids)
