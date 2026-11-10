# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Lay the wire into the channel chain.

The far port is clamped at the chain's entry, with the wire swung off the slot. Only
the other port is picked: it is carried beyond the chain's far end, as far from the
clamp as the wire is long, so the wire comes down straight over the slot and drops
in. Closed fingertips then press it below the wall tops along the run.

This completes a straight chain. On a curved chain a released wire springs back
toward the chord and rests on the walls; see the README's known limits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from .wire_ops import WireOpsPolicy

PORT_LOW = 0.012
"""Port root above the table at release. Pressing a held port into the table diverges the wire."""


@dataclass(frozen=True)
class Swing:
    """How a wire is brought over the chain, in metres."""

    slack: float
    """How far inside its reach the port travels."""
    height: float
    """Port height above the table while it travels."""
    set_slack: float
    """How far inside its reach the port is set down."""
    lift: float
    """Lift after the grasp."""
    points: int
    """Waypoints of the arc about the far port, from the port's start bearing to the chain's line."""


SWING = {
    # A chain can't stretch, so it travels a little inside its reach; held much slacker it bows
    # and won't drop into the slot.
    "articulation": Swing(slack=0.02, height=0.10, set_slack=0.008, lift=0.07, points=6),
    # The soft wire sticks to the table, so it is drawn tight until its middle lifts clear of the
    # walls, and it is set down still tight, since any slack arches up out of the slot.
    "deformable": Swing(slack=-0.04, height=0.20, set_slack=-0.01, lift=0.07, points=6),
}
RUN_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.97)
"""Press stations along the chain."""
SEATED_MARGIN = 0.05
"""In-channel fraction above the success line at which pressing stops: fingers pressing a seated
chain gain nothing and can make it diverge."""


class ChannelPolicy(WireOpsPolicy):
    def build(self) -> None:
        self.task = self.env.task
        self.line = None
        self.wire_length = None
        self.start_fraction = None
        self.swing = None
        self.fraction = torch.zeros(self.env.num_envs, device=self.device)
        self.settle()
        kind = self.env.cfg.wire.kind
        self.plan = replace(SWING[kind], **self.env.cfg.task.scripted_swing.get(kind, {}))
        self.pick("insertion", "END", slack=self.plan.slack, height=self.plan.lift, regrasp=True)
        self.carry("END", "insertion", self.park, self.outward_yaw)
        self.release("END")
        self.retreat("END")
        self.press_along("RUN", RUN_FRACTIONS)
        self.finish()

    def centerline(self) -> torch.Tensor:
        """(N, K, 2) line ordered from the end nearest the far port."""
        line = self.task.line_w()
        if self.line is None:
            far = self.env.wire_state.far_end_pos[:, :2]
            self.line = (line[:, 0] - far).norm(dim=-1) > (line[:, -1] - far).norm(dim=-1)
        return torch.where(self.line[:, None, None], line.flip(1), line)

    def _z(self, height: float) -> torch.Tensor:
        return self.env.cfg.table_top_z + height + self.env.scene.env_origins[:, 2]

    def _outward(self, line: torch.Tensor) -> torch.Tensor:
        vector = line[:, -1] - line[:, -4]
        return vector / vector.norm(dim=-1, keepdim=True)

    def outward_yaw(self) -> torch.Tensor:
        out = self._outward(self.centerline())
        return torch.atan2(out[:, 1], out[:, 0])

    def park(self) -> torch.Tensor:
        """Over to the chain's line about the far port, then down beyond its last end."""
        line = self.centerline()
        far = self.env.wire_state.far_end_pos[:, :2]
        out = self._outward(line)
        heading = torch.atan2(out[:, 1], out[:, 0])
        if self.swing is None:
            port = self.env.wire_state.insertion_end_pos[:, :2] - far
            self.swing = self._wrap(torch.atan2(port[:, 1], port[:, 0]) - heading)
        share = torch.linspace(1.0, 0.0, self.plan.points, device=self.device)
        angle = heading.unsqueeze(1) + self.swing.unsqueeze(1) * share
        radius = (self.wire_length - self.plan.slack).view(-1, 1, 1)
        arc = far.unsqueeze(1) + radius * torch.stack((angle.cos(), angle.sin()), dim=-1)
        spot = (far + out * (self.wire_length - self.plan.set_slack).unsqueeze(-1)).unsqueeze(1)
        path = torch.cat((arc, spot, spot), dim=1)
        z = self._z(self.plan.height).view(-1, 1, 1).expand(-1, path.shape[1], 1).clone()
        z[:, -1] = self._z(PORT_LOW).unsqueeze(-1)
        return torch.cat((path, z), dim=-1)

    def press_along(self, label: str, fractions: tuple[float, ...]) -> None:
        for point, fraction in enumerate(fractions):
            self.press(
                f"{label}_{point}",
                lambda fraction=fraction: self._press_point(fraction),
                lambda: self._z(self.task.wall_top - self.env.cfg.table_top_z + 0.002),
                lambda fraction=fraction: self._press_yaw(fraction),
                self._seated,
            )

    def _seated(self) -> torch.Tensor:
        return self.fraction > self.env.cfg.task.success_fraction + SEATED_MARGIN

    def _along(self, fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Point at ``fraction`` of the chain's arc length and the unit tangent there, per env."""
        line = self.centerline()
        pieces = line[:, 1:] - line[:, :-1]
        lengths = pieces.norm(dim=-1)
        ends = lengths.cumsum(dim=1)
        reach = ends[:, -1:] * fraction
        index = (ends < reach).sum(dim=1, keepdim=True).clamp(max=lengths.shape[1] - 1)
        rows = torch.arange(line.shape[0], device=self.device)
        piece, length = pieces[rows, index[:, 0]], lengths.gather(1, index)
        start = ends.gather(1, index) - length
        return line[rows, index[:, 0]] + piece * ((reach - start) / length), piece / length

    def _press_point(self, fraction: float) -> torch.Tensor:
        point, _ = self._along(fraction)
        return torch.cat((point, self._z(0.0).unsqueeze(-1)), dim=-1)

    def _press_yaw(self, fraction: float) -> torch.Tensor:
        _, tangent = self._along(fraction)
        # Across the slot: a wire running along the gap between the fingers wedges in and gets pinched.
        return torch.atan2(tangent[:, 1], tangent[:, 0]) + math.pi / 2

    def observe(self, extras: dict) -> None:
        super().observe(extras)
        self.env_success |= extras["is_success"]
        self.fraction = extras["in_channel_fraction"]
        if self.wire_length is None:
            # Along the wire as spawned, which starts bent: the deformable wire draws in once it drops.
            state = self.env.wire_state
            path = torch.cat((state.far_end_pos.unsqueeze(1), state.centerline, state.insertion_end_pos.unsqueeze(1)), dim=1)
            self.wire_length = (path[:, 1:] - path[:, :-1]).norm(dim=-1).sum(dim=1)
        # Taken once the wire has settled: what the rollout started from.
        if self.start_fraction is None and self.steps >= 40:
            self.start_fraction = self.fraction.clone()
        self.mark_success(extras["is_success"] & self.env.arm.is_released & self.at("DONE"))

    def details(self, env_id: int) -> dict:
        return {
            "in_channel_fraction": round(float(self.fraction[env_id]), 3),
            "start_fraction": round(float(self.start_fraction[env_id]), 3),
            "required_fraction": self.env.cfg.task.success_fraction,
            "env_success": bool(self.env_success[env_id]),
            "regrasps": int(self.regrasps[env_id]),
            "carry_lost": dict(zip(("opening", "port_N_left", "port_N_right", "cable_N_left", "cable_N_right", "ee_to_port_m"),
                                   (round(float(v), 4) for v in self.lost_detail[env_id]))) if bool(self.dropped[env_id]) else None,
            # Centerline points outside at the end, from the far port: index, mm off the slot axis, mm above the walls.
            "outside_points": [
                (i, round(float(self.task.lateral[env_id, i]) * 1000, 1), round(float(self.task.height[env_id, i]) * 1000, 1))
                for i in (~self.task.inside[env_id]).nonzero().flatten().tolist()
            ],
        }
