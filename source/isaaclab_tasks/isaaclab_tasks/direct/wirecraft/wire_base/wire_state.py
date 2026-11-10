# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Wire semantics that tasks consume, independent of the physics backend."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab.utils.math import quat_apply


@dataclass
class WireState:
    """World-frame state, shared by articulation and deformable wires.

    ``insertion_end`` is the port built at cable X=length and ``far_end`` the
    port at X=0. Backends resolve those names to their own bodies, so tasks
    never see segment numbers or deformable prim names.
    """

    insertion_end_pos: torch.Tensor
    insertion_end_quat: torch.Tensor
    far_end_pos: torch.Tensor
    far_end_quat: torch.Tensor
    tip_pos: torch.Tensor
    """Centre of the insertion port's leading face."""
    centerline: torch.Tensor
    """(num_envs, num_points, 3), equally spaced by arc length, far end first."""


def tip_from_root(root_pos: torch.Tensor, root_quat: torch.Tensor, tip_length: float) -> torch.Tensor:
    """Leading-face centre of a port from its shoulder-frame root pose."""
    offset = torch.zeros_like(root_pos)
    offset[:, 0] = tip_length
    return root_pos + quat_apply(root_quat, offset)


def resample_polyline(points: torch.Tensor, count: int) -> torch.Tensor:
    """Resample (N, K, 3) polylines to ``count`` equally spaced points."""
    segments = (points[:, 1:] - points[:, :-1]).norm(dim=-1)
    cumulative = torch.cat([torch.zeros_like(segments[:, :1]), segments.cumsum(dim=1)], dim=1)
    total = cumulative[:, -1:].clamp_min(1e-9)
    targets = torch.linspace(0.0, 1.0, count, device=points.device).unsqueeze(0) * total
    right = torch.searchsorted(cumulative.contiguous(), targets.contiguous()).clamp(1, points.shape[1] - 1)
    left = right - 1
    c0, c1 = cumulative.gather(1, left), cumulative.gather(1, right)
    weight = ((targets - c0) / (c1 - c0).clamp_min(1e-9)).unsqueeze(-1)
    index = lambda i: i.unsqueeze(-1).expand(-1, -1, 3)  # noqa: E731
    return torch.lerp(points.gather(1, index(left)), points.gather(1, index(right)), weight)


@dataclass
class FarAnchor:
    """Where a task wants the far port at reset, and how the cable leaves it.

    The cable runs ``straight`` metres from the far port along ``heading``, then turns by
    ``swing`` evenly over the next ``bend`` metres, and runs straight again.
    """

    pos: torch.Tensor
    """(N, 3) world position of the far port root."""
    heading: torch.Tensor
    """(N,) yaw of the far port's axis, pointing into the cable."""
    swing: torch.Tensor
    """(N,) turn of the rest of the cable relative to ``heading``."""
    straight: float
    bend: float

    def heading_at(self, x: torch.Tensor) -> torch.Tensor:
        """Yaw of the cable at cable coordinate ``x`` (N, M), measured from the far port."""
        share = ((x - self.straight) / self.bend).clamp(0.0, 1.0)
        return self.heading.unsqueeze(-1) + self.swing.unsqueeze(-1) * share

    def point_at(self, x: torch.Tensor) -> torch.Tensor:
        """(N, M, 2) horizontal position of the cable at cable coordinate ``x`` (N, M)."""
        heading, swing = self.heading.unsqueeze(-1), self.swing.unsqueeze(-1)
        start = x.clamp(max=self.straight)
        turned = (x - self.straight).clamp(0.0, self.bend)
        after = (x - self.straight - self.bend).clamp(min=0.0)
        angle = self.heading_at(x)
        # Arc over the bend, integrated in closed form; a vanishing swing is a straight run.
        small = swing.abs() < 1e-6
        rate = torch.where(small, torch.ones_like(swing), swing / self.bend)
        arc_x = torch.where(small, turned * heading.cos(), ((heading + rate * turned).sin() - heading.sin()) / rate)
        arc_y = torch.where(small, turned * heading.sin(), (heading.cos() - (heading + rate * turned).cos()) / rate)
        px = start * heading.cos() + arc_x + after * angle.cos()
        py = start * heading.sin() + arc_y + after * angle.sin()
        return self.pos[:, None, :2] + torch.stack((px, py), dim=-1)
