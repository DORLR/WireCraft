# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Route the wire through each clip in two stages.

1. The port is picked up and carried past the clip beside it, a few centimetres off the slot line, and set
   down well beyond it: the wire behind it now runs along the clip with a gap to it. Holding the rigid port,
   the arm can drag the whole wire.
2. The wire is picked up just beside the clip (the env's wire holder keeps it between the fingers),
   lifted clear of the lips, centred on the slot and drawn down into it while still held, then let go.
   A wire that is not seated is picked up and put in again.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

from .wire_ops import CLOSE, OPEN, Op, WireOpsPolicy

LEAD_IN = 0.06
PORT_GAP = 0.08
"""Port root past the clip's end when it is set down: clear of the clip, with room for the wire pinch."""
PORT_SIDE_GAP = 0.03
"""The port passes the clip this far off the slot line, on the wire's side, so the wire is laid beside the
clip (not into it) and the second stage picks it up there and puts it in."""
PORT_HEIGHT = 0.015
"""Port root above the table when it is set down."""
PORT_CLEARANCE = 0.022
"""Port centre above the lip tops while it is carried past the clip."""
SEAT_HEIGHT = 0.005
"""Pinch above the table when the held wire is drawn into the slot."""
OVERSHOOT = 0.01
"""The port stroke runs this far past its set-down point and comes back, letting out the wire's stress."""
CLEARANCE = 0.010
"""Wire pinch above the lip tops before it is lowered."""
TRIM_TOLERANCE = 0.001
"""The lip gap clears the wire by 0.5 mm a side, so the wire has to be centred before it is lowered."""
TRIM_STEPS = 240
PORT_GRASP_SHIFT = 0.003
"""Routing pinches the port this much further toward its tip than the connector's ``grasp_x`` (the insertion
grasp, at the back edge of the housing), so less of each pad lands on the cable. Not further: the deformable
cable is embedded in the back 8 mm of the housing, and pads squeezing it there blow the cable up."""
PORT_ATTEMPTS = 2
"""Port picks per clip: a port that left the fingers on the way is picked up and carried again."""
ATTEMPTS = 4
"""Wire picks per clip: the first and the retries for a wire that did not seat."""
HOLD_STEPS = 600
"""Control steps, 10 s, the released wire has to stay seated for the rollout to count."""


class RoutingPolicy(WireOpsPolicy):
    def build(self) -> None:
        env = self.env
        self.task = env.task
        self.count = env.cfg.task.num_clips
        cfg = env.cfg.task
        self.clip_top = env.cfg.table_top_z + cfg.clip_base_height + cfg.clip_wall_height + cfg.clip_lip_height
        self.order = None
        self.wire_rank = 0
        self.grasp_local[:, 0] = env.connector.grasp_x + PORT_GRASP_SHIFT
        self.max_steps = 2500 + 6000 * self.count
        self.held = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        self.port_lost = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
        self.port_side = torch.ones(self.env.num_envs, device=self.device)
        self.engaged = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
        """The current attempt has the wire in hand, so it runs to the end even once the wire seats."""
        self.settle()
        # One clip at a time: a seated clip keeps the wire while the port goes on to the next.
        for rank in range(self.count):
            for attempt in range(PORT_ATTEMPTS):
                self.move_port(rank, f"{rank}_{attempt}", retry=attempt > 0)
            for attempt in range(ATTEMPTS):
                self.place_wire(rank, f"{rank}_{attempt}")
        self.finish()
        self.hold()

    def move_port(self, rank: int, tag: str, retry: bool) -> None:
        """Carry the port past the clip beside it and set it down beyond; a retry runs only where it was dropped."""
        start = len(self.ops)

        def side(mask, rank=rank):
            # Pass on the side the port starts on, so the wire is not dragged across the clip.
            position, _ = self._clip(rank)
            offset = ((self.env.wire_state.insertion_end_pos - position) * self.lateral(rank)).sum(-1)
            self.port_side = torch.where(mask, torch.where(offset < 0, -1.0, 1.0), self.port_side)
            return self.env.arm.ee_pos, self.env.arm.ee_quat, self.full(OPEN), torch.ones_like(mask)

        self.ops.append(Op(f"SIDE_{tag}", side))
        self.pick("insertion", f"PORT_{tag}", regrasp=True)
        self.carry(f"STROKE_{tag}", "insertion", lambda: self.port_stroke(rank), lambda: self.heading(rank))

        def mark(mask):
            self.port_lost = torch.where(mask, self.dropped, self.port_lost)
            return self.env.arm.ee_pos, self.env.arm.ee_quat, self.full(OPEN), torch.ones_like(mask)

        self.ops.append(Op(f"CHECK_PORT_{tag}", mark))
        self.release(f"PORT_{tag}")
        self.retreat(f"PORT_{tag}")
        if retry:
            for op in self.ops[start:]:
                op.run = self._only_if_port_lost(op.run)

    def _only_if_port_lost(self, run):
        def wrapped(mask):
            position, quat, grip, done = run(mask)
            return position, quat, grip, done | ~self.port_lost

        return wrapped

    def place_wire(self, rank: int, tag: str) -> None:
        """Pick the wire beside the clip, centre it over the slot, draw it in while held and let go."""
        start = len(self.ops)

        def select(mask, rank=rank):
            self.wire_rank = rank
            return self.env.arm.ee_pos, self.env.arm.ee_quat, self.full(OPEN), torch.ones_like(mask)

        self.ops.append(Op(f"SELECT_{tag}", select))
        self.pick("wire", f"WIRE_{tag}", height=0.0, regrasp=True, strict=True)
        held = len(self.ops)
        self.carry(f"OVER_{tag}", "wire", lambda: self.wire_spot(rank, self.clip_top + CLEARANCE).unsqueeze(1),
                   lambda: self.heading(rank))
        self.trim(rank, f"HIGH_{tag}")
        self.lower(tag)
        self.trim(rank, tag)
        self.release(f"WIRE_{tag}")
        self.retreat(f"WIRE_{tag}")
        # A seated wire skips the attempt, but only until the wire is in hand: once held it is let go first.
        self.ops[start + 1].run = self._engage(self.ops[start + 1].run, False)
        self.ops[start + 3].run = self._engage(self.ops[start + 3].run, True, rank)
        for op in self.ops[start:]:
            op.run = self._unless_seated(op.run, rank)
        # A wire that left the fingers is picked up again rather than steered.
        for op in self.ops[held:-2]:
            op.run = self._unless_dropped(op.run)

    def wire_grasp_point(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The wire where it crosses the pinch spot beside the current clip, and a frame along it."""
        spot = self.wire_spot(self.wire_rank, None)
        points = self.env.wire_state.centerline
        rows = torch.arange(points.shape[0], device=points.device)
        index = (points[..., :2] - spot[:, None, :2]).norm(dim=-1).argmin(dim=1).clamp(1, points.shape[1] - 2)
        before, after = points[rows, index - 1], points[rows, index + 1]
        heading = torch.atan2(after[:, 1] - before[:, 1], after[:, 0] - before[:, 0])
        zero = torch.zeros_like(heading)
        return points[rows, index], quat_from_euler_xyz(zero, zero, heading)

    def wire_spot(self, rank: int, height: float | None) -> torch.Tensor:
        """Point on the slot line just past the clip on the port side; ``height`` above the world origin, if given."""
        position, _ = self._clip(rank)
        spot = position + self.axis(rank) * (self.env.cfg.task.clip_depth / 2 + self.env.cfg.robot.wire_pinch_clearance)
        if height is not None:
            spot = spot.clone()
            spot[:, 2] = height + self.env.scene.env_origins[:, 2]
        return spot

    def _engage(self, run, value: bool, rank: int | None = None):
        """Set ``engaged`` for the envs running ``run``: cleared on approach, set once the fingers close.

        With ``rank``, an env whose wire is already seated in that clip is not engaged, so the retry skips
        the close instead of pinching air.
        """

        def wrapped(mask):
            wanted = mask
            if value and rank is not None:
                rows = torch.arange(self.env.num_envs, device=self.device)
                wanted = mask & ~self.task.seated()[rows, self._ranked()[:, rank]]
            self.engaged = torch.where(wanted, value, self.engaged)
            return run(mask)

        return wrapped

    def _unless_seated(self, run, rank: int):
        def wrapped(mask):
            position, quat, grip, done = run(mask)
            rows = torch.arange(self.env.num_envs, device=self.device)
            seated = self.task.seated()[rows, self._ranked()[:, rank]]
            return position, quat, grip, done | (seated & ~self.engaged)

        return wrapped

    def _unless_dropped(self, run):
        def wrapped(mask):
            position, quat, grip, done = run(mask)
            return position, quat, grip, done | self.dropped

        return wrapped

    def lower(self, tag: str) -> None:
        """Bring the held wire down without losing the trimmed sideways position."""

        def run(mask):
            low = self.env.cfg.table_top_z + SEAT_HEIGHT + self.env.scene.env_origins[:, 2]
            drop = self.port("wire")[0][:, 2] - low
            target = self.setpoint_pos.clone()
            target[:, 2] -= drop
            return target, self.setpoint_quat, self.full(CLOSE), (self.wait > 10) & (drop.abs() < 0.004)

        self.ops.append(Op(f"LOWER_{tag}", run))

    def hold(self) -> None:
        """Stand clear of the released wire and watch that the clips keep it."""

        def run(mask):
            arm = self.env.arm
            return arm.ee_pos, arm.ee_quat, self.full(OPEN), self.wait >= HOLD_STEPS

        self.ops.append(Op("HOLD", run, limit=HOLD_STEPS + 1))

    def trim(self, rank: int, tag: str) -> None:
        """Slide the held wire sideways until it is centred on the slot."""

        def run(mask):
            rows = torch.arange(self.env.num_envs, device=self.device)
            index = self._ranked()[:, rank]
            offsets = torch.stack([self.task.wire_offset(i) for i in range(self.task.cfg.num_clips)], dim=1)
            offset = offsets[rows, index]
            lateral = torch.zeros_like(self.setpoint_pos)
            lateral[:, 1] = 1.0
            lateral = quat_apply(self._clip(rank)[1], lateral)
            target = self.setpoint_pos - lateral * offset.unsqueeze(-1)
            done = ((self.wait > 20) & (offset.abs() < TRIM_TOLERANCE)) | (self.wait >= TRIM_STEPS)
            return target, self.setpoint_quat, self.full(CLOSE), done

        self.ops.append(Op(f"TRIM_{tag}", run))

    def _ranked(self) -> torch.Tensor:
        """Clip indices sorted from the far end outward, per env."""
        if self.order is None:
            start = self.env.wire_state.far_end_pos.unsqueeze(1)
            self.order = (self.task.clip_pos - start).norm(dim=-1).argsort(dim=1)
        return self.order

    def _clip(self, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
        rows = torch.arange(self.env.num_envs, device=self.device)
        index = self._ranked()[:, rank]
        return self.task.clip_pos[rows, index], self.task.clip_quat[rows, index]

    def clip_point(self, rank: int) -> torch.Tensor:
        return self._clip(rank)[0]

    def clip_yaw(self, rank: int) -> torch.Tensor:
        return self.yaw_of(self._clip(rank)[1]) + 1.5707963

    def axis(self, rank: int) -> torch.Tensor:
        """Slot direction, signed to point away from the far end."""
        position, quat = self._clip(rank)
        x = torch.zeros_like(position)
        x[:, 0] = 1.0
        x = quat_apply(quat, x)
        ahead = ((position - self.env.wire_state.far_end_pos) * x).sum(-1, keepdim=True)
        return torch.where(ahead < 0, -x, x)

    def heading(self, rank: int) -> torch.Tensor:
        axis = self.axis(rank)
        return torch.atan2(axis[:, 1], axis[:, 0])

    def lateral(self, rank: int) -> torch.Tensor:
        """Horizontal unit vector across the slot."""
        axis = self.axis(rank)
        lateral = torch.zeros_like(axis)
        lateral[:, 0], lateral[:, 1] = -axis[:, 1], axis[:, 0]
        return lateral / lateral.norm(dim=-1, keepdim=True).clamp_min(1e-9)

    def port_stroke(self, rank: int) -> torch.Tensor:
        """The held port past the clip ``PORT_SIDE_GAP`` off the slot line, a little past and back
        (``OVERSHOOT``), then down onto the table well beyond it: the wire is laid beside the clip."""
        position, _ = self._clip(rank)
        position = position + self.lateral(rank) * (PORT_SIDE_GAP * self.port_side).unsqueeze(-1)
        axis = self.axis(rank)
        origin_z = self.env.scene.env_origins[:, 2]
        above = (self.clip_top + PORT_CLEARANCE + origin_z).unsqueeze(1)
        low = (self.env.cfg.table_top_z + PORT_HEIGHT + origin_z).unsqueeze(1)
        past = position + axis * (self.env.cfg.task.clip_depth / 2 + PORT_GAP)
        beyond = past + axis * OVERSHOOT
        points = torch.stack((position - axis * LEAD_IN, position, beyond, past, past), dim=1).clone()
        points[:, :4, 2] = above
        points[:, 0, 2] += 0.04
        points[:, 4, 2] = low[:, 0]
        return points

    def observe(self, extras: dict) -> None:
        super().observe(extras)
        self.env_success |= extras["is_success"]
        # Judged after the final settle; a wire merely resting near a clip at reset doesn't count.
        seated = extras["clips_seated"] == self.count
        self.held &= seated | ~self.at("HOLD")
        self.mark_success(extras["is_success"] & seated & self.held & self.at("DONE"))
        self.last_routed, self.last_seated = extras["clips_routed"], extras["clips_seated"]

    def details(self, env_id: int) -> dict:
        return {
            "clips": self.count,
            "routed": int(self.last_routed[env_id]),
            "seated": int(self.last_seated[env_id]),
            "held_10s": bool(self.held[env_id]),
            "env_success": bool(self.env_success[env_id]),
            "regrasps": int(self.regrasps[env_id]),
            "wire_holds": int(self.env.wire_holder.engagements[env_id]) if self.env.wire_holder is not None else None,
            "slot_z_mm": [round(float(self.task.slot_fit(i)[0][env_id]) * 1000, 1) for i in range(self.count)],
            "slot_y_mm": [round(float(self.task.slot_fit(i)[1][env_id]) * 1000, 1) for i in range(self.count)],
        }
