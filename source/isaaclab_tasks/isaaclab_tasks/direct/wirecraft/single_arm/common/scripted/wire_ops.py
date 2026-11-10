# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Pick, carry and press operations shared by routing and channel policies.

A policy is a flat list of named operations. Each operation returns a world
target, a gripper command and a per-env completion mask, so envs walk the same
program at their own pace from their own reset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from isaaclab.utils.math import apply_delta_pose, compute_pose_error, quat_apply, quat_from_euler_xyz, quat_mul

from .base import ScriptedPolicy, bound

OPEN, CLOSE = -1.0, 1.0
DESCEND_STEPS = 150
CARRY_LIMIT = 1500
REGRASPS = 2
"""Times a pick that closed on nothing opens and reaches for the port again."""
LOST_STEPS = 30
"""Control steps without a grasp after which a carry stops chasing a port it no longer holds."""
WIRE_GRASP_BACK = 0.03
"""Where a wire pick pinches the cable: this far along it from the insertion port, meters."""


@dataclass
class Op:
    name: str
    run: Callable
    """``run(mask) -> (position, quat, grip, done)``; all tensors span every env."""
    free: bool = False
    """The tool holds nothing and touches nothing, so its setpoint may move ``FREE_FACTOR`` times faster."""
    limit: int | None = None
    """Control steps before the operation gives way; ``None`` uses ``OP_LIMIT``."""


class WireOpsPolicy(ScriptedPolicy):
    """Runs ``self.ops``; subclasses build the list in ``build()``."""

    TRAVEL_SPEED = 0.0008
    """Setpoint motion per control step, in metres."""
    TURN_SPEED = 0.004
    """Setpoint rotation per control step, in radians."""
    FREE_FACTOR = 3.0
    RAMP_STEPS = 60
    SERVO_STEP = 0.03
    SERVO_ANGLE = 0.06
    """Largest command toward the setpoint. Commands are relative to the measured pose, so
    this, not the setpoint speed, sets how hard the arm holds its path against the wire."""
    OP_LIMIT = 600
    """Control steps after which a blocked operation gives way to the next one."""
    max_steps = 6000

    def __init__(self, env):
        self.ops: list[Op] = []
        self.env = env
        self.device = env.device
        self.grasp_local = torch.zeros(env.num_envs, 3, device=env.device)
        self.grasp_local[:, 0] = env.connector.grasp_x
        self.anchor_pos = torch.zeros(env.num_envs, 3, device=env.device)
        self.anchor_quat = torch.zeros(env.num_envs, 4, device=env.device)
        self.path_index = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.grasp_heading = torch.zeros(env.num_envs, device=env.device)
        self.lost_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.dropped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.lost_detail = torch.zeros(env.num_envs, 6, device=env.device)
        self.regrasps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        """The last carry ended because the port had left the fingers; cleared by the next pick."""
        wire = env.cfg.wire
        self.wire_radius = wire.radius if wire.kind == "articulation" else wire.fem_radius
        self.setpoint_pos = env.arm.ee_pos.clone()
        self.setpoint_quat = env.arm.ee_quat.clone()
        self.build()
        self.ops.append(Op("DONE", self._done))
        type(self).PHASES = tuple(op.name for op in self.ops)
        super().__init__(env)
        self.env_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def build(self) -> None:
        raise NotImplementedError

    def _done(self, mask):
        arm = self.env.arm
        return arm.ee_pos, arm.ee_quat, self.full(OPEN), torch.zeros_like(mask)

    def full(self, value: float) -> torch.Tensor:
        return torch.full((self.env.num_envs,), value, device=self.device)

    # ---- geometry ----------------------------------------------------------
    def port(self, end: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Pose of what a pick on ``end`` holds: a port, or for ``"wire"`` the pinched cable point.

        The cable point's +X runs along the wire away from the insertion port. While the env's
        wire holder has the wire, that is the holder's pose, which moves with the tool.
        """
        state = self.env.wire_state
        if end == "insertion":
            return state.insertion_end_pos, state.insertion_end_quat
        if end == "wire":
            holder = self.env.wire_holder
            position, quat = self.wire_grasp_point()
            if holder is not None:
                held, held_pos, held_quat = holder.engaged, *holder.pose()
            else:
                # Without the ring, a wire between the closed fingers sits at the tool point.
                arm = self.env.arm
                held, held_pos = self.holding("wire"), arm.ee_pos
                zero = torch.zeros_like(held_pos[:, 0])
                held_quat = quat_from_euler_xyz(zero, zero, self.heading_of(arm.ee_quat))
            held = held.unsqueeze(-1)
            return torch.where(held, held_pos, position), torch.where(held, held_quat, quat)
        return state.far_end_pos, state.far_end_quat

    def wire_grasp_point(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Where a pick on ``"wire"`` pinches the cable; policies may choose another spot."""
        return self.wire_point(WIRE_GRASP_BACK)

    def wire_point(self, back: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Centerline point ``back`` meters along the cable from the insertion port, and a yaw-only frame along it."""
        points = self.env.wire_state.centerline
        # Order the points from the insertion port, whichever end of the polyline that is.
        port = self.env.wire_state.insertion_end_pos.unsqueeze(1)
        reverse = (points[:, -1:] - port).norm(dim=-1) < (points[:, :1] - port).norm(dim=-1)
        points = torch.where(reverse.unsqueeze(-1), points.flip(1), points)
        length = torch.cat((torch.zeros_like(points[:, :1, 0]), (points[:, 1:] - points[:, :-1]).norm(dim=-1)), 1).cumsum(1)
        index = (length < back).sum(dim=1).clamp(1, points.shape[1] - 1)
        rows = torch.arange(points.shape[0], device=points.device)
        before, after = points[rows, index - 1], points[rows, index]
        share = ((back - length[rows, index - 1]) / (length[rows, index] - length[rows, index - 1]).clamp_min(1e-9))
        position = before + share.clamp(0.0, 1.0).unsqueeze(-1) * (after - before)
        heading = torch.atan2(after[:, 1] - before[:, 1], after[:, 0] - before[:, 0])
        zero = torch.zeros_like(heading)
        return position, quat_from_euler_xyz(zero, zero, heading)

    def holding(self, end: str) -> torch.Tensor:
        """The pick on ``end`` still has it: fingers on the port, or the wire holder engaged."""
        if end == "wire":
            if self.env.wire_holder is not None:
                return self.env.wire_holder.holding
            # Jaws resting at the wire's width have it; jaws that shut all the way lost it.
            low, high = self.env.cfg.robot.wire_pinch_opening
            arm = self.env.arm
            return (arm.opening > low) & (arm.opening < high)
        return self.env.arm.is_grasping

    @staticmethod
    def heading_of(quat: torch.Tensor) -> torch.Tensor:
        """Unfolded yaw of a body's +X axis."""
        axis = torch.zeros(quat.shape[0], 3, device=quat.device)
        axis[:, 0] = 1.0
        x = quat_apply(quat, axis)
        return torch.atan2(x[:, 1], x[:, 0])

    @staticmethod
    def _wrap(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(angle.sin(), angle.cos())

    def near(self, position: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(position - self.env.arm.ee_pos, dim=-1)

    # ---- operations --------------------------------------------------------
    def settle(self, steps: int = 45) -> None:
        def run(mask):
            arm = self.env.arm
            return arm.ee_pos, arm.ee_quat, self.full(OPEN), self.wait >= steps

        self.ops.append(Op("SETTLE", run))

    def pick(self, end: str, label: str, slack: float = 0.0, height: float = 0.07, regrasp: bool = False,
             strict: bool = False) -> None:
        """Top-down pinch on a port's housing, then a lift.

        ``slack`` moves the lift that far toward the other port, for a wire
        that is already taut and can't follow a vertical lift. ``regrasp`` reopens and reaches again,
        up to ``REGRASPS`` times, when the closed jaws hold nothing. ``strict`` closes only once the tool is
        within 3 mm of the grasp point: a wire pinched a few millimetres too high slips out under the pads.
        """

        def grasp_pose():
            port, quat = self.port(end)
            if end == "wire":
                # A wire on the table: the pinch height comes from the table, not from the cable's own z.
                position = port.clone()
                floor = self.env.cfg.table_top_z + self.env.scene.env_origins[:, 2] + self.wire_radius
                position[:, 2] = torch.maximum(position[:, 2], floor) + self.env.cfg.robot.wire_grasp_z_offset
            else:
                position = port + quat_apply(quat, self.grasp_local)
                position = position.clone()
                position[:, 2] += self.env.cfg.robot.grasp_z_offset
            return position, self.top_down(self.yaw_of(quat))

        def approach(mask):
            self.dropped &= ~mask
            position, quat = grasp_pose()
            position[:, 2] += 0.10
            return position, quat, self.full(OPEN), (self.wait > 12) & (self.near(position) < 0.004)

        def descend(mask):
            position, quat = grasp_pose()
            near = self.near(position)
            # Close to the base the IK settles a little short, so a nearly seated tool has to do.
            # A strict pinch (on the wire) closes within 3 mm: the wire moves as the fingers reach it, and a pinch
            # up to 2 mm above its centre still holds it between the pads (measured on the deformable wire).
            done = (self.wait > 15) & (near < (0.003 if strict else 0.0015))
            if not strict:
                done |= (self.wait >= DESCEND_STEPS) & (near < 0.005)
            done |= self.wait >= self.OP_LIMIT - 1
            self._snapshot(done, self.env.arm.ee_pos, self.env.arm.ee_quat)
            return position, quat, self.full(OPEN), done

        def close(mask):
            done = self.wait >= 120
            # Jaws that stopped beside the port, on the wire or on a wall close on nothing: open and reach again.
            missed = mask & done & ~self.holding(end) & (self.regrasps < REGRASPS) & regrasp
            if missed.any():
                self.regrasps += missed.long()
                self.advance(missed, to=f"APPROACH_{label}")
                done = done & ~missed
            self.grasp_heading = torch.where(done, self.heading_of(self.port(end)[1]), self.grasp_heading)
            return self.anchor_pos, self.anchor_quat, self.full(CLOSE), done

        def lift(mask):
            target = self.anchor_pos.clone()
            target[:, 2] += height
            if slack:
                other, _ = self.port("far" if end == "insertion" else "insertion")
                toward = other[:, :2] - self.anchor_pos[:, :2]
                target[:, :2] += slack * toward / toward.norm(dim=-1, keepdim=True)
            return target, self.anchor_quat, self.full(CLOSE), (self.wait > 20) & (self.near(target) < 0.005)

        self.ops.append(Op(f"APPROACH_{label}", approach, free=True))
        for name, run in (("DESCEND", descend), ("CLOSE", close), ("LIFT", lift)):
            self.ops.append(Op(f"{name}_{label}", run))

    def _snapshot(self, mask, position, quat) -> None:
        self.anchor_pos = torch.where(mask.unsqueeze(-1), position, self.anchor_pos)
        self.anchor_quat = torch.where(mask.unsqueeze(-1), quat, self.anchor_quat)

    def carry(self, label: str, end: str, path: Callable, heading: Callable | None = None, tolerance: float = 0.004) -> None:
        """Bring the held port along ``path()``, an (N, K, 3) path, one waypoint at a time.

        The measured port pose closes the loop, so slip inside the fingers doesn't
        shift the wire. The loop only means anything while the port is held: once it
        has left the fingers the carry ends, or the arm would chase the growing offset.

        ``heading()`` turns the held port to that yaw about the vertical; the
        wire leaves the port along its axis, so this sets the wire's direction.
        """

        def run(mask):
            points = path()
            count = points.shape[1]
            rows = torch.arange(points.shape[0], device=self.device)
            index = self.path_index.clamp(max=count - 1)
            port, port_quat = self.port(end)
            arm = self.env.arm
            offset, quat = arm.ee_pos - port, self.anchor_quat
            if heading is not None:
                wanted = heading()
                zero = torch.zeros_like(wanted)
                # The tool stays upright: its yaw is set from the grasp pose, never from itself.
                quat = quat_mul(quat_from_euler_xyz(zero, zero, self._wrap(wanted - self.grasp_heading)), self.anchor_quat)
                offset = quat_apply(quat_from_euler_xyz(zero, zero, self._wrap(wanted - self.heading_of(port_quat))), offset)
            target = points[rows, index] + offset
            reached = torch.linalg.vector_norm(points[rows, index] - port, dim=-1) < tolerance
            self.path_index = torch.where(mask & reached, self.path_index + 1, self.path_index)
            held = torch.where(self.holding(end), torch.zeros_like(self.lost_steps), self.lost_steps + 1)
            self.lost_steps = torch.where(mask, held, self.lost_steps)
            lost = self.lost_steps >= LOST_STEPS
            newly = mask & lost & ~self.dropped
            if newly.any():
                # Why the hold was judged lost: jaw opening, port and cable force per pad, tool to port distance.
                values = torch.stack((arm.opening, arm.force_left, arm.force_right, arm.cable_force_left,
                                      arm.cable_force_right, torch.linalg.vector_norm(arm.ee_pos - port, dim=-1)), -1)
                self.lost_detail = torch.where(newly.unsqueeze(-1), values, self.lost_detail)
            self.dropped |= mask & lost
            done = (self.path_index >= count) | lost
            self.lost_steps = torch.where(mask & done, torch.zeros_like(self.lost_steps), self.lost_steps)
            self.path_index = torch.where(mask & done, torch.zeros_like(self.path_index), self.path_index)
            return target, quat, self.full(CLOSE), done

        self.ops.append(Op(f"CARRY_{label}", run, limit=CARRY_LIMIT))

    def release(self, label: str, steps: int = 400) -> None:
        def run(mask):
            arm = self.env.arm
            done = arm.is_released | (self.wait >= steps)
            first = mask & (self.wait <= 1)
            self._snapshot(first, arm.ee_pos, arm.ee_quat)
            return self.anchor_pos, self.anchor_quat, self.full(OPEN), done

        self.ops.append(Op(f"RELEASE_{label}", run))

    def retreat(self, label: str, height: float = 0.10) -> None:
        def run(mask):
            target = self.anchor_pos.clone()
            target[:, 2] = self.env.cfg.table_top_z + height + self.env.scene.env_origins[:, 2]
            return target, self.anchor_quat, self.full(OPEN), (self.wait > 10) & (self.near(target) < 0.01)

        self.ops.append(Op(f"RETREAT_{label}", run, free=True))

    def press(self, label: str, point: Callable, depth_z: Callable, yaw: Callable, skip: Callable | None = None) -> None:
        """Close the empty gripper and push the wire down at ``point()``, unless ``skip()`` says it's in."""

        def skipped() -> torch.Tensor:
            return skip() if skip is not None else torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)

        lift = self.env.cfg.robot.press_z_offset

        def over(mask):
            target = point().clone()
            target[:, 2] = depth_z() + lift + 0.04
            # Fingers still closing would straddle the wire and pinch it instead.
            done = (self.wait > 20) & (self.near(target) < 0.004) & self.env.arm.is_fully_closed
            return target, self.top_down(yaw()), self.full(CLOSE), done | skipped()

        def down(mask):
            target = point().clone()
            target[:, 2] = depth_z() + lift
            reached = (self.wait > 40) & (self.near(target) < 0.003)
            return target, self.top_down(yaw()), self.full(CLOSE), reached | (self.wait > 220) | skipped()

        def up(mask):
            target = point().clone()
            target[:, 2] = depth_z() + lift + 0.05
            return target, self.top_down(yaw()), self.full(CLOSE), ((self.wait > 10) & (self.near(target) < 0.006)) | skipped()

        for name, run, free in (("OVER", over, True), ("DOWN", down, False), ("UP", up, True)):
            self.ops.append(Op(f"PRESS_{name}_{label}", run, free))

    def finish(self) -> None:
        """Open fully and let the wire settle; tasks are judged after this."""

        def run(mask):
            arm = self.env.arm
            return arm.ee_pos, arm.ee_quat, self.full(OPEN), (self.wait >= 90) & arm.is_released

        self.ops.append(Op("FINAL_SETTLE", run))

    # ---- execution ---------------------------------------------------------
    def act(self) -> torch.Tensor:
        self.steps += 1
        self.wait += 1
        arm = self.env.arm
        position, quat, grip = arm.ee_pos.clone(), arm.ee_quat.clone(), self.full(OPEN)
        finished = torch.zeros_like(self.success)
        scale = torch.ones(self.env.num_envs, 1, device=self.device)
        for index, op in enumerate(self.ops):
            mask = self.phase == index
            if not mask.any():
                continue
            p, q, g, done = op.run(mask)
            position[mask], quat[mask], grip[mask] = p[mask], q[mask], g[mask]
            if op.free:
                scale[mask] = self.FREE_FACTOR
            if op is not self.ops[-1]:
                finished |= mask & (done | (self.wait >= (op.limit or self.OP_LIMIT)))
        self.advance(finished)
        # A move starts gently: a sudden pull on the wire can twist the port out of the pinch.
        scale = scale * (self.wait.float() / self.RAMP_STEPS).clamp(0.15, 1.0).unsqueeze(-1)
        delta, angle = compute_pose_error(self.setpoint_pos, self.setpoint_quat, position, quat, rot_error_type="axis_angle")
        step = torch.cat((bound(delta, self.TRAVEL_SPEED * scale), bound(angle, self.TURN_SPEED * scale)), dim=-1)
        self.setpoint_pos, self.setpoint_quat = apply_delta_pose(self.setpoint_pos, self.setpoint_quat, step)
        return self.command(self.setpoint_pos, self.setpoint_quat, grip, self.SERVO_STEP, self.SERVO_ANGLE)
