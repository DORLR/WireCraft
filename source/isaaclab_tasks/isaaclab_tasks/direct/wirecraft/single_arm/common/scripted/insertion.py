# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Pick the insertion port up by its housing and push it into the socket.

Geometry feedback only: after the grasp the measured port pose, not a fixed
transform, drives the alignment, which absorbs any slip inside the fingers.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import compute_pose_error, quat_apply, quat_inv, quat_mul

from .base import ScriptedPolicy, bound

STANDOFF = 0.012
"""Tip distance before the mouth at which alignment finishes."""
INSERT_STEP = 0.00025
"""Advance of the insertion reference per control step."""
REGRASPS = 2
JAM_LAG = 0.004
"""Tip this far behind the insertion reference means the port stubbed on the mouth."""
REINSERTS = 2
LIFT_WAIT = 240
"""Control steps after which a lift that cleared the table by 3 cm moves on."""


class InsertionPolicy(ScriptedPolicy):
    PHASES = ("SETTLE", "APPROACH", "DESCEND", "CLOSE", "LIFT", "PREALIGN", "ALIGN", "INSERT", "HOLD", "RELEASE", "DONE")

    def __init__(self, env):
        super().__init__(env)
        geometry = env.connector
        count = env.num_envs
        # The env counts a port as seated once its whole leading face is seat_depth in; aim 3 mm past that.
        self.required_depth = env.task.seat_depth
        self.final_depth = min(self.required_depth + 0.003, geometry.tip_length - 0.0002, geometry.socket_depth - 0.0002)
        self.mouth_to_root = geometry.socket_mouth_x - geometry.tip_length
        self.grasp_local = torch.zeros(count, 3, device=self.device)
        self.grasp_local[:, 0] = geometry.grasp_x
        self.hold_pos = torch.zeros(count, 3, device=self.device)
        self.hold_quat = torch.zeros(count, 4, device=self.device)
        self.hold_quat[:, 0] = 1.0
        self.start_z = None
        self.advance_steps = torch.zeros(count, device=self.device)
        self.regrasps = torch.zeros(count, dtype=torch.long, device=self.device)
        self.reinserts = torch.zeros(count, dtype=torch.long, device=self.device)
        self.bias = torch.zeros(count, 3, device=self.device)
        self.best_depth = torch.full((count,), -1.0, device=self.device)

    def act(self) -> torch.Tensor:
        env, arm, task = self.env, self.env.arm, self.env.task
        state = env.wire_state
        self.steps += 1
        self.wait += 1
        ee, ee_quat = arm.ee_pos, arm.ee_quat
        port, port_quat = state.insertion_end_pos, state.insertion_end_quat
        socket, socket_quat = task.socket_pos, task.socket_quat
        # The wire drops at reset, so its resting height is read only once it has settled.
        if self.start_z is None:
            self.start_z = port[:, 2].clone()
        self.start_z = torch.where(self.at("SETTLE"), port[:, 2], self.start_z)
        depth = task.tip_depth()
        self.best_depth = torch.maximum(self.best_depth, depth)

        grasp = port + quat_apply(port_quat, self.grasp_local)
        grasp[:, 2] += env.cfg.robot.grasp_z_offset
        turn = quat_mul(socket_quat, quat_inv(port_quat))
        # Either of two opposite pinches holds the port; take the one the turn to the socket leaves
        # within 90 degrees of the rest heading, or a port lying across the socket drives the wrist onto its limit.
        yaw = self.yaw_of(port_quat)
        heading = torch.remainder(yaw + self._yaw(turn) + math.pi, 2 * math.pi) - math.pi
        yaw = torch.where(heading.abs() > math.pi / 2, yaw - torch.sign(heading) * math.pi, yaw)
        pinch = self.top_down(yaw)
        offset = quat_apply(turn, ee - port)
        local = torch.zeros_like(port)
        local[:, 0] = self.mouth_to_root - STANDOFF
        inserting = self.at("INSERT")
        self.advance_steps += inserting.float()
        reference = torch.clamp(-STANDOFF + INSERT_STEP * self.advance_steps, max=self.final_depth)
        local[:, 0] = torch.where(inserting, self.mouth_to_root + reference, local[:, 0])
        desired = socket + quat_apply(socket_quat, local)
        aligned_pos, aligned_quat = desired + offset, quat_mul(turn, ee_quat)

        distance = torch.linalg.vector_norm(desired - port, dim=-1)
        _, angle = compute_pose_error(port, port_quat, desired, socket_quat, rot_error_type="axis_angle")
        angle = torch.linalg.vector_norm(angle, dim=-1)

        target, quat = ee.clone(), ee_quat.clone()
        grip = torch.full((env.num_envs,), -1.0, device=self.device)
        angle_step = 0.012

        above = grasp.clone()
        above[:, 2] += 0.10
        mask = self.at("APPROACH")
        target[mask], quat[mask] = above[mask], pinch[mask]
        mask = self.at("DESCEND")
        target[mask], quat[mask] = grasp[mask], pinch[mask]
        close = self.at("CLOSE")
        target[close], quat[close] = self.hold_pos[close], self.hold_quat[close]
        lift = self.at("LIFT")
        lifted = self.hold_pos.clone()
        lifted[:, 2] = torch.maximum(socket[:, 2] + 0.015, self.start_z + 0.07)
        target[lift], quat[lift] = lifted[lift], self.hold_quat[lift]
        align = self.at("PREALIGN") | self.at("ALIGN") | inserting
        target[align], quat[align] = aligned_pos[align], aligned_quat[align]
        hold = self.at("HOLD") | self.at("RELEASE") | self.at("DONE")
        target[hold], quat[hold] = self.hold_pos[hold], self.hold_quat[hold]
        grip[close | lift | align | self.at("HOLD")] = 1.0

        # A step too coarse for the arm's lag overshoots: on the way down it can fold the wrist onto its
        # limit above the port, and at the socket it swings the port back and forth across the mouth.
        # The lift keeps the coarse step: a short one can't pull the wire's load up to the target.
        fine = self.at("APPROACH") | self.at("DESCEND") | align
        coarse = env.cfg.robot.ik_tracking_step
        step = torch.where(fine, env.cfg.robot.fine_tracking_step or coarse, coarse).unsqueeze(-1)
        action = self.command(target, quat, grip, step, angle_step)
        action = self._load_bias(action, align, desired, port, socket_quat)

        near = torch.linalg.vector_norm(target - ee, dim=-1)
        self.advance(self.at("SETTLE") & (self.wait >= 45))
        self.advance(self.at("APPROACH") & (self.wait > 12) & (near < 0.003))
        seated = self.at("DESCEND") & (self.wait > 15) & (near < 0.0015)
        self.hold_pos = torch.where(seated.unsqueeze(-1), ee, self.hold_pos)
        self.hold_quat = torch.where(seated.unsqueeze(-1), ee_quat, self.hold_quat)
        self.advance(seated)
        closed = self.at("CLOSE") & (self.wait >= 120)
        # Fingers stopped on a coil of wire beside the port close on nothing: open and reach again.
        missed = closed & ~env.arm.is_grasping & (self.regrasps < REGRASPS)
        self.regrasps += missed.long()
        self.advance(missed, to="APPROACH")
        self.advance(closed & ~missed)
        # A small arm can fall short of a high socket under the wire's load; past LIFT_WAIT, clearing the
        # table is enough, and the alignment carries the port the rest of the way.
        lifted_enough = (self.wait > LIFT_WAIT) & (env.wire_state.insertion_end_pos[:, 2] > self.start_z + 0.03)
        self.advance(self.at("LIFT") & (((self.wait > 20) & (near < 0.003)) | lifted_enough))
        self.advance(self.at("PREALIGN") & (self.wait > 30) & (distance < 0.004))
        self.advance(self.at("ALIGN") & (self.wait > 35) & (distance < 0.001) & (angle < 0.006))
        # A port that stubs on the mouth falls behind the advancing reference: back out, align again and retry.
        jammed = inserting & (reference - depth > JAM_LAG) & (self.reinserts < REINSERTS)
        self.reinserts += jammed.long()
        self.advance_steps = torch.where(jammed, torch.zeros_like(self.advance_steps), self.advance_steps)
        self.advance(jammed, to="ALIGN")
        inserting = inserting & ~jammed
        inserted = inserting & task.seated & env.arm.is_grasping
        self.hold_pos = torch.where(inserted.unsqueeze(-1), ee, self.hold_pos)
        self.hold_quat = torch.where(inserted.unsqueeze(-1), ee_quat, self.hold_quat)
        self.advance(inserted)
        self.advance(self.at("HOLD") & (self.wait >= 60))
        self.advance(self.at("RELEASE") & (self.wait >= 60))
        return action

    @staticmethod
    def _yaw(quat: torch.Tensor) -> torch.Tensor:
        """Unfolded yaw of a rotation's +X axis."""
        axis = torch.zeros(quat.shape[0], 3, device=quat.device)
        axis[:, 0] = 1.0
        x = quat_apply(quat, axis)
        return torch.atan2(x[:, 1], x[:, 0])

    def _load_bias(self, action, align, desired, port, socket_quat):
        """Integrate out the steady error the cable load leaves on a proportional servo."""
        arm = self.env.arm
        base = quat_inv(arm.robot.data.root_quat_w)
        axis = torch.zeros_like(port)
        axis[:, 0] = 1.0
        axis = quat_apply(base, quat_apply(socket_quat, axis))
        error = quat_apply(base, desired - port)
        transverse = error - (error * axis).sum(-1, keepdim=True) * axis
        gate = (self.at("ALIGN") | self.at("INSERT")).unsqueeze(-1).float()
        self.bias = (self.bias + 0.04 * gate * transverse).clamp(-0.002, 0.002)
        action = action.clone()
        action[:, :3] += self.bias * align.unsqueeze(-1).float() / self.env.cfg.action_scale
        return action

    def observe(self, extras: dict) -> None:
        super().observe(extras)
        # Success comes from the env: the whole leading face 10 mm into the bore at some step.
        self.mark_success(extras["is_success"])

    def details(self, env_id: int) -> dict:
        return {
            "best_depth_mm": round(float(self.best_depth[env_id]) * 1000, 2),
            "required_depth_mm": round(self.required_depth * 1000, 2),
            "final_depth_mm": round(float(self.env.task.tip_depth()[env_id]) * 1000, 2),
            "regrasps": int(self.regrasps[env_id]),
            "reinserts": int(self.reinserts[env_id]),
        }
