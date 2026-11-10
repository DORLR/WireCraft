# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Insert the wire's insertion-end port into the paired socket.

The reward is the v14.5 staged insertion reward the released policies were trained on, plus a
per-step bonus for holding a seated port. Frames follow ``wire_base.connectors``: the port root is
the tip's rear shoulder with the tip along +X, and the socket's +X points into the bore.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_inv, quat_mul

from ..env_cfg import PRIM_PATH


class InsertionTask:
    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg.task

    def spawn(self) -> None:
        env, cfg = self.env, self.cfg
        self.socket = RigidObject(cfg.socket_cfg)
        self.backboard = RigidObject(cfg.backboard_cfg)
        env.scene.rigid_objects["socket"] = self.socket
        env.scene.rigid_objects["backboard"] = self.backboard

    def post_clone(self) -> None:
        pass

    def initialize(self) -> None:
        env, cfg = self.env, self.cfg
        device, count, geometry = env.device, env.num_envs, env.connector
        self.seat_depth = min(cfg.seat_depth, geometry.socket_depth - 0.002, geometry.tip_length - 0.002)
        """Depth past the mouth every leading-face vertex needs to count as seated."""
        self.face = torch.tensor([(geometry.tip_length, y, z) for y, z in geometry.tip_outline], device=device)
        self.hole = torch.tensor(geometry.hole, device=device)
        self.best_insert_depth = torch.zeros(count, device=device)
        self.seated = torch.zeros(count, dtype=torch.bool, device=device)
        self.inserted = torch.zeros(count, dtype=torch.bool, device=device)
        """Seated at some step of the current episode: the success flag."""
        unit = torch.zeros(count, 3, device=device)
        self.x_axis, self.y_axis = unit.clone(), unit.clone()
        self.x_axis[:, 0] = 1.0
        self.y_axis[:, 1] = 1.0
        tilt = torch.full((count,), env.cfg.robot.grasp_tilt, device=device)
        zero = torch.zeros_like(tilt)
        self.tilt_b = quat_from_euler_xyz(zero, tilt, zero)
        self.rest_w = torch.tensor(env.cfg.robot.grasp_rest_quat, device=device).repeat(count, 1)

    def update(self) -> None:
        env, cfg = self.env, self.cfg
        self.socket_pos = self.socket.data.root_pos_w
        self.socket_quat = self.socket.data.root_quat_w
        self.socket_local_pos = self.socket_pos - env.scene.env_origins
        offset = self.x_axis * cfg.seated_target_offset_x
        self.seated_target = self.socket_local_pos + quat_apply(self.socket_quat, offset)

    def tip_in_socket(self) -> torch.Tensor:
        """Tip position in the socket root frame, for diagnostics and policies."""
        return quat_apply(quat_inv(self.socket_quat), self.env.wire_state.tip_pos - self.socket_pos)

    def tip_depth(self) -> torch.Tensor:
        """Depth of the leading face's centre past the socket mouth along insertion, in meters."""
        return self.tip_in_socket()[:, 0] - self.env.connector.socket_mouth_x

    def is_seated(self) -> torch.Tensor:
        """The whole leading face is in the bore: every vertex at least ``seat_depth`` past the mouth and
        before the back, inside the bore cross-section, with the port pointing in.

        A pose test on the procedural geometry, not a contact certificate.
        """
        env, tolerance = self.env, 1.0e-5
        state = env.wire_state
        count, vertices = env.num_envs, self.face.shape[0]
        inverse = quat_inv(self.socket_quat)
        face_w = quat_apply(state.insertion_end_quat.repeat_interleave(vertices, 0), self.face.repeat(count, 1))
        relative = face_w.view(count, vertices, 3) + (state.insertion_end_pos - self.socket_pos).unsqueeze(1)
        face = quat_apply(inverse.repeat_interleave(vertices, 0), relative.view(-1, 3)).view(count, vertices, 3)
        depth = face[..., 0] - env.connector.socket_mouth_x
        axial = (depth.amin(-1) >= self.seat_depth - tolerance) & (depth.amax(-1) <= env.connector.socket_depth + tolerance)
        # Inside a convex counterclockwise polygon: left of every edge.
        edge_start, edge = self.hole, self.hole.roll(-1, 0) - self.hole
        point = face[..., 1:].unsqueeze(2) - edge_start
        cross = edge[:, 0] * point[..., 1] - edge[:, 1] * point[..., 0]
        inside = (cross >= -tolerance * edge.norm(dim=-1)).all(-1).all(-1)
        forward = quat_apply(inverse, quat_apply(state.insertion_end_quat, self.x_axis))[:, 0] > 0
        return axial & inside & forward

    def _grasp_terms(self):
        """Distance and orientation of the hand against the grasp the reward asks for, the closure, and the grasp test."""
        env, cfg, arm = self.env, self.cfg, self.env.arm
        port_quat = env.wire_state.insertion_end_quat
        anchor_x = env.connector.grasp_x if cfg.grasp_anchor_offset_x is None else cfg.grasp_anchor_offset_x
        anchor = env.port_local_pos + quat_apply(port_quat, self.x_axis * anchor_x)
        d_ee = torch.norm(env.ee_local_pos - anchor, dim=-1)
        # The grasp pose rotates with the port; the tilt is about the robot base Y axis.
        root = arm.robot.data.root_quat_w
        target = quat_mul(port_quat, quat_mul(quat_mul(root, quat_mul(self.tilt_b, quat_inv(root))), self.rest_w))
        ee_align = torch.sum(arm.ee_quat * target, dim=-1).abs()
        # Gripper closure in the finger_joint radians of the UR5 Robotiq the thresholds were tuned on.
        gripper = arm.closure * cfg.gripper_reward_travel
        pose_grasp = (d_ee < 0.02) & (ee_align > 0.95)
        grasped = pose_grasp & (gripper > 0.4)
        return d_ee, ee_align, gripper, grasped

    def observations(self) -> torch.Tensor:
        env, arm = self.env, self.env.arm
        port = env.port_local_pos
        to_seated = self.seated_target - port
        return torch.cat((
            arm.joint_pos, arm.joint_vel, env.ee_local_pos, env.ee_local_quat,
            torch.clamp(arm.force_left.unsqueeze(-1) / 10.0, 0.0, 1.0),
            torch.clamp(arm.force_right.unsqueeze(-1) / 10.0, 0.0, 1.0),
            to_seated, port, env.wire_state.insertion_end_quat, self.socket_local_pos, self.seated_target,
            # Residual insertion prior gate: always 0, kept so the observation matches trained checkpoints.
            torch.zeros_like(port[:, :1]),
            to_seated[:, :1].abs(), torch.norm(to_seated[:, 1:], dim=-1, keepdim=True),
        ), dim=-1)

    def success(self) -> torch.Tensor:
        """The whole leading face reached ``seat_depth`` (10 mm) into the bore at some step of the episode.

        Grasping or releasing does not matter; once inserted, the episode counts as a success.
        """
        self.seated = self.is_seated()
        self.inserted |= self.seated
        self.env.extras["tip_depth"] = self.tip_depth()
        return self.inserted.clone()

    def rewards(self) -> torch.Tensor:
        env, cfg, arm = self.env, self.cfg, self.env.arm
        state = env.wire_state
        port, port_quat = env.port_local_pos, state.insertion_end_quat
        d_ee, ee_align, gripper, grasped = self._grasp_terms()
        d_seated = torch.norm(port - self.seated_target, dim=-1)

        # Stage 1: reach the grasp anchor and close on it.
        in_zone = d_ee < 0.05
        reach = (1.0 / (1.0 + 50.0 * d_ee**2)) ** 2
        reach = torch.where(d_ee <= 0.02, reach * 2.0, reach)
        grasp = torch.where(in_zone, gripper * torch.clamp(1.0 - d_ee / 0.05, 0.0, 1.0), -gripper * 0.2)
        ee_align_reward = ee_align**4

        # Stage 2/3: line the port's front and body points up with the bore axis.
        inverse = quat_inv(self.socket_quat)
        front = port + quat_apply(port_quat, self.x_axis * env.connector.tip_length)
        body = port + quat_apply(port_quat, self.x_axis * cfg.port_body_offset_x)
        front_rel = quat_apply(inverse, front - self.seated_target)
        body_rel = quat_apply(inverse, body - self.seated_target)
        front_yz, body_yz = front_rel[:, 1:].norm(dim=-1), body_rel[:, 1:].norm(dim=-1)
        worst_yz = torch.maximum(front_yz, body_yz)
        two_point = 1.0 - torch.tanh(25.0 * (0.5 * (front_yz + body_yz) + 4.5 * worst_yz))
        two_point = torch.where(grasped, two_point, 0.0)
        # The bore is not symmetric under a half turn about the insertion axis, so no abs().
        roll = torch.sum(quat_apply(port_quat, self.y_axis) * quat_apply(self.socket_quat, self.y_axis), dim=-1)

        insert_gate = grasped & (front_yz < 0.006) & (body_yz < 0.008) & (d_seated < 0.12)
        safe_push_gate = grasped & (front_yz < 0.0035) & (body_yz < 0.005) & (d_seated < 0.12)
        body_x = body_rel[:, 0]
        window = cfg.highwater_insert_window
        highwater_gate = safe_push_gate & (body_x > -window) & (body_x < 0.01) & (roll > cfg.roll_align_gate_threshold)

        # Stage 4: pay only new forward progress along the bore, while inside its footprint.
        insert_depth = torch.clamp(body_x + window, 0.0, window)
        improvement = insert_depth - self.best_insert_depth
        highwater = torch.where(highwater_gate, improvement.clamp(0.0, cfg.highwater_insert_step_cap), 0.0)
        self.best_insert_depth = torch.where(
            highwater_gate, torch.maximum(self.best_insert_depth, insert_depth), self.best_insert_depth
        )

        port_rot_sim = torch.sum(port_quat * self.socket_quat, dim=-1).abs()
        port_rot = torch.where(grasped & (d_seated < 0.12), port_rot_sim**4, 0.0)
        roll_reward = torch.where(grasped, roll.clamp(0.0, 1.0) ** 16, 0.0)
        bring = 1.0 / (1.0 + 100.0 * d_seated**2)
        bring = torch.where(port_rot_sim > 0.85, bring, bring * 0.5)
        if cfg.step_wise:
            bring = torch.where(grasped, bring, 0.0)
        push_in = torch.where(insert_gate, torch.exp(-80.0 * d_seated), 0.0)

        # Every step the port is seated (the success test) earns the bonus, however it is held.
        bonus = torch.where(self.seated, cfg.seated_bonus, 0.0)

        action_penalty = torch.sum(env.actions**2, dim=-1)
        ee_table = torch.where(arm.ee_pos[:, 2] < cfg.table_height_limit - 0.03, -1.0, 0.0)
        fist = torch.where(gripper > 0.7, -1.0, 0.0)
        object_table = torch.where(port[:, 2] < cfg.table_height_limit, -1.0, 0.0)

        reward = (
            reach * cfg.reach_reward_scale + grasp * cfg.grasp_reward_scale + ee_align_reward * cfg.ee_align_reward_scale
            + bring * cfg.bring_reward_scale + two_point * cfg.two_point_align_reward_scale
            + highwater * cfg.highwater_insert_reward_scale + push_in * cfg.push_in_reward_scale
            + port_rot * cfg.port_rot_reward_scale + roll_reward * cfg.roll_align_reward_scale
            + object_table * cfg.object_table_penalty_scale + fist * cfg.gripper_closed_penalty_scale
            + ee_table * cfg.ee_table_penalty_scale + bonus - action_penalty * cfg.action_penalty_scale
        )
        # Each term as it enters the reward (already scaled), then a few task metrics.
        penalties = (object_table * cfg.object_table_penalty_scale + fist * cfg.gripper_closed_penalty_scale
                     + ee_table * cfg.ee_table_penalty_scale - action_penalty * cfg.action_penalty_scale)
        env.extras["log"] = {
            "reward/reach": (reach * cfg.reach_reward_scale).mean(),
            "reward/grasp": (grasp * cfg.grasp_reward_scale).mean(),
            "reward/hand_orientation": (ee_align_reward * cfg.ee_align_reward_scale).mean(),
            "reward/bring": (bring * cfg.bring_reward_scale).mean(),
            "reward/line_up": (two_point * cfg.two_point_align_reward_scale).mean(),
            "reward/insert_progress": (highwater * cfg.highwater_insert_reward_scale).mean(),
            "reward/push_in": (push_in * cfg.push_in_reward_scale).mean(),
            "reward/port_orientation": (port_rot * cfg.port_rot_reward_scale).mean(),
            "reward/roll": (roll_reward * cfg.roll_align_reward_scale).mean(),
            "reward/seated_bonus": bonus.mean(),
            "reward/penalties": penalties.mean(),
            "reward/total": reward.mean(),
            "metrics/grasped_rate": grasped.float().mean(),
            "metrics/seated_rate": self.seated.float().mean(),
            "metrics/hand_to_grasp_point_m": d_ee.mean(),
            "metrics/port_to_seat_m": d_seated.mean(),
            "metrics/best_insert_depth_m": self.best_insert_depth.mean(),
        }
        return reward

    def reset(self, env_ids: torch.Tensor) -> None:
        env, cfg = self.env, self.cfg
        self.best_insert_depth[env_ids] = 0.0
        self.inserted[env_ids] = False
        socket = self.socket.data.default_root_state[env_ids].clone()
        board = self.backboard.data.default_root_state[env_ids].clone()
        if cfg.randomize_socket:
            count = len(env_ids)
            y = (torch.rand(count, device=env.device) - 0.5) * cfg.socket_y_range
            z = (torch.rand(count, device=env.device) - 0.5) * cfg.socket_z_range
            socket[:, 1] += y
            socket[:, 2] += z
            board[:, 1] += y
        # The pillar top tracks the socket top so it never blocks the wrist camera.
        board[:, 2] = socket[:, 2] + cfg.socket_half_height - cfg.pillar_height_resolved / 2.0
        origins = env.scene.env_origins[env_ids]
        socket[:, :3] += origins
        board[:, :3] += origins
        self.socket.write_root_state_to_sim(socket, env_ids=env_ids)
        self.socket.reset(env_ids)
        self.backboard.write_root_state_to_sim(board, env_ids=env_ids)
        self.backboard.reset(env_ids)


@configclass
class InsertionCfg:
    task_class: type = InsertionTask
    name: str = "insertion"

    reach_reward_scale: float = 15.0
    grasp_reward_scale: float = 25.0
    ee_align_reward_scale: float = 10.0
    bring_reward_scale: float = 40.0
    two_point_align_reward_scale: float = 20.0
    highwater_insert_reward_scale: float = 6000.0
    push_in_reward_scale: float = 40.0
    port_rot_reward_scale: float = 10.0
    roll_align_reward_scale: float = 10.0
    object_table_penalty_scale: float = 5.0
    gripper_closed_penalty_scale: float = 5.0
    ee_table_penalty_scale: float = 0.0
    action_penalty_scale: float = 0.01
    seated_bonus: float = 50.0
    """Paid every control step the port is seated, grasped or not."""
    step_wise: bool = False
    """Pay the bring reward only while grasped."""
    table_height_limit: float = 0.792

    grasp_anchor_offset_x: float | None = -0.02
    """Grasp point along the port's X from its root, as the v14.5 runs scored it; None takes the connector's ``grasp_x``."""
    port_body_offset_x: float = 0.0
    """Second alignment point along the port's X; the first is the leading face."""
    seated_target_offset_x: float = 0.005
    """Port root target along the socket's X from the socket root, past the backstop so it keeps pushing."""
    gripper_reward_travel: float = 0.785
    """Closure is scaled to this many radians, the UR5 finger_joint travel the thresholds 0.4 and 0.7 refer to."""
    roll_align_gate_threshold: float = 0.996
    highwater_insert_window: float = 0.06
    highwater_insert_step_cap: float = 0.002
    seat_depth: float = 0.010
    """Leading-face depth past the mouth to count as seated; shallow sockets and short tips cap it 2 mm short."""

    socket_pos_xy: tuple[float, float] = (0.67, 0.0)
    socket_z_min: float = 0.80
    socket_half_height: float = 0.020
    socket_y_range: float = 0.28
    socket_z_range: float = 0.15
    randomize_socket: bool = False
    pillar_height: float | None = None
    """None sizes the pillar to cover the socket at its highest pose."""
    pillar_floor_z: float = 0.780

    # Derived by resolve.
    socket_cfg: object = None
    backboard_cfg: object = None
    pillar_height_resolved: float = 0.0

    def observation_dim(self, joints: int) -> int:
        return 2 * joints + 3 + 4 + 1 + 1 + 3 + 3 + 4 + 3 + 3 + 1 + 1 + 1

    def resolve(self, env_cfg) -> None:
        kinematic = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
        z = self.socket_z_min + self.socket_z_range / 2.0
        self.socket_cfg = RigidObjectCfg(
            prim_path=f"{PRIM_PATH}/socket",
            spawn=env_cfg.connector.socket_cfg(rigid_props=kinematic),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(*self.socket_pos_xy, z), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        top = self.socket_z_min + self.socket_z_range + self.socket_half_height
        auto_height = top - self.pillar_floor_z
        self.pillar_height_resolved = auto_height if self.pillar_height is None else float(self.pillar_height)
        self.backboard_cfg = RigidObjectCfg(
            prim_path=f"{PRIM_PATH}/backboard",
            spawn=sim_utils.CuboidCfg(
                size=(0.04, 0.05, self.pillar_height_resolved),
                rigid_props=kinematic,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply", restitution_combine_mode="multiply",
                    static_friction=1.0, dynamic_friction=1.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9), metallic=0.1, roughness=0.5),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(self.socket_pos_xy[0] + 0.03, self.socket_pos_xy[1], self.pillar_floor_z + auto_height / 2.0),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )
