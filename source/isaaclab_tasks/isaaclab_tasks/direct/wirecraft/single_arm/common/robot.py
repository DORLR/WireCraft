# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Robot adapter interface and the shared differential-IK action path."""

from __future__ import annotations

import torch
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.utils import get_current_stage
from isaaclab.utils.math import apply_delta_pose, matrix_from_quat, quat_apply, quat_inv, subtract_frame_transforms

from .env_cfg import PRIM_PATH

TOOL_BODY = "grasp_center"
TOOL_MASS = 1.0e-4
"""PhysX needs a positive mass; this is 1/100 of a port, small enough not to load the wrist."""
TOOL_INERTIA = 1.0e-8


class RobotAdapter:
    """Binds one robot to the shared environment.

    Subclasses supply joint/body names through :class:`RobotCfg` and override
    the small hooks below. Tasks use only the attributes set in :meth:`update`.
    """

    prim_name = "robot"

    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg.robot

    # ---- scene -------------------------------------------------------------
    @property
    def prim_path(self) -> str:
        return f"{PRIM_PATH}/{self.prim_name}"

    @classmethod
    def camera_cfgs(cls, cfg) -> dict:
        """Wrist cameras keyed by observation name."""
        return {}

    @classmethod
    def resolve_usd_path(cls, robot_cfg) -> str:
        raise NotImplementedError

    def spawn(self) -> None:
        self.robot = Articulation(self.cfg.articulation)
        self.env.scene.articulations["robot"] = self.robot
        if self.cfg.tool_parent:
            self._add_tool_body()
        # The insertion port comes first: force_matrix_w column 0 is the port, the rest is cable.
        target = [self.env.cfg.wire_prim_path + "/end_port", *self.env.wire_backend.cable_contact_paths()]
        self.contact_left, self.contact_right = (
            ContactSensor(
                ContactSensorCfg(
                    prim_path=f"{self.prim_path}/{body}",
                    update_period=0.0,
                    history_length=3,
                    filter_prim_paths_expr=target,
                )
            )
            for body in self.cfg.finger_bodies
        )
        self.env.scene.sensors["contact_left"] = self.contact_left
        self.env.scene.sensors["contact_right"] = self.contact_right

    def _add_tool_body(self) -> None:
        """Fix a massless-in-effect body with no geometry at the grasp point, before the envs are cloned.

        It has no collider and nothing to render; it only gives the IK a link that sits between the pads.
        """
        stage = get_current_stage()
        root = f"/World/envs/env_0/{self.prim_name}"
        parent = stage.GetPrimAtPath(f"{root}/{self.cfg.tool_parent}")
        if not parent.IsValid():
            raise ValueError(f"tool_parent {self.cfg.tool_parent!r} is not a link of {root}.")
        cache = UsdGeom.XformCache()
        offset = Gf.Vec3d(*self.cfg.tool_offset)
        world = Gf.Matrix4d().SetTranslate(offset) * cache.GetLocalToWorldTransform(parent)
        local = world * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(root)).GetInverse()
        body = UsdGeom.Xform.Define(stage, f"{root}/{TOOL_BODY}")
        body.AddTransformOp().Set(local)
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass.CreateMassAttr().Set(TOOL_MASS)
        mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(TOOL_INERTIA, TOOL_INERTIA, TOOL_INERTIA))
        PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim()).CreateDisableGravityAttr().Set(True)
        joint = UsdPhysics.FixedJoint.Define(stage, f"{root}/{TOOL_BODY}_joint")
        joint.CreateBody0Rel().SetTargets([parent.GetPath()])
        joint.CreateBody1Rel().SetTargets([body.GetPath()])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*self.cfg.tool_offset))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

    def post_clone(self) -> None:
        """Bind the high-friction pad material; port contact friction comes only from it."""
        if self.cfg.pad_static_friction is None:
            return
        material = sim_utils.RigidBodyMaterialCfg(
            static_friction=float(self.cfg.pad_static_friction),
            dynamic_friction=float(self.cfg.pad_dynamic_friction),
            friction_combine_mode=self.cfg.pad_friction_combine_mode,
            restitution=0.0,
        )
        path = "/World/Materials/gripper_pad"
        material.func(path, material)
        for env_id in range(self.env.scene.cfg.num_envs):
            for body in self.cfg.finger_bodies:
                for prim in sim_utils.find_matching_prim_paths(f"/World/envs/env_{env_id}/{self.prim_name}/{body}"):
                    sim_utils.bind_physics_material(prim, path, stronger_than_descendants=True)

    # ---- runtime -----------------------------------------------------------
    def initialize(self) -> None:
        env, robot = self.env, self.robot
        self.arm_ids = [robot.find_joints(name)[0][0] for name in self.cfg.arm_joints]
        self.gripper_id = robot.find_joints(self.cfg.gripper_joint)[0][0]
        self.joint_ids = [*self.arm_ids, self.gripper_id]
        self.ee_id = robot.find_bodies(self.cfg.ee_body)[0][0]
        # Fixed-base Jacobians omit the root link.
        self.jacobian_id = self.ee_id - 1 if robot.is_fixed_base else self.ee_id
        self.ee_offset = torch.tensor(self.cfg.ee_offset, device=env.device).repeat(env.num_envs, 1)
        if self.cfg.joint_lower is None:
            limits = robot.data.soft_joint_pos_limits[0, self.arm_ids]
            self.arm_lower, self.arm_upper = limits[:, 0].clone(), limits[:, 1].clone()
        else:
            self.arm_lower = torch.tensor(self.cfg.joint_lower, device=env.device)
            self.arm_upper = torch.tensor(self.cfg.joint_upper, device=env.device)
        self.gripper_range = abs(self.cfg.gripper_closed_pos - self.cfg.gripper_open_pos)
        wire_closed = self.cfg.wire_gripper_closed_pos
        self.closed_target = wire_closed if env.cfg.gripper_close_on_wire and wire_closed is not None else self.cfg.gripper_closed_pos
        self.targets = robot.data.default_joint_pos[:, self.joint_ids].clone()
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=env.num_envs,
            device=env.device,
        )
        if self.cfg.ik_joint_reference not in ("control", "live"):
            raise ValueError(f"robot.ik_joint_reference must be 'control' or 'live', got {self.cfg.ik_joint_reference!r}.")
        self.control_joint_pos = self.targets[:, : len(self.arm_ids)].clone()
        # Optional wrist F/T sensor: the joint reaction wrench into ``ft_sensor_body`` from its parent link.
        self.ft_id = robot.find_bodies(self.cfg.ft_sensor_body)[0][0] if self.cfg.ft_sensor_body else None
        self.wrench = torch.zeros(env.num_envs, 6, device=env.device)
        self.fixed_quat_b = torch.zeros(env.num_envs, 4, device=env.device)
        self.fixed_quat_b[:, 0] = 1.0
        self.update()

    def update(self) -> None:
        data = self.robot.data
        self.joint_pos = data.joint_pos[:, self.joint_ids]
        self.joint_vel = data.joint_vel[:, self.joint_ids]
        body_pos, body_quat = data.body_pos_w[:, self.ee_id], data.body_quat_w[:, self.ee_id]
        self.ee_quat = body_quat
        self.ee_pos = body_pos + quat_apply(body_quat, self.ee_offset)
        if getattr(self, "ft_id", None) is not None:
            # (fx, fy, fz, tx, ty, tz) in N and N m, in the parent link's frame (UR5: wrist_3_link). Raw: it
            # includes the gripper's own weight; subtract a reading taken at rest to tare it.
            self.wrench = data.body_incoming_joint_wrench_b[:, self.ft_id]
        self.force_left, self.cable_force_left = self._forces(self.contact_left)
        self.force_right, self.cable_force_right = self._forces(self.contact_right)
        travel = (self.joint_pos[:, -1] - self.cfg.gripper_closed_pos).abs()
        self.opening = travel / self.gripper_range

    @staticmethod
    def _forces(sensor: ContactSensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normal force of one pad on the insertion port, and on the cable, in N."""
        force = torch.norm(sensor.data.force_matrix_w, dim=-1)
        return force[..., 0].sum(dim=-1), force[..., 1:].sum(dim=[-1, -2])

    @property
    def is_grasping(self) -> torch.Tensor:
        low, high = self.cfg.grasp_opening_range
        return (
            (self.force_left > 0.05) & (self.force_right > 0.05) & (self.opening > low) & (self.opening < high)
        )

    @property
    def closure(self) -> torch.Tensor:
        """0 open, 1 fully closed."""
        return 1.0 - self.opening

    @property
    def is_released(self) -> torch.Tensor:
        return self.opening > self.cfg.release_opening

    @property
    def is_fully_closed(self) -> torch.Tensor:
        return self.opening < self.cfg.fully_closed_opening

    def joint_speed(self) -> torch.Tensor:
        """Fastest arm joint in rad/s; NaN once any joint state is lost.

        Gripper joints are left out of the speed: an instantly switched jaw (UR5) and the Robotiq's passive
        inner fingers snap far past the arm's limit when they open or meet the table or a wire, which is not a
        simulation fault. Their state still counts as lost when it turns non-finite.
        """
        data = self.robot.data
        speed = data.joint_vel[:, self.arm_ids].abs().amax(dim=-1)
        finite = torch.isfinite(data.joint_pos).all(-1) & torch.isfinite(data.joint_vel).all(-1)
        return torch.where(finite, speed, torch.nan)

    def joint_limit_mask(self) -> torch.Tensor:
        arm = self.joint_pos[:, :-1]
        return ((arm < self.arm_lower) | (arm > self.arm_upper)).any(dim=-1)

    def base_frame_ee(self) -> tuple[torch.Tensor, torch.Tensor]:
        root = self.robot.data.root_pose_w
        return subtract_frame_transforms(root[:, :3], root[:, 3:7], self.ee_pos, self.ee_quat)

    def _jacobian_b(self) -> torch.Tensor:
        """Base-frame geometric Jacobian of the tool point."""
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.jacobian_id, :6, :][:, :, self.arm_ids].clone()
        arm = quat_apply(self.ee_quat, self.ee_offset)
        angular = jacobian[:, 3:6, :]
        jacobian[:, :3, :] += torch.cross(angular.transpose(1, 2), arm[:, None, :].expand(-1, angular.shape[2], -1), dim=-1).transpose(1, 2)
        rotation = matrix_from_quat(quat_inv(self.robot.data.root_quat_w))
        return torch.cat((rotation @ jacobian[:, :3, :], rotation @ angular), dim=1)

    def latch_control_state(self) -> None:
        """Called once per control step, before the physics substeps."""
        self.control_joint_pos = self.robot.data.joint_pos[:, self.arm_ids].clone()

    def apply_action(self, action: torch.Tensor, scale: float, fix_orient: bool, control_mode: str) -> None:
        """Action layout: arm command, then one gripper value (<0 opens)."""
        self.update()
        arm, gripper = action[:, :-1] * scale, action[:, -1:]
        count = len(self.arm_ids)
        reference = self.control_joint_pos if self.cfg.ik_joint_reference == "control" else self.joint_pos[:, :count]
        if control_mode == "cartesian":
            ee_pos_b, ee_quat_b = self.base_frame_ee()
            if fix_orient:
                pose = torch.cat((ee_pos_b + arm, self.fixed_quat_b), dim=-1)
            else:
                pose = torch.cat(apply_delta_pose(ee_pos_b, ee_quat_b, arm), dim=-1)
            self.ik.set_command(pose)
            jacobian = self._jacobian_b()
            self.targets[:, :count] = self.ik.compute(ee_pos_b, ee_quat_b, jacobian, reference)
            self.targets[:, :count] += self.null_space_step(jacobian)
        else:
            self.targets[:, :count] = reference + arm
        step = self.cfg.max_joint_step
        if step is not None:
            current = self.joint_pos[:, :count]
            self.targets[:, :count] = current + (self.targets[:, :count] - current).clamp(-step, step)
        self.targets[:, :count] = torch.maximum(torch.minimum(self.targets[:, :count], self.arm_upper), self.arm_lower)
        command = torch.where(
            gripper[:, 0] < 0,
            torch.full_like(gripper[:, 0], self.cfg.gripper_open_pos),
            torch.full_like(gripper[:, 0], self.closed_target),
        )
        self.targets[:, count] = self._slew(self.targets[:, count], command)
        self.robot.set_joint_position_target(self.targets, joint_ids=self.joint_ids)
        self.apply_gripper_extras(gripper)
        self.robot.write_data_to_sim()

    def null_space_step(self, jacobian: torch.Tensor) -> torch.Tensor | float:
        """Joint motion that leaves the tool still; only redundant arms have any."""
        return 0.0

    def _slew(self, current: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        delta = command - current
        opening = delta * (self.cfg.gripper_open_pos - self.cfg.gripper_closed_pos) > 0
        close, release = self.cfg.gripper_max_step, self.cfg.gripper_release_step
        limit = torch.where(opening, float("inf") if release is None else release, float("inf") if close is None else close)
        return current + torch.maximum(torch.minimum(delta, limit), -limit)

    def apply_gripper_extras(self, gripper: torch.Tensor) -> None:
        """Hook for mimic or passive gripper joints."""

    def reset(self, env_ids: torch.Tensor) -> None:
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        self.robot.reset(env_ids)
        self.targets[env_ids] = joint_pos[:, self.joint_ids]
        self.control_joint_pos[env_ids] = joint_pos[:, self.arm_ids]
        self.ik.reset(env_ids)
        # Orientation is held at the tool's value in the default posture.
        self.update()
        _, quat_b = self.base_frame_ee()
        self.fixed_quat_b[env_ids] = quat_b[env_ids]
