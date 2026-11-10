# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Franka adapter: asset path, mimic finger and the wrist camera."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg

from ..common.env_cfg import PRIM_PATH
from ..common.robot import RobotAdapter
from .robot_cfg import PRIM_NAME

# Isaac Lab's Franka visuomotor benchmark wrist camera (manipulation/stack, stack_ik_rel_visuomotor_env_cfg.py),
# so wrist images line up with data collected there.
WRIST_POS = (0.13, 0.0, -0.15)
WRIST_ROT = (-0.70614, 0.03701, 0.03701, -0.70614)
# The seventh joint is spent keeping the arm off its limits; pure pose IK lets it drift into them near the base.
NULL_GAIN = 0.1
NULL_STEP = 0.01
DAMPING = 0.01


class FrankaAdapter(RobotAdapter):
    prim_name = PRIM_NAME

    @classmethod
    def resolve_usd_path(cls, robot_cfg) -> str:
        return robot_cfg.articulation.spawn.usd_path

    @classmethod
    def camera_cfgs(cls, cfg) -> dict:
        size = cfg.camera_resolution
        return {
            "color": TiledCameraCfg(
                prim_path=f"{PRIM_PATH}/{PRIM_NAME}/panda_hand/wrist_cam",
                offset=TiledCameraCfg.OffsetCfg(pos=WRIST_POS, rot=WRIST_ROT, convention="ros"),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.0)
                ),
                width=size,
                height=size,
            )
        }

    def initialize(self) -> None:
        super().initialize()
        self.mimic_id = self.robot.find_joints("panda_finger_joint2")[0]

    def null_space_step(self, jacobian: torch.Tensor) -> torch.Tensor:
        """Pull the arm toward mid-range in the Jacobian's null space."""
        arm = self.joint_pos[:, : len(self.arm_ids)]
        toward = (NULL_GAIN * (0.5 * (self.arm_lower + self.arm_upper) - arm)).clamp(-NULL_STEP, NULL_STEP)
        transpose = jacobian.transpose(1, 2)
        eye = torch.eye(6, device=jacobian.device)
        inverse = transpose @ torch.linalg.solve(jacobian @ transpose + DAMPING**2 * eye, jacobian)
        return toward - (inverse @ toward.unsqueeze(-1)).squeeze(-1)

    def apply_gripper_extras(self, gripper: torch.Tensor) -> None:
        self.robot.set_joint_position_target(self.targets[:, -1:], joint_ids=self.mimic_id)
