# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""UR5 adapter: asset paths, passive finger joints and the wrist camera."""

from __future__ import annotations

import torch
from pxr import UsdPhysics

from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.utils import get_current_stage

from ..common.env_cfg import PRIM_PATH, robot_asset
from ..common.robot import RobotAdapter
from .robot_cfg import PRIM_NAME

PASSIVE_TARGETS = {"right_inner_finger_joint": 0.7854, "left_inner_finger_joint": -0.7854}
WRIST_CAMERAS = {
    "left": "Camera_OmniVision_OV9782_Left",
    "right": "Camera_OmniVision_OV9782_Right",
    "color": "Camera_OmniVision_OV9782_Color",
    "depth": "Camera_Pseudo_Depth",
}


class UR5Adapter(RobotAdapter):
    prim_name = PRIM_NAME

    @classmethod
    def resolve_usd_path(cls, robot_cfg) -> str:
        return robot_asset("ur5_with_gripper_camera_v2.usd")

    @classmethod
    def camera_cfgs(cls, cfg) -> dict:
        size = cfg.camera_resolution
        return {
            name: TiledCameraCfg(
                prim_path=f"{PRIM_PATH}/{PRIM_NAME}/RSD455/{prim}",
                spawn=None,
                data_types=["distance_to_camera" if name == "depth" else "rgb"],
                width=size,
                height=size,
            )
            for name, prim in WRIST_CAMERAS.items()
        }

    def post_clone(self) -> None:
        super().post_clone()
        # The asset authors a negative mass on this tracking body.
        stage = get_current_stage()
        for env_id in range(self.env.scene.cfg.num_envs):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/{PRIM_NAME}/{self.cfg.ee_body}")
            if prim.IsValid():
                mass = UsdPhysics.MassAPI(prim)
                mass.CreateMassAttr().Set(0.01)
                mass.CreateDiagonalInertiaAttr().Set((0.0001, 0.0001, 0.0001))

    def initialize(self) -> None:
        super().initialize()
        ids, targets = self.robot.find_joints(list(PASSIVE_TARGETS), preserve_order=True)[0], list(PASSIVE_TARGETS.values())
        self.passive_ids = ids
        self.passive_targets = torch.tensor(targets, device=self.env.device).repeat(self.env.num_envs, 1)

    def apply_gripper_extras(self, gripper: torch.Tensor) -> None:
        self.robot.set_joint_position_target(self.passive_targets, joint_ids=self.passive_ids)
