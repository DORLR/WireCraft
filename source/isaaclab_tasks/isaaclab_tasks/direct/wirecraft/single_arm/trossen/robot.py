# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Trossen adapter: asset path and the right arm's D405 wrist camera."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg

from ..common.env_cfg import PRIM_PATH
from ..common.robot import RobotAdapter
from .robot_cfg import PRIM_NAME

# Trossen's own Isaac asset, pinned to the commit that last changed it; ``env.robot.usd_path`` replaces it.
USD_URL = (
    "https://raw.githubusercontent.com/TrossenRobotics/trossen_ai_isaac/"
    "01d36c6b14b77677f49a6b77bedeba89fed3c235/assets/robots/stationary_ai/stationary_ai.usd"
)


class TrossenAdapter(RobotAdapter):
    prim_name = PRIM_NAME

    @classmethod
    def resolve_usd_path(cls, robot_cfg) -> str:
        return USD_URL

    @classmethod
    def camera_cfgs(cls, cfg) -> dict:
        size = cfg.camera_resolution
        return {
            # The kit's camera link is a ROS camera_link: +X out of the lens, +Z up, as in the world convention.
            "color": TiledCameraCfg(
                prim_path=f"{PRIM_PATH}/{PRIM_NAME}/follower_right_camera_link/CameraColor",
                offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
                data_types=["rgb"],
                # D405: 1.88 mm lens, 87 x 58 deg field of view.
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=1.88, horizontal_aperture=3.58, vertical_aperture=2.08, clipping_range=(0.05, 10.0)
                ),
                width=size,
                height=size,
            )
        }
