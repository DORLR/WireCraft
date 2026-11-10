# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

from ...common.registry import register_clip_routing

register_clip_routing("franka", "isaaclab_tasks.direct.wirecraft.single_arm.franka.robot_cfg:build_robot_cfg", __file__)
