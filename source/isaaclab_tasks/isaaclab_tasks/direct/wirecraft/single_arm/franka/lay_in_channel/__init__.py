# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

from ...common.registry import register_lay_in_channel

register_lay_in_channel("franka", "isaaclab_tasks.direct.wirecraft.single_arm.franka.robot_cfg:build_robot_cfg", __file__)
