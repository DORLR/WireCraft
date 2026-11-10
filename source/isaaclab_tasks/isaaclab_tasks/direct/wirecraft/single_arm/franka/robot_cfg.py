# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda with its parallel gripper.

The asset is Isaac Lab's official Panda. ``panda_hand`` is the flange-side body; the
grasp point sits 0.1034 m along its +Z axis, between the finger pads, where the adapter
fixes a ``grasp_center`` body for the IK to track.
"""

from __future__ import annotations

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

from ..common.env_cfg import PRIM_PATH, RobotCfg

PRIM_NAME = "franka"
ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]


def build_robot_cfg() -> RobotCfg:
    articulation = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path=f"{PRIM_PATH}/{PRIM_NAME}").copy()
    articulation.init_state.pos = (0.10, -0.35, 0.7851)
    articulation.spawn.articulation_props.solver_position_iteration_count = 32
    articulation.spawn.activate_contact_sensors = True
    # The stock 200 N hand slams a 10 g port hard enough to blow up the wire solve.
    articulation.actuators["panda_hand"].effort_limit_sim = 70.0
    return RobotCfg(
        name="franka",
        adapter="isaaclab_tasks.direct.wirecraft.single_arm.franka.robot:FrankaAdapter",
        articulation=articulation,
        arm_joints=list(ARM_JOINTS),
        gripper_joint="panda_finger_joint1",
        # The official asset has no body between the pads, so one is hung on the hand after loading.
        ee_body="grasp_center",
        tool_parent="panda_hand",
        tool_offset=(0.0, 0.0, 0.1034),
        gripper_open_pos=0.04,
        gripper_closed_pos=0.0,
        gripper_max_step=0.0004,
        gripper_release_step=0.00008,
        # Normalized per-finger travel: a held port keeps the fingers apart.
        grasp_opening_range=(0.03, 0.9),
        release_opening=0.9,
        fully_closed_opening=0.03,
        # Measured: fingers on the 6 mm wire rest at 0.029-0.039.
        # In clip routing the fingers stop 2.5 mm each from centre, a 5 mm gap on the 6 mm wire: it is held, not
        # crushed, and the wire holder ring keeps it. Any closed opening then counts as a pinch.
        wire_gripper_closed_pos=0.0025,
        # The Franka fingers are thicker along the wire than the Robotiq's: pinched 2 cm past the clip they
        # pass into it as the wire is drawn down.
        wire_pinch_clearance=0.035,
        wire_pinch_opening=(-1.0, 0.5),
        # The fingertips reach the table with the tool point about 9 mm up; pinch the wire from there.
        wire_grasp_z_offset=0.0065,
        finger_bodies=("panda_leftfinger", "panda_rightfinger"),
        grasp_rest_quat=(0.0, 1.0, 0.0, 0.0),
        grasp_z_offset=0.005,
        ik_tracking_step=0.006,
    )
