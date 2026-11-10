# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Trossen AI Stationary kit, used as a single arm: its right follower arm.

The kit is one articulation: a tabletop with a frame and two 6-DoF arms that face
each other across it. The whole kit is turned -90 deg about Z so the right arm faces
+X like the UR5 and Franka; the left arm stays at its rest pose at the far end.
Each arm carries a D405 wrist camera, and its second jaw follows the first through
a PhysX mimic joint, so only ``follower_right_left_carriage_joint`` is driven.

The arm reaches about half as far as a UR5, so its task layouts (``*/task_cfg``) are
the UR5 ones scaled by ``LAYOUT_SCALE`` toward the arm.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from ..common.env_cfg import PRIM_PATH, RobotCfg

PRIM_NAME = "trossen"
ARM_JOINTS = [f"follower_right_joint_{i}" for i in range(6)]
# Right arm base in the kit frame, and the kit's tabletop top above the kit origin.
RIGHT_BASE_XY = (-0.02, -0.4575)
MOUNT_Z = 0.0391
# World X of the right arm base. A top-down pinch reaches 0.10-0.45 m from it at the table and
# 0.10-0.40 m at hover height, so the scaled layouts start 0.16 m and end 0.45 m out.
BASE_X = 0.10
TABLE_TOP_Z = 0.7825
LAYOUT_SCALE = 0.55
"""Trossen layout = UR5 layout scaled by this about x = 0.4 (the UR5 workspace centre), y = 0."""

# Gains are the drive settings authored in the kit USD.
_ARM_GAINS = {
    0: (100.0, 663.98, 5.53),
    1: (100.0, 734.56, 6.12),
    2: (100.0, 737.60, 6.14),
    3: (30.0, 62.30, 0.778),
    4: (30.0, 33.82, 0.846),
    5: (30.0, 33.33, 0.83),
}


def build_robot_cfg() -> RobotCfg:
    # Kit turned -90 deg about Z: kit (x, y) lands on world (y, -x).
    base_world = (RIGHT_BASE_XY[1], -RIGHT_BASE_XY[0])
    articulation = ArticulationCfg(
        prim_path=f"{PRIM_PATH}/{PRIM_NAME}",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, disable_gravity=True, max_depenetration_velocity=5.0
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=True,
                enabled_self_collisions=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # The kit's tabletop sits flush with the workstation table.
            pos=(BASE_X - base_world[0], -base_world[1], TABLE_TOP_Z - MOUNT_Z),
            rot=(0.70710678, 0.0, 0.0, -0.70710678),
            joint_pos={
                # Right arm: tool pointing down 0.25 m out, 0.14 m above the table, jaws across world Y.
                "follower_right_joint_0": 0.0,
                "follower_right_joint_1": 1.4835,
                "follower_right_joint_2": 1.2741,
                "follower_right_joint_3": -1.3614,
                "follower_right_joint_4": 0.0,
                "follower_right_joint_5": 0.0,
                "follower_right_left_carriage_joint": 0.04,
                "follower_left_joint_0": -0.0489,
                "follower_left_joint_[1-5]": 0.0,
                "follower_left_left_carriage_joint": 0.0,
            },
        ),
        actuators={
            **{
                f"arm_joint_{i}": ImplicitActuatorCfg(
                    # Both arms by index; the left one only holds its rest pose.
                    joint_names_expr=[f"follower_.*_joint_{i}"], effort_limit_sim=effort, stiffness=kp, damping=kd
                )
                for i, (effort, kp, kd) in _ARM_GAINS.items()
            },
            # The kit authors 400 N on the jaws; even 70 N slams a 10 g port into the wire solve.
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["follower_.*_left_carriage_joint"], effort_limit_sim=20.0, stiffness=2.0e5, damping=1.0e3
            ),
        },
    )
    return RobotCfg(
        name="trossen",
        adapter="isaaclab_tasks.direct.wirecraft.single_arm.trossen.robot:TrossenAdapter",
        articulation=articulation,
        arm_joints=list(ARM_JOINTS),
        gripper_joint="follower_right_left_carriage_joint",
        # Link 6 +X runs out between the jaws, whose pads span 0.087-0.156 m along it. The grasp point
        # sits 8 mm short of the tips, so the tips clear the table when it reaches a port lying there.
        ee_body="grasp_center",
        tool_parent="follower_right_link_6",
        tool_offset=(0.148, 0.0, 0.0),
        gripper_open_pos=0.04,
        gripper_closed_pos=0.0,
        gripper_max_step=0.0004,
        gripper_release_step=0.00008,
        grasp_opening_range=(0.03, 0.9),
        release_opening=0.9,
        fully_closed_opening=0.03,
        # The fingertips stop about 9 mm above the table, above a 6 mm wire lying on it: the jaws shut over the
        # wire, and the wire holder ring (not the fingers) carries it. Any closed opening counts as a pinch.
        wire_pinch_opening=(-1.0, 0.5),
        wire_gripper_closed_pos=0.0025,
        wire_grasp_z_offset=0.0065,
        finger_bodies=("follower_right_gripper_left", "follower_right_gripper_right"),
        # Link 6 +X down and its jaw axis (+Y) along world +Y, as on the UR5. The jaw axis along -Y
        # needs the wrist roll past its limit from every reachable posture.
        grasp_rest_quat=(0.70710678, 0.0, 0.70710678, 0.0),
        grasp_z_offset=0.005,
        # The closed jaws are wide plates; pressed at the UR5 depth they pin the short chain against the
        # table and clip and its solve diverges.
        press_z_offset=0.004,
        ik_tracking_step=0.006,
        # At 6 mm a step insertion overshoots: the port swings +-10 mm about the socket mouth on the
        # deformable wire, and one descent in five folds the wrist onto its -90 deg limit above the port.
        fine_tracking_step=0.002,
    )
