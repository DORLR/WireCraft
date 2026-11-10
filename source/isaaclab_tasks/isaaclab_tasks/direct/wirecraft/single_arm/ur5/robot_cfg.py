# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""UR5 with a Robotiq 2F-85 gripper and an RSD455 wrist camera."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from ..common.env_cfg import PRIM_PATH, RobotCfg

PRIM_NAME = "ur5_robot"
ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


def build_robot_cfg() -> RobotCfg:
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
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.1, 0.0, 0.799),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "shoulder_pan_joint": 3.14159,
                "shoulder_lift_joint": -1.972,
                "elbow_joint": 0.785,
                "wrist_1_joint": 1.658,
                "wrist_2_joint": 1.571,
                "wrist_3_joint": 1.571,
                "finger_joint": 0.0,
            },
        ),
        actuators={
            "ur5_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"],
                effort_limit_sim=150.0,
                stiffness=1008.43994,
                damping=40.03376,
            ),
            "ur5_wrist": ImplicitActuatorCfg(
                joint_names_expr=["wrist_[1-3]_joint"], effort_limit_sim=70.0, stiffness=1000.0, damping=10.0
            ),
            # About 10x the manufacturer values, for finger synchronization on table contact.
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"], effort_limit_sim=100.0, stiffness=1.7, damping=0.2
            ),
            "gripper_passive": ImplicitActuatorCfg(
                joint_names_expr=[".*inner_finger_joint.*"], effort_limit_sim=2.0, stiffness=0.2, damping=0.01
            ),
            "gripper_connection_only": ImplicitActuatorCfg(
                joint_names_expr=[".*outer_finger_joint.*", ".*knuckle.*"],
                effort_limit_sim=0.0,
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )
    return RobotCfg(
        name="ur5",
        adapter="isaaclab_tasks.direct.wirecraft.single_arm.ur5.robot:UR5Adapter",
        articulation=articulation,
        arm_joints=list(ARM_JOINTS),
        gripper_joint="finger_joint",
        ee_body="position_approximation",
        gripper_open_pos=0.0,
        gripper_closed_pos=0.785,
        # The released UR5 policies were trained with the jaw target jumping to open or closed at once.
        gripper_release_step=None,
        # finger_joint in (0.2, 0.7) rad, grasp; < 0.4 rad, released; > 0.7 rad, empty fist.
        grasp_opening_range=(0.108, 0.745),
        release_opening=0.49,
        fully_closed_opening=0.108,
        # Measured: jaws on the 6 mm wire rest at 0.036-0.049, shut on nothing at 0.003.
        wire_pinch_opening=(0.03, 0.3),
        finger_bodies=(".*gripper_left_inner_finger", ".*gripper_right_inner_finger"),
        # The tool axis is +X of this body; rotating it down keeps the fingers across world Y.
        grasp_rest_quat=(0.70710678, 0.0, 0.70710678, 0.0),
        # The insertion reward's grasp pose tilts the fingers back by 0.3 rad, as the policy was trained.
        grasp_tilt=-0.3,
        # The released UR5 policies were trained on the unclipped IK step.
        max_joint_step=None,
        joint_lower=[-6.28] * 6,
        joint_upper=[6.28] * 6,
    )
