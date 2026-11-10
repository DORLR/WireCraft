# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration: one ID per robot, wire, connector and task variant."""

from __future__ import annotations

import os

import gymnasium as gym

from .compose import TaskSpec, agent_entry_point, env_entry_point

ENV_ENTRY = "isaaclab_tasks.direct.wirecraft.single_arm.common.env:SingleArmWireEnv"
ROBOT_NAMES = {"ur5": "UR5", "franka": "Franka", "trossen": "Trossen"}
WIRE_NAMES = {"articulation": "Articulation", "deformable": "Deformable"}
CONNECTOR_NAMES = {
    "cylinder": "Cylinder",
    "cuboid": "Cuboid",
    "stacked_cuboid": "StackedCuboid",
    "displayport_style": "DisplayPortStyle",
    "ethernet": "Ethernet",
    "displayport": "DisplayPort",
    "hdmi": "HDMI",
    "usb_a": "USBA",
}
ROUTING_VARIANTS = {f"{n}clip": ("stacked_cuboid", n) for n in (1, 2, 3)} | {"1clip_usb_a": ("usb_a", 1)}
CHANNEL_CONNECTORS = ("stacked_cuboid", "usb_a")

SPECS: dict[str, TaskSpec] = {}


def _register(task_id: str, spec: TaskSpec) -> None:
    SPECS[task_id] = spec
    gym.register(
        id=task_id,
        entry_point=ENV_ENTRY,
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": env_entry_point(spec), "rsl_rl_cfg_entry_point": agent_entry_point(spec)},
    )


def _prefix(robot: str, wire: str, connector: str) -> str:
    return f"WireCraft-{ROBOT_NAMES[robot]}-{WIRE_NAMES[wire]}-{CONNECTOR_NAMES[connector]}"


def register_insertion(robot: str, robot_cfg: str, leaf: str) -> None:
    for wire in WIRE_NAMES:
        for connector in CONNECTOR_NAMES:
            _register(
                f"{_prefix(robot, wire, connector)}-Insertion-v0",
                TaskSpec(robot, robot_cfg, "insertion", wire, connector, leaf_dir=os.path.dirname(leaf)),
            )


def register_clip_routing(robot: str, robot_cfg: str, leaf: str) -> None:
    for wire in WIRE_NAMES:
        for variant, (connector, clips) in ROUTING_VARIANTS.items():
            _register(
                f"{_prefix(robot, wire, connector)}-{clips}Clip-Routing-v0",
                TaskSpec(
                    robot, robot_cfg, "clip_routing", wire, connector, f"{clips}clip",
                    {"task": {"num_clips": clips}}, os.path.dirname(leaf),
                ),
            )


def register_lay_in_channel(robot: str, robot_cfg: str, leaf: str) -> None:
    for wire in WIRE_NAMES:
        for connector in CHANNEL_CONNECTORS:
            for channel in ("straight", "curved"):
                _register(
                    f"{_prefix(robot, wire, connector)}-{channel.capitalize()}Channel-v0",
                    TaskSpec(
                        robot, robot_cfg, "lay_in_channel", wire, connector, channel,
                        {"task": {"channel_type": channel}}, os.path.dirname(leaf),
                    ),
                )


def register_alias(old_id: str, new_id: str) -> None:
    """Thin forward for an ID that existing scripts still name."""
    _register(old_id, SPECS[new_id])
