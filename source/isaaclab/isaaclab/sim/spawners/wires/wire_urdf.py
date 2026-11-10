# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""URDF export utility for wire configurations."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

from .wire_cfg import WireCfg


def export_wire_urdf(wire_cfg: WireCfg) -> str:
    """Generate a URDF XML string from a :class:`WireCfg`.

    Each segment becomes a ``<link>`` with cylinder ``<visual>`` and ``<collision>``
    geometry, plus auto-computed ``<inertial>`` from density.

    Since URDF has no native D6 or spherical joint, each connection uses 3 revolute
    joints (X, Y, Z) with intermediate massless links to approximate spherical DOFs.

    Args:
        wire_cfg: Wire configuration.

    Returns:
        A URDF XML string.
    """
    seg_length = wire_cfg.total_length / wire_cfg.num_segments
    seg_mass = wire_cfg.density * math.pi * wire_cfg.radius**2 * seg_length

    # Cylinder inertia (solid cylinder, axis along X)
    r = wire_cfg.radius
    l = seg_length  # noqa: E741
    ixx = seg_mass * (3 * r**2 + l**2) / 12.0
    iyy = ixx
    izz = seg_mass * r**2 / 2.0

    robot = ET.Element("robot", name="wire")

    for i in range(wire_cfg.num_segments):
        _add_segment_link(robot, i, seg_length, wire_cfg.radius, seg_mass, ixx, iyy, izz, wire_cfg.color)

    for i in range(1, wire_cfg.num_segments):
        _add_spherical_joint_chain(robot, i, seg_length, wire_cfg)

    # Optionally add fixed base
    if wire_cfg.fix_base:
        world_link = ET.SubElement(robot, "link", name="world")
        ET.SubElement(
            robot,
            "joint",
            name="fixed_base",
            type="fixed",
        )
        parent = ET.SubElement(robot[-1], "parent", link="world")
        child = ET.SubElement(robot[-1], "child", link="seg_0")
        ET.SubElement(robot[-1], "origin", xyz="0 0 0", rpy="0 0 0")

    ET.indent(robot, space="  ")
    return ET.tostring(robot, encoding="unicode", xml_declaration=True)


def save_wire_urdf(wire_cfg: WireCfg, path: str) -> str:
    """Export a URDF and write it to a file.

    Args:
        wire_cfg: Wire configuration.
        path: File path to write the URDF to.

    Returns:
        The URDF XML string.
    """
    urdf_str = export_wire_urdf(wire_cfg)
    with open(path, "w") as f:
        f.write(urdf_str)
    return urdf_str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _add_segment_link(
    robot: ET.Element,
    idx: int,
    seg_length: float,
    radius: float,
    mass: float,
    ixx: float,
    iyy: float,
    izz: float,
    color: tuple[float, float, float],
) -> None:
    """Add a <link> element for a cylinder segment."""
    link = ET.SubElement(robot, "link", name=f"seg_{idx}")

    # Inertial
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass", value=f"{mass:.6f}")
    ET.SubElement(
        inertial,
        "inertia",
        ixx=f"{ixx:.8f}",
        ixy="0",
        ixz="0",
        iyy=f"{iyy:.8f}",
        iyz="0",
        izz=f"{izz:.8f}",
    )
    # Cylinder default axis in URDF is Z; we rotate 90deg about Y to align with X
    ET.SubElement(inertial, "origin", xyz="0 0 0", rpy=f"0 {math.pi / 2:.6f} 0")

    # Visual
    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", xyz="0 0 0", rpy=f"0 {math.pi / 2:.6f} 0")
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "cylinder", radius=f"{radius:.6f}", length=f"{seg_length:.6f}")
    material = ET.SubElement(visual, "material", name="wire_color")
    ET.SubElement(material, "color", rgba=f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} 1.0")

    # Collision
    collision = ET.SubElement(link, "collision")
    ET.SubElement(collision, "origin", xyz="0 0 0", rpy=f"0 {math.pi / 2:.6f} 0")
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "cylinder", radius=f"{radius:.6f}", length=f"{seg_length:.6f}")


def _add_spherical_joint_chain(
    robot: ET.Element,
    idx: int,
    seg_length: float,
    wire_cfg: WireCfg,
) -> None:
    """Approximate a spherical joint with 3 revolute joints (X, Y, Z).

    Creates two intermediate massless links between seg_{idx-1} and seg_{idx}.
    """
    half_seg = seg_length / 2.0
    axes = [("1 0 0", "x"), ("0 1 0", "y"), ("0 0 1", "z")]

    parent_link = f"seg_{idx - 1}"

    for j, (axis_vec, axis_name) in enumerate(axes):
        joint_name = f"joint_{idx}_{axis_name}"
        joint_type = "revolute" if wire_cfg.joint_limit is not None else "continuous"

        joint = ET.SubElement(robot, "joint", name=joint_name, type=joint_type)
        ET.SubElement(joint, "parent", link=parent_link)

        # For the first axis joint, the origin is at +half_seg from parent center
        # For subsequent axes, origin is at 0 (same point, just adding DOF)
        if j == 0:
            ET.SubElement(joint, "origin", xyz=f"{half_seg:.6f} 0 0", rpy="0 0 0")
        else:
            ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")

        ET.SubElement(joint, "axis", xyz=axis_vec)

        if wire_cfg.joint_limit is not None:
            ET.SubElement(
                joint,
                "limit",
                lower=f"{-wire_cfg.joint_limit:.6f}",
                upper=f"{wire_cfg.joint_limit:.6f}",
                effort="100",
                velocity="100",
            )
        elif joint_type == "continuous":
            # continuous joints still need effort/velocity in some parsers
            ET.SubElement(joint, "limit", effort="100", velocity="100")

        # Intermediate link (massless) for first two axes
        if j < 2:
            inter_name = f"inter_{idx}_{axis_name}"
            inter_link = ET.SubElement(robot, "link", name=inter_name)
            inertial = ET.SubElement(inter_link, "inertial")
            ET.SubElement(inertial, "mass", value="0.0001")
            ET.SubElement(inertial, "inertia", ixx="0.000001", ixy="0", ixz="0", iyy="0.000001", iyz="0", izz="0.000001")
            ET.SubElement(joint, "child", link=inter_name)
            parent_link = inter_name
        else:
            # Last axis connects to the actual segment
            # Origin at -half_seg to place the child segment center correctly
            ET.SubElement(joint, "child", link=f"seg_{idx}")
