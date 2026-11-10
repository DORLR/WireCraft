# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Inputs for a wire with a port at each end; spawn configs are derived."""

from __future__ import annotations

from isaaclab.sim.spawners.wires import ProceduralDeformableWireCfg, WireAssemblyCfg, WireCfg
from isaaclab.utils import configclass

from .connectors import ConnectorCfg


@configclass
class WireSceneCfg:
    """Physical parameters are the values confirmed for the current wires."""

    kind: str = "articulation"
    """``articulation`` (capsule chain) or ``deformable`` (FEM)."""
    root_pos: tuple[float, float, float] = (0.15, 0.0, 0.805)
    """Cable X=0, env-local. The cable runs along +X, so the insertion port is nearest the socket."""
    length: float = 0.4
    color: tuple[float, float, float] = (0.05, 0.05, 0.05)
    centerline_points: int = 41

    # articulation: the cable the v14.5 insertion policies were trained on
    num_segments: int = 12
    radius: float = 0.003
    stiffness: float = 0.3015928947446202
    """Bending drive stiffness of each joint, Nm/rad."""
    damping: float = 0.005
    joint_limit: float = 0.5
    joint_velocity_limit: float = 40.0
    """rad/s, on the joint drives."""
    joint_armature: float = 1.0e-6
    density: float = 800.0
    solver_position_iterations: int = 64
    """Below this a held chain buzzes at 2-12 rad/s between the fingers and the far hold, and diverges."""
    solver_velocity_iterations: int = 4
    capsule_overlap: float | None = 0.008
    """Axial overlap of adjacent capsules in meters; None keeps the spawner's one diameter."""
    max_linear_velocity: float = 5.0
    max_angular_velocity: float = 60.0
    """deg/s, as PhysX stores it."""
    max_depenetration_velocity: float = 0.1
    segment_linear_damping: float = 5.0
    segment_angular_damping: float = 5.0
    contact_offset: float = 0.001
    rest_offset: float = 0.0
    contact_slop_coefficient: float = 0.0005
    port_linear_damping: float = 5.0
    port_angular_damping: float = 0.5
    port_cable_embed_depth: float = 0.015
    """Articulation cable inside each port's housing: 15 mm ends it at the tip shoulder, the port origin,
    as the insertion policies were trained. The FEM cable keeps the spawner's 8 mm."""

    # deformable
    fem_num_segments: int = 40
    fem_radius: float = 0.003
    fem_youngs_modulus: float = 5.0e7
    fem_density: float = 800.0
    fem_dynamic_friction: float = 5.0
    attachment_offset: float = 0.015
    """FEM vertex capture margin in meters."""

    # reset randomization
    randomize_xy: bool = False
    randomize_xy_range: tuple[float, float] = (0.05, 0.15)
    """Half-ranges in X and Y."""
    randomize_yaw: float = 0.0
    """Half-range in radians about the wire centre."""
    randomize_tilt: float = 0.0
    """Half-range of roll and of pitch in radians about the wire centre."""
    randomize_lift: tuple[float, float] = (0.0, 0.0)
    """Range of the height in meters the wire is raised by at reset; it then drops onto the table."""

    def validate(self):
        if self.kind not in ("articulation", "deformable"):
            raise ValueError(f"wire.kind must be 'articulation' or 'deformable', got {self.kind!r}.")
        if self.length <= 0 or self.centerline_points < 2:
            raise ValueError("wire.length must be positive and wire.centerline_points at least 2.")

    def assembly_cfg(self, connector: ConnectorCfg) -> WireAssemblyCfg:
        """Assembly spawn config derived from the final inputs."""
        self.validate()
        if self.kind == "articulation":
            wire = WireCfg(
                num_segments=self.num_segments,
                total_length=self.length,
                radius=self.radius,
                stiffness=self.stiffness,
                damping=self.damping,
                joint_limit=self.joint_limit,
                fix_base=False,
                density=self.density,
                self_collision=True,
                color=tuple(self.color),
                init_axis="X",
                solver_position_iteration_count=self.solver_position_iterations,
                solver_velocity_iteration_count=self.solver_velocity_iterations,
                capsule_overlap=self.capsule_overlap,
                max_linear_velocity=self.max_linear_velocity,
                max_angular_velocity=self.max_angular_velocity,
                max_depenetration_velocity=self.max_depenetration_velocity,
                linear_damping=self.segment_linear_damping,
                angular_damping=self.segment_angular_damping,
                contact_offset=self.contact_offset,
                rest_offset=self.rest_offset,
                contact_slop_coefficient=self.contact_slop_coefficient,
                joint_velocity_limit=self.joint_velocity_limit,
                joint_armature=self.joint_armature,
            )
        else:
            wire = ProceduralDeformableWireCfg(
                total_length=self.length,
                num_segments=self.fem_num_segments,
                radius=self.fem_radius,
                color=tuple(self.color),
                dynamic_friction=self.fem_dynamic_friction,
                youngs_modulus=self.fem_youngs_modulus,
                density=self.fem_density,
            )
        ports = [connector.port_cfg() for _ in range(2)]
        if self.kind == "articulation":
            for port in ports:
                port.linear_damping, port.angular_damping = self.port_linear_damping, self.port_angular_damping
                port.cable_embed_depth = self.port_cable_embed_depth
        return WireAssemblyCfg(
            wire=wire,
            start_port=ports[0],
            end_port=ports[1],
            attachment_offset=self.attachment_offset,
        )
