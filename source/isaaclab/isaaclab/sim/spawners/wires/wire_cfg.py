# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.spawners import materials
from isaaclab.sim.schemas import ArticulationRootPropertiesCfg
from isaaclab.utils import configclass

from . import wire_spawner


@configclass
class WirePortCfg:
    """Configuration for spawning an optional port at a wire endpoint.

    Exactly one of ``family``, ``usd_path`` or ``port_type`` must be set:

    - ``family``: use the shared printable/real connector profile.
    - ``usd_path``: reference an explicitly supplied endpoint USD.
    - ``port_type``: legacy spelling for the same family; ``DP`` maps to
      ``displayport_style``. It does not select a separate geometry builder.
    """

    family: str | None = None
    """Shared printable/real profile; same port geometry for articulation and FEM."""

    visual_usd_path: str | None = None
    """Geometry-only visual resolved by isaaclab_assets; no machine-specific default."""

    cable_embed_depth: float = 0.008
    """Length of cable inside the housing, from its back, in meters. 15 mm ends the cable at the tip
    shoulder (the port origin) of a printable port."""

    max_depenetration_velocity: float = 5.0
    max_linear_velocity: float = 5.0
    max_angular_velocity: float = 60.0
    """Independent rigid-port limits: m/s, m/s, deg/s, respectively."""

    linear_damping: float | None = None
    angular_damping: float | None = None
    """Port rigid-body damping; None keeps the PhysX default."""

    usd_path: str | None = None
    """Path to the USD asset to attach at the wire endpoint (mutually exclusive with port_type)."""

    port_type: str | None = None
    """Procedural port type (mutually exclusive with usd_path).
    One of ``"cylinder" | "cuboid" | "stacked_cuboid" | "DP" | "usb_a"``."""

    mass: float = 0.01
    """Mass applied when spawning a procedural port. Ignored when usd_path is set."""

    static_friction: float = 2.0
    """Static friction applied when spawning a procedural port. Ignored when usd_path is set."""

    dynamic_friction: float = 2.0
    """Dynamic friction applied when spawning a procedural port. Ignored when usd_path is set."""

    trans_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Local translation offset from the endpoint anchor in meters."""

    rot_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Local XYZ Euler rotation offset in radians."""

    scale: tuple[float, float, float] | None = None
    """Optional local scale of the endpoint USD. If None, USD default scale is used."""

    attributes: dict[str, object] | None = None
    """Optional additional USD attributes applied during prim creation."""


@configclass
class WireCfg:
    """High-level configuration describing a procedural wire (rigid-body articulation chain)."""

    num_segments: int = 20
    """Number of rigid wire segments. Defaults to 20."""

    total_length: float = 0.3
    """End-to-end length of the wire in meters. Defaults to 0.3."""

    radius: float = 0.002
    """Segment radius in meters. Defaults to 0.002."""

    stiffness: float = 1.0
    """D6 joint angular drive stiffness in Nm/rad. Defaults to 1.0."""

    damping: float = 0.1
    """D6 joint angular drive damping. Defaults to 0.1."""

    joint_limit: float | None = None
    """Max bend angle per joint in radians. None = unlimited. Defaults to None."""

    self_collision: bool = True
    """Enable self-collisions between segments. Defaults to True."""

    fix_base: bool = False
    """Whether root link is fixed to world. Defaults to False."""

    density: float = 1000.0
    """Density in kg/m^3 for auto mass computation. Defaults to 1000.0."""

    static_friction: float = 1.0
    """Static friction coefficient. Defaults to 1.0."""

    dynamic_friction: float = 1.0
    """Dynamic friction coefficient. Defaults to 1.0."""

    restitution: float = 0.0
    """Restitution coefficient. Defaults to 0.0."""

    color: tuple[float, float, float] = (0.2, 0.2, 0.2)
    """RGB color for visual material. Defaults to (0.2, 0.2, 0.2)."""

    init_axis: str = "auto"
    """Axis to lay the wire along at spawn time.

    - ``"auto"``: checks X, Y, Z in order and picks the first with enough clearance.
    - ``"X"``, ``"Y"``, ``"Z"``: force a specific axis.

    Defaults to ``"auto"``.
    """

    solver_position_iteration_count: int = 32
    """Solver position iteration count. Defaults to 32."""

    solver_velocity_iteration_count: int = 4
    """Solver velocity iteration count. Defaults to 4."""

    segment_shape: str = "capsule"
    """Capsule by default; cylinder retains the straight-sided alternative."""

    capsule_overlap: float | None = None
    """Adjacent cap overlap in meters. None uses the diameter; 0 makes tips meet."""

    max_depenetration_velocity: float = 0.1
    """Wire segment depenetration limit in m/s; independent of the port."""

    max_linear_velocity: float = 5.0
    """Wire segment linear velocity limit in m/s."""

    max_angular_velocity: float = 60.0
    """Wire segment angular velocity limit in deg/s."""

    linear_damping: float | None = None
    """Wire segment linear damping; None keeps the PhysX default."""

    angular_damping: float | None = None
    """Wire segment angular damping; None keeps the PhysX default."""

    contact_offset: float | None = None
    """Wire segment collision contact offset in meters; None keeps the PhysX default."""

    rest_offset: float | None = None
    """Wire segment collision rest offset in meters; None keeps the PhysX default."""

    contact_slop_coefficient: float | None = None
    """Wire segment contact slop coefficient; None keeps the PhysX default."""

    joint_velocity_limit: float | None = None
    """Velocity limit of the joint drives in rad/s; None leaves them unlimited."""

    joint_armature: float | None = None
    """Armature of the joint drives; None keeps the asset value."""

    start_port: WirePortCfg | None = None
    """Optional USD attachment to spawn at the beginning of the wire."""

    end_port: WirePortCfg | None = None
    """Optional USD attachment to spawn at the end of the wire."""


@configclass
class WireChainSpawnerCfg(RigidObjectSpawnerCfg):
    """Configuration parameters for spawning a wire chain articulation.

    This spawner creates a rigid-body articulation chain of capsules connected
    by D6 joints at runtime from a :class:`WireCfg` description.
    """

    func: Callable = wire_spawner.spawn_wire_chain

    wire_cfg: WireCfg = WireCfg()
    """Wire configuration describing geometry, physics, and joint properties."""

    visual_material_path: str = "material"
    """Path to the visual material to use for the prim. Defaults to "material"."""

    visual_material: materials.VisualMaterialCfg | None = None
    """Visual material properties. If None, a default PreviewSurface is created from wire_cfg.color."""

    physics_material_path: str = "material"
    """Path to the physics material to use for the prim. Defaults to "material"."""

    physics_material: materials.PhysicsMaterialCfg | None = None
    """Physics material properties. If None, one is created from wire_cfg friction/restitution."""

    articulation_props: ArticulationRootPropertiesCfg | None = None
    """Articulation root properties. If None, auto-populated from wire_cfg."""
