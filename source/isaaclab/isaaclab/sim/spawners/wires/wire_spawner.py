# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawner function for procedural wire chain articulations."""

from __future__ import annotations

import enum
import math
from typing import TYPE_CHECKING

import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab.sim.schemas import (
    ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg,
    MassPropertiesCfg,
    RigidBodyPropertiesCfg,
    define_articulation_root_properties,
    define_collision_properties,
    define_mass_properties,
    define_rigid_body_properties,
)
from isaaclab.sim.spawners.materials import physics_materials_cfg, visual_materials_cfg
from isaaclab.sim.utils import bind_physics_material, bind_visual_material, clone

if TYPE_CHECKING:
    from .wire_cfg import WireChainSpawnerCfg, WirePortCfg


@clone
def spawn_wire_chain(
    prim_path: str,
    cfg: WireChainSpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a rigid-body articulation chain of capsules connected by D6 joints.

    The wire is spawned as a single articulation with ``num_segments`` rigid capsule
    links connected by D6 joints. Each joint has angular drives configured as
    spring-dampers to give the wire its bending stiffness.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at.
        cfg: The wire chain spawner configuration.
        translation: The translation to apply to the prim w.r.t. its parent prim.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim.
        **kwargs: Additional keyword arguments.

    Returns:
        The created root prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    # Check prim doesn't already exist
    if prim_utils.is_prim_path_valid(prim_path):
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    wire = cfg.wire_cfg
    stage = get_current_stage()
    if wire.num_segments < 1 or wire.total_length <= 0 or wire.radius <= 0:
        raise ValueError("Wire segment count, length and radius must be positive.")
    limits = (wire.max_depenetration_velocity, wire.max_linear_velocity, wire.max_angular_velocity)
    if not all(math.isfinite(v) and v > 0 for v in (*limits, wire.total_length, wire.radius)):
        raise ValueError("Wire dimensions and speed limits must be finite and positive.")
    if wire.segment_shape not in {"capsule", "cylinder"}:
        raise ValueError("segment_shape must be 'capsule' or 'cylinder'.")
    seg_length = wire.total_length / wire.num_segments

    # -- Create root Xform
    prim_utils.create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation)

    # -- Determine layout axis
    axis_str = wire.init_axis.upper() if wire.init_axis != "auto" else "auto"
    if axis_str == "auto":
        origin = translation or (0.0, 0.0, 0.0)
        init_axis = _find_clear_axis(origin, wire.total_length, wire.radius, stage, prim_path)
    else:
        try:
            init_axis = Axis[axis_str]
        except KeyError:
            raise ValueError(
                f"Invalid init_axis: '{wire.init_axis}'. Must be 'auto', 'X', 'Y', or 'Z'."
            )

    # -- Create shared materials (once, on the root)
    _create_materials(prim_path, cfg, stage)

    # -- Build ordered chain nodes: [start_port?, seg_0..seg_n-1, end_port?]
    chain_nodes: list[dict[str, object]] = []

    if wire.start_port is not None:
        _create_wire_port(prim_path, "start", wire.start_port, wire.total_length, axis=init_axis)
        chain_nodes.append({"kind": "port", "name": "start", "path": f"{prim_path}/start_port", "port_cfg": wire.start_port})

    # -- Create segments
    for i in range(wire.num_segments):
        _create_segment(prim_path, i, seg_length, wire, cfg, stage, init_axis)
        chain_nodes.append({"kind": "segment", "index": i, "path": f"{prim_path}/seg_{i}"})

    if wire.end_port is not None:
        _create_wire_port(prim_path, "end", wire.end_port, wire.total_length, axis=init_axis)
        chain_nodes.append({"kind": "port", "name": "end", "path": f"{prim_path}/end_port", "port_cfg": wire.end_port})

    # -- Create joints between adjacent chain nodes
    for parent_node, child_node in zip(chain_nodes[:-1], chain_nodes[1:]):
        _create_chain_joint(parent_node, child_node, seg_length, wire, stage, init_axis)

    # -- Handle base fixation (fix the middle segment)
    if wire.fix_base:
        # mid_idx = wire.num_segments // 2
        _create_fixed_base_joint(prim_path, seg_length, stage, init_axis, seg_idx=0)

    # -- Apply articulation root API to the root Xform
    art_props = cfg.articulation_props
    if art_props is None:
        art_props = ArticulationRootPropertiesCfg(
            articulation_enabled=True,
            enabled_self_collisions=wire.self_collision,
            solver_position_iteration_count=wire.solver_position_iteration_count,
            solver_velocity_iteration_count=wire.solver_velocity_iteration_count,
        )
    define_articulation_root_properties(prim_path, art_props)

    return prim_utils.get_prim_at_path(prim_path)


# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------

class Axis(enum.IntEnum):
    """Layout axis for wire initialization."""

    X = 0
    Y = 1
    Z = 2


def _axis_vec(axis: Axis, val: float) -> Gf.Vec3f:
    """Return a ``Gf.Vec3f`` with *val* along *axis* and 0 elsewhere."""
    v = [0.0, 0.0, 0.0]
    v[axis] = val
    return Gf.Vec3f(*v)


def _quat_mul(q0: tuple[float, float, float, float], q1: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Multiply two quaternions in ``(w, x, y, z)`` format."""
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    return (
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    )


def _quat_from_euler_xyz(rot_xyz: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Convert XYZ Euler angles (radians) into quaternion ``(w, x, y, z)``."""
    rx, ry, rz = rot_xyz
    hx, hy, hz = 0.5 * rx, 0.5 * ry, 0.5 * rz
    cx, sx = math.cos(hx), math.sin(hx)
    cy, sy = math.cos(hy), math.sin(hy)
    cz, sz = math.cos(hz), math.sin(hz)

    qx = (cx, sx, 0.0, 0.0)
    qy = (cy, 0.0, sy, 0.0)
    qz = (cz, 0.0, 0.0, sz)
    return _quat_mul(_quat_mul(qx, qy), qz)


def _axis_quat(axis: Axis) -> tuple[float, float, float, float]:
    """Return the quaternion rotating +X into the selected wire axis."""
    if axis == Axis.X:
        return (1.0, 0.0, 0.0, 0.0)
    if axis == Axis.Y:
        s = math.sqrt(0.5)
        return (s, 0.0, 0.0, s)
    s = math.sqrt(0.5)
    return (s, 0.0, -s, 0.0)


def _aabb_overlap(
    a_min: Gf.Vec3d, a_max: Gf.Vec3d, b_min: Gf.Vec3d, b_max: Gf.Vec3d
) -> bool:
    """Return True if two axis-aligned bounding boxes overlap."""
    return (
        a_min[0] <= b_max[0]
        and a_max[0] >= b_min[0]
        and a_min[1] <= b_max[1]
        and a_max[1] >= b_min[1]
        and a_min[2] <= b_max[2]
        and a_max[2] >= b_min[2]
    )


def _find_clear_axis(
    origin: tuple[float, float, float],
    wire_length: float,
    wire_radius: float,
    stage: Usd.Stage,
    root_path: str,
) -> Axis:
    """Find the first axis (X, Y, Z) with enough clearance for the wire.

    For each candidate axis the function computes the bounding box the wire
    would occupy and checks for overlap with every existing ``UsdGeom.Boundable``
    prim on the stage (excluding the wire root and its descendants).

    Args:
        origin: World-space spawn position of the wire root.
        wire_length: Total end-to-end length of the wire.
        wire_radius: Radius of each wire segment.
        stage: The current USD stage.
        root_path: Prim path of the wire root (skipped during checks).

    Returns:
        The first :class:`Axis` with enough clearance.

    Raises:
        ValueError: If no axis has enough clearance.
    """
    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])

    # Collect world-space AABBs of all existing boundable prims.
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    scene_boxes: list[tuple[Gf.Vec3d, Gf.Vec3d]] = []
    for prim in stage.Traverse():
        path_str = str(prim.GetPath())
        if path_str == root_path or path_str.startswith(root_path + "/"):
            continue
        if prim.IsA(UsdGeom.Boundable):
            aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if not aligned.IsEmpty():
                scene_boxes.append((aligned.GetMin(), aligned.GetMax()))

    # Try each axis in order.
    r = wire_radius
    for axis in Axis:
        w_min = [ox - r, oy - r, oz - r]
        w_max = [ox + r, oy + r, oz + r]
        # Wire extends from origin along +axis for wire_length.
        w_min[axis] = origin[axis] - r
        w_max[axis] = origin[axis] + wire_length + r

        wire_min = Gf.Vec3d(*w_min)
        wire_max = Gf.Vec3d(*w_max)

        blocked = False
        for box_min, box_max in scene_boxes:
            if _aabb_overlap(wire_min, wire_max, box_min, box_max):
                blocked = True
                break

        if not blocked:
            return axis

    raise ValueError(
        f"Cannot spawn wire at '{root_path}': no clear axis found. "
        f"The wire requires {wire_length:.3f}m of clearance along at least one "
        f"axis (X, Y, Z) but all are blocked by existing scene geometry."
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _create_materials(
    root_path: str,
    cfg: WireChainSpawnerCfg,
    stage: Usd.Stage,
) -> None:
    """Create shared visual and physics materials under the root prim."""
    wire = cfg.wire_cfg
    mat_path = f"{root_path}/materials"

    # Visual material
    vis_mat_path = f"{mat_path}/visual"
    if cfg.visual_material is not None:
        cfg.visual_material.func(vis_mat_path, cfg.visual_material)
    else:
        # Create a default PreviewSurface from wire_cfg.color
        vis_cfg = visual_materials_cfg.PreviewSurfaceCfg(diffuse_color=wire.color)
        vis_cfg.func(vis_mat_path, vis_cfg)

    # Physics material
    phys_mat_path = f"{mat_path}/physics"
    if cfg.physics_material is not None:
        cfg.physics_material.func(phys_mat_path, cfg.physics_material)
    else:
        phys_cfg = physics_materials_cfg.RigidBodyMaterialCfg(
            static_friction=wire.static_friction,
            dynamic_friction=wire.dynamic_friction,
            restitution=wire.restitution,
        )
        phys_cfg.func(phys_mat_path, phys_cfg)


def _create_wire_port(
    root_path: str,
    port_name: str,
    port_cfg: WirePortCfg | None,
    wire_length: float,
    axis: Axis = Axis.X,
) -> None:
    """Create an optional attachment at the wire start or end port anchor.

    ``family`` and legacy ``port_type`` use the shared connector factory.
    ``usd_path`` remains an explicit generic endpoint USD reference.
    """
    if port_cfg is None:
        return

    if port_name not in {"start", "end"}:
        raise ValueError(f"Invalid port_name '{port_name}'. Expected 'start' or 'end'.")

    if sum(value is not None for value in (port_cfg.family, port_cfg.usd_path, port_cfg.port_type)) != 1:
        raise ValueError(
            "WirePortCfg: exactly one of family, usd_path or port_type must be set."
        )

    family = port_cfg.family or ("displayport_style" if port_cfg.port_type == "DP" else port_cfg.port_type)
    if family is not None:
        from isaaclab.sim.spawners.ports import PortCfg, get_connector_profile
        from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg

        profile = get_connector_profile(family)
        outward = (0.0, 0.0, 0.0, 1.0) if port_name == "start" else (1.0, 0.0, 0.0, 0.0)
        rotation = _quat_mul(_axis_quat(axis), _quat_mul(outward, _quat_from_euler_xyz(port_cfg.rot_xyz)))
        q = Gf.Quatf(rotation[0], Gf.Vec3f(*rotation[1:]))
        anchor = _axis_vec(axis, wire_length if port_name == "end" else 0.0)
        cable_point = Gf.Vec3f(profile.cable_anchor_x(port_cfg.cable_embed_depth), 0.0, 0.0)
        position = anchor + Gf.Vec3f(*port_cfg.trans_xyz) - q.Transform(cable_point)
        if port_cfg.scale is not None:
            raise ValueError("Family ports use dimensions in meters; do not scale the rigid-body frame.")
        inline_cfg = PortCfg(
            family=family,
            visual_usd_path=port_cfg.visual_usd_path,
            mass_props=MassPropertiesCfg(mass=port_cfg.mass),
            rigid_props=RigidBodyPropertiesCfg(
                max_depenetration_velocity=port_cfg.max_depenetration_velocity,
                max_linear_velocity=port_cfg.max_linear_velocity,
                max_angular_velocity=port_cfg.max_angular_velocity,
                linear_damping=port_cfg.linear_damping,
                angular_damping=port_cfg.angular_damping,
            ),
            physics_material=RigidBodyMaterialCfg(
                static_friction=port_cfg.static_friction, dynamic_friction=port_cfg.dynamic_friction,
                friction_combine_mode="max",
            ),
        )
        inline_cfg.func(f"{root_path}/{port_name}_port", inline_cfg,
                        translation=tuple(position), orientation=rotation)
        return

    anchor = _axis_vec(axis, wire_length if port_name == "end" else 0.0)
    offset = Gf.Vec3f(*port_cfg.trans_xyz)
    pos = anchor + offset

    axis_quat = _axis_quat(axis)
    euler_quat = _quat_from_euler_xyz(port_cfg.rot_xyz)
    orientation = _quat_mul(axis_quat, euler_quat)

    port_prim_path = f"{root_path}/{port_name}_port"

    # USD-reference branch (legacy).
    prim_utils.create_prim(
        port_prim_path,
        usd_path=port_cfg.usd_path,
        translation=(pos[0], pos[1], pos[2]),
        orientation=orientation,
        scale=port_cfg.scale,
        attributes=port_cfg.attributes,
    )


def _create_chain_joint(
    parent_node: dict[str, object],
    child_node: dict[str, object],
    seg_length: float,
    wire,
    stage: Usd.Stage,
    axis: Axis = Axis.X,
) -> None:
    """Create the correct joint type between two adjacent chain nodes."""
    parent_kind = parent_node["kind"]
    child_kind = child_node["kind"]

    if parent_kind == "segment" and child_kind == "segment":
        _create_segment_joint(
            parent_path=str(parent_node["path"]),
            child_path=str(child_node["path"]),
            seg_length=seg_length,
            wire=wire,
            stage=stage,
            axis=axis,
        )
        return

    if parent_kind == "port" and child_kind == "segment":
        _create_port_segment_joint(
            port_path=str(parent_node["path"]),
            port_name=str(parent_node["name"]),
            seg_path=str(child_node["path"]),
            port_is_parent=True,
            seg_length=seg_length,
            stage=stage,
            axis=axis,
            port_cfg=parent_node.get("port_cfg"),
        )
        return

    if parent_kind == "segment" and child_kind == "port":
        _create_port_segment_joint(
            port_path=str(child_node["path"]),
            port_name=str(child_node["name"]),
            seg_path=str(parent_node["path"]),
            port_is_parent=False,
            seg_length=seg_length,
            stage=stage,
            axis=axis,
            port_cfg=child_node.get("port_cfg"),
        )
        return

    raise ValueError(
        f"Unsupported chain adjacency: parent kind '{parent_kind}', child kind '{child_kind}'."
    )


def _create_port_segment_joint(
    port_path: str,
    port_name: str,
    seg_path: str,
    port_is_parent: bool,
    seg_length: float,
    stage: Usd.Stage,
    axis: Axis = Axis.X,
    port_cfg: WirePortCfg | None = None,
) -> None:
    """Create a fixed joint between a port and an adjacent segment."""
    joint_path = f"{port_path}/joint_to_wire"
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)

    if port_is_parent:
        joint.CreateBody0Rel().SetTargets([Sdf.Path(port_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(seg_path)])
    else:
        joint.CreateBody0Rel().SetTargets([Sdf.Path(seg_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(port_path)])

    # Match the two actual body frames at creation time, including endpoint
    # rotations. Sibling transforms must be computed via their world matrices.
    cache = UsdGeom.XformCache()
    port_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(port_path))
    segment_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(seg_path))
    relative = port_world * segment_world.GetInverse()
    q = relative.ExtractRotationQuat()
    segment_position = Gf.Vec3f(relative.ExtractTranslation())
    segment_rotation = Gf.Quatf(q.GetReal(), Gf.Vec3f(q.GetImaginary()))
    identity = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    if port_is_parent:
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.0))
        joint.CreateLocalRot0Attr(identity)
        joint.CreateLocalPos1Attr(segment_position)
        joint.CreateLocalRot1Attr(segment_rotation)
    else:
        joint.CreateLocalPos0Attr(segment_position)
        joint.CreateLocalRot0Attr(segment_rotation)
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        joint.CreateLocalRot1Attr(identity)
    joint.CreateCollisionEnabledAttr(False)
    joint.CreateExcludeFromArticulationAttr(False)
    # The port and capsule remain sibling rigid bodies with separate masses
    # and depenetration limits. Only nearby attachment pairs are filtered.
    index = int(seg_path.rsplit("seg_", 1)[1])
    neighbor = index + (1 if port_name == "start" else -1)
    neighbor_path = seg_path.rsplit("/", 1)[0] + f"/seg_{neighbor}"
    excluded = [Sdf.Path(seg_path)]
    if stage.GetPrimAtPath(neighbor_path).IsValid():
        excluded.append(Sdf.Path(neighbor_path))
    UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(port_path)).CreateFilteredPairsRel().SetTargets(excluded)


def _create_segment(
    root_path: str,
    idx: int,
    seg_length: float,
    wire,
    cfg: WireChainSpawnerCfg,
    stage: Usd.Stage,
    axis: Axis = Axis.X,
) -> None:
    """Create a rigid capsule segment, or an explicitly selected cylinder."""
    seg_path = f"{root_path}/seg_{idx}"
    geom_path = f"{seg_path}/geometry"
    mesh_path = f"{geom_path}/mesh"

    # Position: segments are laid out along the chosen axis
    offset = _axis_vec(axis, idx * seg_length + seg_length / 2.0)
    prim_utils.create_prim(seg_path, prim_type="Xform", translation=(offset[0], offset[1], offset[2]))

    # Adjacent capsules overlap while the two outer cable tips stay at 0/L.
    height, shift = seg_length, 0.0
    if wire.segment_shape == "capsule":
        overlap = 2.0 * wire.radius if wire.capsule_overlap is None else wire.capsule_overlap
        if not math.isfinite(overlap) or not 0 <= overlap < seg_length:
            raise ValueError("capsule_overlap must be finite and in [0, segment pitch).")
        if wire.num_segments == 1:
            height = seg_length - 2.0 * wire.radius
        elif idx in (0, wire.num_segments - 1):
            height = seg_length + overlap / 2.0 - 2.0 * wire.radius
            shift = overlap / 4.0 * (1 if idx == 0 else -1)
        else:
            height = seg_length + overlap - 2.0 * wire.radius
        if height <= 0:
            raise ValueError("Segment pitch is too short for the radius/overlap.")
    prim_utils.create_prim(
        mesh_path,
        prim_type="Capsule" if wire.segment_shape == "capsule" else "Cylinder",
        translation=tuple(_axis_vec(axis, shift)),
        attributes={"radius": wire.radius, "height": height, "axis": axis.name},
    )

    # Apply collision
    collision_cfg = cfg.collision_props
    if collision_cfg is None:
        collision_cfg = CollisionPropertiesCfg(
            collision_enabled=True, contact_offset=wire.contact_offset, rest_offset=wire.rest_offset
        )
    define_collision_properties(mesh_path, collision_cfg)

    # Apply rigid body
    rigid_cfg = cfg.rigid_props
    if rigid_cfg is None:
        rigid_cfg = RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_depenetration_velocity=wire.max_depenetration_velocity,
            max_linear_velocity=wire.max_linear_velocity,
            max_angular_velocity=wire.max_angular_velocity,
            linear_damping=wire.linear_damping,
            angular_damping=wire.angular_damping,
        )
    define_rigid_body_properties(seg_path, rigid_cfg)
    if wire.contact_slop_coefficient is not None:
        # Not a RigidBodyPropertiesCfg field in this Isaac Lab release; author the PhysX attribute directly.
        prim = stage.GetPrimAtPath(seg_path)
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateContactSlopCoefficientAttr().Set(float(wire.contact_slop_coefficient))

    # Apply mass via density
    mass_cfg = cfg.mass_props
    if mass_cfg is None:
        # Cap overlap changes contact geometry, not the nominal cable mass.
        mass_cfg = MassPropertiesCfg(mass=wire.density * math.pi * wire.radius**2 * seg_length)
    define_mass_properties(seg_path, mass_cfg)

    # Bind materials
    vis_mat_path = f"{root_path}/materials/visual"
    phys_mat_path = f"{root_path}/materials/physics"
    bind_visual_material(mesh_path, vis_mat_path)
    bind_physics_material(mesh_path, phys_mat_path)


def _create_segment_joint(
    parent_path: str,
    child_path: str,
    seg_length: float,
    wire,
    stage: Usd.Stage,
    axis: Axis = Axis.X,
) -> None:
    """Create a D6 joint between two adjacent wire segments."""
    # Unique joint name per segment avoids PhysX "Duplicate joint/DOF name" warnings.
    child_name = child_path.rsplit("/", 1)[-1]
    joint_path = f"{child_path}/joint_{child_name}_to_parent"

    # Define the D6 joint
    joint = UsdPhysics.Joint.Define(stage, joint_path)

    # Set body targets
    joint.CreateBody0Rel().SetTargets([Sdf.Path(parent_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(child_path)])

    # Set local poses: joint at the +axis end of parent, -axis end of child
    half_seg = seg_length / 2.0
    joint.CreateLocalPos0Attr().Set(_axis_vec(axis, half_seg))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(_axis_vec(axis, -half_seg))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    joint.CreateCollisionEnabledAttr(False)

    # Lock all translational DOFs (keep segments rigidly connected positionally)
    joint_prim = stage.GetPrimAtPath(joint_path)

    # Lock translational axes (low > high signals a locked DOF to PhysX)
    for dof in ["transX", "transY", "transZ"]:
        limit_api = UsdPhysics.LimitAPI.Apply(joint_prim, dof)
        limit_api.CreateLowAttr().Set(1.0)
        limit_api.CreateHighAttr().Set(-1.0)

    # Lock twist DOF (rotation around wire axis) — cables bend but don't twist
    twist_dof = f"rot{axis.name}"
    twist_limit = UsdPhysics.LimitAPI.Apply(joint_prim, twist_dof)
    twist_limit.CreateLowAttr().Set(1.0)
    twist_limit.CreateHighAttr().Set(-1.0)

    # Configure bending DOFs (perpendicular to wire axis) with angular drives
    bend_dofs = [f"rot{a}" for a in ["X", "Y", "Z"] if a != axis.name]
    for dof in bend_dofs:
        # Apply joint limits if specified
        if wire.joint_limit is not None:
            limit_api = UsdPhysics.LimitAPI.Apply(joint_prim, dof)
            limit_deg = math.degrees(wire.joint_limit)
            limit_api.CreateLowAttr().Set(-limit_deg)
            limit_api.CreateHighAttr().Set(limit_deg)

        # Apply angular drive (spring-damper)
        drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, dof)
        drive_api.CreateTypeAttr().Set("force")
        drive_api.CreateStiffnessAttr().Set(wire.stiffness)
        drive_api.CreateDampingAttr().Set(wire.damping)
        drive_api.CreateTargetPositionAttr().Set(0.0)


def _create_fixed_base_joint(
    root_path: str,
    seg_length: float,
    stage: Usd.Stage,
    axis: Axis = Axis.X,
    seg_idx: int = 0,
) -> None:
    """Create a fixed joint from a segment to the world (no body0 target = world)."""
    seg_path = f"{root_path}/seg_{seg_idx}"
    joint_path = f"{seg_path}/fixed_base_joint"

    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    # body0 = world (empty), body1 = chosen segment
    joint.CreateBody1Rel().SetTargets([Sdf.Path(seg_path)])

    # Joint frame at the center of the chosen segment
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
