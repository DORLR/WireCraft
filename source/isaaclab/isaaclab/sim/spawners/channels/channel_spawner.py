# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Open U-channels matching the packaged printable modules.

Lengths are in meters. Local Z=0 is the underside of the floor; local X
follows the straight channel or the chord of the curved module. Curves are
centered on their midpoint and have a symmetric range of tangent angles.
"""

from __future__ import annotations

import math

from pxr import Gf, UsdGeom, UsdPhysics
from isaacsim.core.utils.stage import get_current_stage

from isaaclab.sim import schemas
from isaaclab.sim.schemas import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg, RigidBodyMaterialCfg
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import bind_physics_material, bind_visual_material, clone
from isaaclab.utils import configclass


def _channel_centerline(cfg):
    if cfg.channel_type not in ("straight", "curved"):
        raise ValueError("channel_type must be 'straight' or 'curved'.")
    length = cfg.length if cfg.length is not None else (0.110 if cfg.channel_type == "straight" else 0.066)
    if not all(math.isfinite(v) and v > 0 for v in (length, cfg.width, cfg.depth, cfg.wall_thickness)):
        raise ValueError("Channel length, inner width, depth and wall thickness must be positive and finite.")
    if cfg.channel_type == "straight":
        return [(-length / 2, 0.0), (length / 2, 0.0)]
    angle = math.radians(cfg.bend_angle_deg)
    if not math.isfinite(angle) or not 0 < abs(angle) < math.pi:
        raise ValueError("A curved channel requires 0 < abs(bend_angle_deg) < 180.")
    if not isinstance(cfg.curve_segments, int) or isinstance(cfg.curve_segments, bool) or cfg.curve_segments < 2:
        raise ValueError("curve_segments must be an integer >= 2.")
    radius = length / abs(angle)
    if radius * math.cos(abs(angle) / (2 * cfg.curve_segments)) <= cfg.width / 2 + cfg.wall_thickness:
        raise ValueError("Bend radius/segment count must leave positive width at the inner miter boundary.")
    sign = math.copysign(1.0, angle)
    angles = [-abs(angle) / 2 + abs(angle) * i / cfg.curve_segments for i in range(cfg.curve_segments + 1)]
    return [(radius * math.sin(a), sign * radius * (1 - math.cos(a))) for a in angles]


def _offset_centerline(points, distance):
    """Miter offsets of a polyline, with flat ends, as in the printable CAD."""
    normals = []
    for a, b in zip(points, points[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        normals.append((-dy / length, dx / length))
    result = []
    for i, point in enumerate(points):
        if i == 0:
            nx, ny = normals[0]
        elif i == len(points) - 1:
            nx, ny = normals[-1]
        else:
            a, b = normals[i - 1], normals[i]
            scale = 1 + a[0] * b[0] + a[1] * b[1]
            nx, ny = (a[0] + b[0]) / scale, (a[1] + b[1]) / scale
        result.append((point[0] + distance * nx, point[1] + distance * ny))
    return result


def _channel_parts(cfg):
    points = _channel_centerline(cfg)
    half = cfg.width / 2
    left_outer = _offset_centerline(points, half + cfg.wall_thickness)
    left_inner = _offset_centerline(points, half)
    right_inner = _offset_centerline(points, -half)
    right_outer = _offset_centerline(points, -half - cfg.wall_thickness)
    parts = []
    for i in range(len(points) - 1):
        sections = (
            ("floor", left_outer, right_outer, 0.0, cfg.wall_thickness),
            ("left", left_outer, left_inner, cfg.wall_thickness, cfg.wall_thickness + cfg.depth),
            ("right", right_inner, right_outer, cfg.wall_thickness, cfg.wall_thickness + cfg.depth),
        )
        for name, outer, inner, z0, z1 in sections:
            outline = [outer[i], inner[i], inner[i + 1], outer[i + 1]]
            area = sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(outline, outline[1:] + outline[:1]))
            if area < 0:
                outline.reverse()
            parts.append((f"{name}_{i}", outline, z0, z1))
    return parts


def _prism_arrays(outline, z0, z1):
    n = len(outline)
    points = [(x, y, z) for z in (z0, z1) for x, y in outline]
    faces = [list(reversed(range(n))), list(range(n, 2 * n))]
    faces += [[i, (i + 1) % n, (i + 1) % n + n, i + n] for i in range(n)]
    return points, faces


def _define_mesh(stage, path, points, faces):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateExtentAttr([Gf.Vec3f(*(min(p[i] for p in points) for i in range(3))),
                           Gf.Vec3f(*(max(p[i] for p in points) for i in range(3)))])
    return mesh


@clone
def spawn_channel(prim_path: str, cfg: ChannelCfg, translation=None, orientation=None):
    """Spawn one static open channel; collision pieces never bridge its slot."""
    parts = _channel_parts(cfg)
    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at {prim_path}.")
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    root.AddOrientOp().Set(Gf.Quatf(*(orientation or (1.0, 0.0, 0.0, 0.0))))
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage)
    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props, stage)
    material = cfg.physics_material_path
    if not material.startswith("/"):
        material = prim_path + "/" + material
    cfg.physics_material.func(material, cfg.physics_material)
    UsdGeom.Scope.Define(stage, prim_path + "/colliders")
    all_points, all_faces = [], []
    for name, outline, z0, z1 in parts:
        points, faces = _prism_arrays(outline, z0, z1)
        path = prim_path + "/colliders/" + name
        mesh = _define_mesh(stage, path, points, faces)
        mesh.CreatePurposeAttr("guide")
        schemas.define_collision_properties(path, cfg.collision_props, stage)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
        bind_physics_material(path, material)
        offset = len(all_points)
        all_points.extend(points)
        all_faces.extend([[offset + i for i in face] for face in faces])
    visual_path = prim_path + "/visual"
    _define_mesh(stage, visual_path, all_points, all_faces)
    visual_material = prim_path + "/visual_material"
    cfg.visual_material.func(visual_material, cfg.visual_material)
    bind_visual_material(visual_path, visual_material)
    return root.GetPrim()


@configclass
class ChannelCfg(RigidObjectSpawnerCfg):
    """Straight or curved printable-module geometry, in meters/wxyz.

    The floor extends from Z=0 to ``wall_thickness``; ``depth`` is the open
    slot height above it. ``width`` is the inner slot width. Defaults match
    the shipped 110 mm straight and 66 mm / 10 degree curved STEP modules.
    """

    func = spawn_channel
    channel_type: str = "straight"
    length: float | None = None
    """Straight length or curved centerline arc length; None uses 0.110/0.066 m."""
    width: float = 0.010
    depth: float = 0.010
    wall_thickness: float = 0.005
    bend_angle_deg: float = 10.0
    """Curved total turn: positive bends toward +Y, negative mirrors toward -Y."""
    curve_segments: int = 40
    """Polyline chord count; 40 matches the printable curved CAD."""
    rigid_props: RigidBodyPropertiesCfg | None = None
    """None keeps the fixture static; an explicit rigid-body config may override."""
    collision_props: CollisionPropertiesCfg = CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0)
    physics_material_path: str = "material"
    physics_material: RigidBodyMaterialCfg = RigidBodyMaterialCfg(
        static_friction=2.0, dynamic_friction=2.0, friction_combine_mode="max",
    )
    visual_material: PreviewSurfaceCfg = PreviewSurfaceCfg(diffuse_color=(0.25, 0.45, 0.7), roughness=0.65)
