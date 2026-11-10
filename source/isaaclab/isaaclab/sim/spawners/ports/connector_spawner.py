# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared procedural port/socket factory; +X is the insertion axis, meters/wxyz.

Socket colliders are separate convex wall pieces. A convex hull is never
applied across the complete socket, which would close its opening.
"""

from __future__ import annotations

import math
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from isaacsim.core.utils.stage import get_current_stage

from isaaclab.sim import schemas
from isaaclab.sim.schemas import RigidBodyPropertiesCfg, MassPropertiesCfg, CollisionPropertiesCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import bind_physics_material, clone
from isaaclab.utils import configclass

from .connector_profiles import get_connector_profile, rectangle


def _area(points):
    return sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(points, points[1:] + points[:1])) / 2


def _offset_polygon(points, distance):
    """Offset convex CCW edges outward by a per-side distance (not a scale)."""
    lines = []
    for a, b in zip(points, points[1:] + points[:1]):
        dy, dz = b[0] - a[0], b[1] - a[1]
        norm = math.hypot(dy, dz)
        ny, nz = dz / norm, -dy / norm
        lines.append((ny, nz, ny * a[0] + nz * a[1] + distance))
    result = []
    for previous, current in zip(lines[-1:] + lines[:-1], lines):
        a, b, c = previous
        d, e, f = current
        determinant = a * e - b * d
        result.append(((c * e - b * f) / determinant, (a * f - c * d) / determinant))
    return result


def _outside_edge(points, a, b):
    """Clip a convex polygon against the outside half-plane of a CCW edge."""
    def side(p):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    result = []
    for p, q in zip(points, points[1:] + points[:1]):
        sp, sq = side(p), side(q)
        if sp <= 0:
            result.append(p)
        if (sp < 0 < sq) or (sq < 0 < sp):
            t = sp / (sp - sq)
            result.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return result


def _prism(stage, path, x_min, x_max, outline, color, collision_cfg, material_path):
    if len(outline) < 3 or abs(_area(outline)) < 1e-14:
        return
    points = [Gf.Vec3f(x, y, z) for x in (x_min, x_max) for y, z in outline]
    n = len(outline)
    faces = [list(reversed(range(n))), list(range(n, 2 * n))]
    faces.extend([i, (i + 1) % n, (i + 1) % n + n, i + n] for i in range(n))
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([v for f in faces for v in f])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    mesh.CreateExtentAttr([Gf.Vec3f(x_min, min(p[0] for p in outline), min(p[1] for p in outline)),
                           Gf.Vec3f(x_max, max(p[0] for p in outline), max(p[1] for p in outline))])
    schemas.define_collision_properties(path, collision_cfg, stage)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    bind_physics_material(path, material_path)


def _box(stage, path, bounds, color, collision_cfg, material_path):
    x0, x1, y0, y1, z0, z1 = bounds
    if min(x1 - x0, y1 - y0, z1 - z0) <= 1e-10:
        return
    _prism(stage, path, x0, x1, list(rectangle(y0, y1, z0, z1)), color, collision_cfg, material_path)


def _create_root(path, cfg, translation, orientation):
    stage = get_current_stage()
    if stage.GetPrimAtPath(path).IsValid():
        raise ValueError(f"A prim already exists at {path}.")
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    root.AddOrientOp().Set(Gf.Quatf(*(orientation or (1.0, 0.0, 0.0, 0.0))))
    material_path = cfg.physics_material_path
    if not material_path.startswith("/"):
        material_path = path + "/" + material_path
    cfg.physics_material.func(material_path, cfg.physics_material)
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(path, cfg.rigid_props, stage)
    if cfg.mass_props is not None:
        schemas.define_mass_properties(path, cfg.mass_props, stage)
    contact = cfg.collision_props
    if contact is None:
        offset = 0.0005 if get_connector_profile(cfg.family).category == "printable" else 0.0001
        contact = CollisionPropertiesCfg(contact_offset=offset, rest_offset=0.0)
    return stage, root, contact, material_path


def _attach_visual(stage, path, cfg, offset_x=0.0):
    """Reference a geometry-only asset; keep the procedural colliders as guides."""
    if cfg.visual_usd_path is None:
        return
    import os
    if not os.path.isfile(cfg.visual_usd_path):
        raise FileNotFoundError(cfg.visual_usd_path)
    visual = UsdGeom.Xform.Define(stage, path + "/visual")
    visual.GetPrim().GetReferences().AddReference(cfg.visual_usd_path)
    visual.AddTranslateOp().Set(Gf.Vec3d(offset_x, 0.0, 0.0))
    for child in stage.GetPrimAtPath(path).GetChildren():
        if child.IsA(UsdGeom.Mesh):
            UsdGeom.Imageable(child).CreatePurposeAttr("guide")


@clone
def spawn_connector_port(prim_path: str, cfg: PortCfg, translation=None, orientation=None) -> Usd.Prim:
    """Spawn an independent rigid port using the same geometry for either wire."""
    profile = get_connector_profile(cfg.family)
    stage, root, contact, material = _create_root(prim_path, cfg, translation, orientation)
    colors = cfg.part_colors or {}
    _prism(stage, prim_path + "/tip", 0.0, profile.tip_length, list(profile.tip_outline), colors.get("tip", cfg.color), contact, material)
    for i, bounds in enumerate(profile.housing_sections or (profile.housing_bounds,)):
        name = "housing" if i == 0 else f"housing_{i}"
        _box(stage, prim_path + "/" + name, bounds, colors.get("housing", cfg.color), contact, material)
    latch = profile.collision_latch_bounds
    if latch is not None:
        _box(stage, prim_path + "/latch", latch, colors.get("latch", cfg.color), contact, material)
    _attach_visual(stage, prim_path, cfg)
    return root.GetPrim()


@clone
def spawn_connector_socket(prim_path: str, cfg: SocketCfg, translation=None, orientation=None) -> Usd.Prim:
    """Spawn a paired socket with configurable cavity and mounting panel.

    Real profiles use an empty equivalent cavity: tongues, pins and contacts
    are not colliders. Printable profiles use their configured insertion shape.
    """
    profile = get_connector_profile(cfg.family)
    latch = profile.collision_latch_bounds
    clearance = profile.clearance if cfg.clearance is None else cfg.clearance
    depth = profile.socket_depth if cfg.depth is None else cfg.depth
    outer = profile.socket_outer_bounds if cfg.outer_bounds_yz is None else cfg.outer_bounds_yz
    values = [clearance, depth, cfg.mouth_x, cfg.backstop_thickness, *outer]
    if not all(math.isfinite(x) for x in values) or clearance < 0 or depth <= 0 or cfg.backstop_thickness < 0:
        raise ValueError("Socket dimensions must be finite; clearance/backstop >= 0 and depth > 0.")
    hole = _offset_polygon(list(profile.tip_outline), clearance)
    y0, y1 = min(p[0] for p in hole), max(p[0] for p in hole)
    z0, z1 = min(p[1] for p in hole), max(p[1] for p in hole)
    oy0, oy1, oz0, oz1 = outer
    latch_top = z1
    if latch is not None:
        latch_top = max(z1, latch[5] + clearance)
    if not (oy0 < y0 < y1 < oy1 and oz0 < z0 < latch_top < oz1):
        raise ValueError("Socket outer bounds must leave positive walls around the configured opening.")
    panel = (profile.socket_mount_panel_size if cfg.mount_panel_size is None else cfg.mount_panel_size) if cfg.mount_panel_enabled else None
    if panel is not None:
        if len(panel) != 3 or not all(math.isfinite(v) and v > 0 for v in panel):
            raise ValueError("Mounting geometry requires positive finite Y size, Z size and X extent.")
        py, pz, _ = panel
        if not (-py / 2 < oy0 < oy1 < py / 2 and -pz / 2 < oz0 < oz1 < pz / 2):
            raise ValueError("Mounting panel must extend beyond all socket outer bounds.")
    stage, root, contact, material = _create_root(prim_path, cfg, translation, orientation)
    x0, x1 = cfg.mouth_x, cfg.mouth_x + depth
    walls = {
        "left": (x0, x1, oy0, y0, oz0, oz1),
        "right": (x0, x1, y1, oy1, oz0, oz1),
        "bottom": (x0, x1, y0, y1, oz0, z0),
        "top": (x0, x1, y0, y1, latch_top, oz1),
    }
    if latch is not None:
        ly0 = latch[2] - clearance
        ly1 = latch[3] + clearance
        walls["shoulder_left"] = (x0, x1, y0, ly0, z1, latch_top)
        walls["shoulder_right"] = (x0, x1, ly1, y1, z1, latch_top)
    for name, bounds in walls.items():
        _box(stage, prim_path + "/" + name, bounds, cfg.color, contact, material)
    # Fill the corner regions outside a convex tip profile. For DisplayPort
    # this is exactly four box walls plus one keyed prism, not thin triangles.
    inner_rectangle = list(rectangle(y0, y1, z0, z1))
    for i, (a, b) in enumerate(zip(hole, hole[1:] + hole[:1])):
        polygon = _outside_edge(inner_rectangle, a, b)
        _prism(stage, prim_path + f"/key_{i}", x0, x1, polygon, cfg.color, contact, material)
    if cfg.backstop_thickness:
        _box(stage, prim_path + "/backstop", (x1, x1 + cfg.backstop_thickness, *outer), cfg.color, contact, material)
    if cfg.inner_color is not None:
        center = Gf.Vec3f(0.0, sum(p[0] for p in hole) / len(hole), sum(p[1] for p in hole) / len(hole))
        for child in root.GetPrim().GetChildren():
            if not child.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(child)
            vertices = mesh.GetPointsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            colors, cursor = [], 0
            for count in mesh.GetFaceVertexCountsAttr().Get():
                points = [vertices[i] for i in indices[cursor:cursor + count]]
                normal = Gf.Cross(points[1] - points[0], points[2] - points[0])
                midpoint = sum(points, Gf.Vec3f(0.0)) / count
                inward = abs(normal[0]) < 1e-10 and Gf.Dot(normal, center - midpoint) > 0
                inward = inward or (child.GetName() == "backstop" and normal[0] < 0)
                colors.append(Gf.Vec3f(*(cfg.inner_color if inward else cfg.color)))
                cursor += count
            mesh.CreateDisplayColorAttr(colors)
            mesh.GetDisplayColorPrimvar().SetInterpolation(UsdGeom.Tokens.uniform)
    _attach_visual(stage, prim_path, cfg, offset_x=cfg.mouth_x)
    if panel is not None:
        py, pz, thickness = panel
        # Convex pieces keep the socket mouth unobstructed; no hull bridges it.
        # The keyed faceplate seats behind the metal mouth lip. Its opening
        # uses the clear cavity profile so the lip covers the outside seam;
        # the port/socket insertion clearance remains independent.
        keyed = latch is None and len(hole) > 4
        opening = hole if keyed else list(rectangle(oy0, oy1, oz0, oz1))
        face_x = x0 + (0.0002 if keyed else 0.0)
        rear_x = max(face_x + thickness, x1 + cfg.backstop_thickness) if keyed else face_x + thickness
        remaining = list(rectangle(-py / 2, py / 2, -pz / 2, pz / 2))
        for i, (a, b) in enumerate(zip(opening, opening[1:] + opening[:1])):
            # Partition the ring into disjoint convex pieces; no hull spans
            # the opening, including its sloped keyed corners.
            outside = _outside_edge(remaining, a, b)
            _prism(stage, prim_path + f"/mount_panel_edge_{i}", face_x, rear_x,
                   outside, cfg.mount_panel_color, contact, material)
            remaining = _outside_edge(remaining, b, a)
        if keyed and cfg.backstop_thickness:
            # Close the complete enclosure behind the configured cavity.
            # A zero backstop retains an open rear passage.
            _box(stage, prim_path + "/mount_panel_back",
                 (max(x1, rear_x - cfg.backstop_thickness), rear_x,
                  -py / 2, py / 2, -pz / 2, pz / 2),
                 cfg.mount_panel_color, contact, material)
    return root.GetPrim()


@configclass
class PortCfg(RigidObjectSpawnerCfg):
    """Rigid port. Physics limits are independent of the wire segments."""
    func = spawn_connector_port
    family: str = "stacked_cuboid"
    color: tuple[float, float, float] = (0.15, 0.4, 0.75)
    part_colors: dict[str, tuple[float, float, float]] | None = None
    """Optional tip/housing/latch color overrides; geometry always comes from the profile."""
    physics_material_path: str = "material"
    visual_usd_path: str | None = None
    """Optional package-resolved geometry-only visual, in the same shoulder frame."""
    rigid_props: RigidBodyPropertiesCfg = RigidBodyPropertiesCfg(
        max_depenetration_velocity=5.0, max_linear_velocity=5.0, max_angular_velocity=60.0,
    )
    mass_props: MassPropertiesCfg = MassPropertiesCfg(mass=0.01)
    physics_material: RigidBodyMaterialCfg = RigidBodyMaterialCfg(
        static_friction=2.0, dynamic_friction=2.0, friction_combine_mode="max",
    )


@configclass
class SocketCfg(RigidObjectSpawnerCfg):
    """Paired socket; all geometric lengths below are in meters."""
    func = spawn_connector_socket
    family: str = "stacked_cuboid"
    clearance: float | None = None
    """Per-side outward clearance; None uses 3 mm printable / 0.25 mm real defaults."""
    depth: float | None = None
    """Cavity depth from mouth to backstop. None uses the family default."""
    mouth_x: float = -0.01
    outer_bounds_yz: tuple[float, float, float, float] | None = None
    """Local (y_min, y_max, z_min, z_max), independently configurable from clearance."""
    mount_panel_enabled: bool = True
    """Spawn the family mounting enclosure/panel. Disable for a bare socket."""
    mount_panel_size: tuple[float, float, float] | None = None
    """Mounting geometry (Y size, Z size, minimum X extent).

    None uses the profile default: DisplayPort/HDMI get full-depth enclosures;
    Ethernet/USB-A and printable profiles have no extra mounting geometry.
    Keyed enclosures start 0.2 mm behind the metal lip, extend through the
    configured cavity/backstop, and close at the rear when backstop > 0.
    The insertion opening and its clearance remain independent.
    """
    mount_panel_color: tuple[float, float, float] = (0.28, 0.31, 0.35)
    backstop_thickness: float = 0.002
    """Zero makes a through socket."""
    color: tuple[float, float, float] = (0.75, 0.2, 0.12)
    inner_color: tuple[float, float, float] | None = None
    physics_material_path: str = "material"
    visual_usd_path: str | None = None
    """Optional package-resolved visual whose local X=0 is the socket mouth."""
    rigid_props: RigidBodyPropertiesCfg = RigidBodyPropertiesCfg(kinematic_enabled=True)
    physics_material: RigidBodyMaterialCfg = RigidBodyMaterialCfg(
        static_friction=2.0, dynamic_friction=2.0, friction_combine_mode="max",
    )
