from typing import Literal

from pxr import Usd, UsdGeom, Vt, Gf, UsdPhysics
from isaacsim.core.utils.stage import get_current_stage
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim import schemas
from isaaclab.utils import configclass
from isaaclab.sim.utils import clone


@clone
def spawn_procedural_clip(
    prim_path: str,
    cfg: "ProceduralClipCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    stage = get_current_stage()

    xform = UsdGeom.Xform.Define(stage, prim_path)

    if translation is not None:
        xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    if orientation is not None:
        xform.AddOrientOp().Set(Gf.Quatf(orientation[0], orientation[1], orientation[2], orientation[3]))

    def apply_collision(prim: Usd.Prim):
        UsdPhysics.CollisionAPI.Apply(prim)
        if cfg.collision_props is not None:
            schemas.define_collision_properties(prim.GetPath().pathString, cfg.collision_props)

    color = Vt.Vec3fArray([Gf.Vec3f(*cfg.color)])

    block_idx = 0

    def add_block(name, pos, scale, block_color=color):
        nonlocal block_idx
        b = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}_{block_idx}")
        block_idx += 1
        b.CreateSizeAttr(1.0)
        b.AddTranslateOp().Set(Gf.Vec3d(*pos))
        b.AddScaleOp().Set(Gf.Vec3d(*scale))
        b.GetDisplayColorAttr().Set(block_color)
        apply_collision(b.GetPrim())

    ow = cfg.opening_width
    ww = cfg.wall_width
    wh = cfg.wall_height
    cd = cfg.clip_depth
    bh = cfg.base_height
    lo = cfg.lip_overhang
    lh = cfg.lip_height

    wall_y_offset = (ow / 2.0) + (ww / 2.0)

    add_block("Base", (0.0, 0.0, bh / 2.0), (cd, ow + 2.0 * ww, bh))
    add_block("WallL", (0.0, -wall_y_offset, bh + wh / 2.0), (cd, ww, wh))
    add_block("WallR", (0.0, wall_y_offset, bh + wh / 2.0), (cd, ww, wh))

    # Upper lips: horizontal overhangs extending inward from the top of each wall.
    # Lip top is flush with wall top; lip extends down by lip_height into the wall's Z range.
    # Top gap between lip inner edges = opening_width - 2 * lip_overhang.
    if lo > 0.0 and lh > 0.0:
        lip_y_center = (ow / 2.0) - (lo / 2.0)
        lip_z_center = bh + wh - (lh / 2.0)
        add_block("LipL", (0.0, -lip_y_center, lip_z_center), (cd, lo, lh))
        add_block("LipR", (0.0, lip_y_center, lip_z_center), (cd, lo, lh))

    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props)
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props)

    return xform.GetPrim()


@configclass
class ProceduralClipCfg(RigidObjectSpawnerCfg):
    func = spawn_procedural_clip

    opening_width: float = 0.015
    """Y-gap between inner faces of the two side walls (meters). Sized to fit wire (max ~8mm diameter) but block the male port (~18mm)."""

    wall_width: float = 0.004
    """Y-thickness of each side wall (meters)."""

    wall_height: float = 0.0125
    """Z-height of the side walls above the base (meters)."""

    clip_depth: float = 0.03
    """X-extent of the clip along the wire routing direction (meters)."""

    base_height: float = 0.004
    """Z-thickness of the connecting base piece (meters). Clip local origin is at the bottom-center of the base."""

    lip_overhang: float = 0.003
    """How far each upper lip extends inward from its wall (meters). Top gap = opening_width - 2*lip_overhang. Set to 0 for a plain U-slot."""

    lip_height: float = 0.0005
    """Z-thickness of each upper lip (meters)."""

    color: tuple[float, float, float] = (0.1, 0.1, 0.8)
    """RGB color for the clip geometry. Defaults to blue."""
