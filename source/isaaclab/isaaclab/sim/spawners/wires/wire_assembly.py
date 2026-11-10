# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn either wire model with independent rigid ports, without a robot/task.

The common assembly frame starts at cable X=0 and ends at X=length. Its
pose is parent-relative in meters and wxyz. Ports point outward at both ends.
"""

from __future__ import annotations

import copy
from dataclasses import MISSING
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics
from isaacsim.core.utils.stage import get_current_stage

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

from .deformable_wire_spawner import ProceduralDeformableWireCfg
from .wire_cfg import WireCfg, WireChainSpawnerCfg, WirePortCfg
from .wire_spawner import Axis, _create_wire_port


def wire_assembly_paths(prim_path: str, cfg: WireAssemblyCfg) -> dict[str, str]:
    """Actor paths for Articulation / DeformableObject / RigidObject configs."""
    result = {"root": prim_path, "wire": prim_path if isinstance(cfg.wire, WireCfg) else prim_path + "/wire"}
    if cfg.start_port is not None:
        result["start_port"] = prim_path + "/start_port"
    if cfg.end_port is not None:
        result["end_port"] = prim_path + "/end_port"
    return result


@clone
def spawn_wire_assembly(prim_path: str, cfg: WireAssemblyCfg, translation=None, orientation=None) -> Usd.Prim:
    """Create a wire and optional ports; no reward, reset or robot dependencies."""
    if not isinstance(cfg.wire, (WireCfg, ProceduralDeformableWireCfg)):
        raise TypeError("wire must be WireCfg or ProceduralDeformableWireCfg.")
    if isinstance(cfg.wire, WireCfg):
        wire = copy.deepcopy(cfg.wire)
        if wire.init_axis == "auto":
            wire.init_axis = "X"
        if wire.init_axis.upper() != "X":
            raise ValueError("The common assembly frame uses +X; rotate it with orientation instead.")
        if wire.start_port is not None or wire.end_port is not None:
            raise ValueError("Set endpoints on WireAssemblyCfg, not on both configuration levels.")
        wire.start_port, wire.end_port = cfg.start_port, cfg.end_port
        chain = WireChainSpawnerCfg(wire_cfg=wire)
        return chain.func(prim_path, chain, translation=translation, orientation=orientation)

    if cfg.attachment_offset < 0:
        raise ValueError("attachment_offset must be nonnegative.")
    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at {prim_path}.")
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    root.AddOrientOp().Set(Gf.Quatf(*(orientation or (1.0, 0.0, 0.0, 0.0))))
    wire = cfg.wire
    length = wire.num_segments * 0.01 if wire.total_length is None else wire.total_length
    mesh_path = prim_path + "/wire"
    wire.func(mesh_path, wire, translation=(length / 2, 0.0, 0.0))
    for side, port in (("start", cfg.start_port), ("end", cfg.end_port)):
        if port is None:
            continue
        _create_wire_port(prim_path, side, port, length, Axis.X)
        port_path = prim_path + f"/{side}_port"
        attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, prim_path + f"/attachment_{side}")
        attachment.GetActor0Rel().SetTargets([Sdf.Path(port_path)])
        attachment.GetActor1Rel().SetTargets([Sdf.Path(mesh_path)])
        auto = PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim())
        auto.CreateEnableDeformableVertexAttachmentsAttr(True)
        auto.CreateDeformableVertexOverlapOffsetAttr(cfg.attachment_offset)
        UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(port_path)).CreateFilteredPairsRel().SetTargets(
            [Sdf.Path(mesh_path)]
        )
    return root.GetPrim()


@configclass
class WireAssemblyCfg(SpawnerCfg):
    func = spawn_wire_assembly
    wire: WireCfg | ProceduralDeformableWireCfg = MISSING
    start_port: WirePortCfg | None = None
    end_port: WirePortCfg | None = None
    attachment_offset: float = 0.001
    """FEM endpoint vertex capture margin in meters; no fixed joint is used for FEM."""
