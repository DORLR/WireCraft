# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Legacy female-port configuration backed by the shared socket factory."""

from dataclasses import fields
from isaaclab.utils import configclass
from .connector_spawner import SocketCfg, spawn_connector_socket


def spawn_procedural_female_port(prim_path, cfg, translation=None, orientation=None):
    family = cfg.family or ("displayport_style" if cfg.port_type == "DP" else cfg.port_type)
    values = {f.name: getattr(cfg, f.name) for f in fields(SocketCfg) if f.name != "func"}
    values["family"] = family
    if cfg.hole_clearance is not None:
        if cfg.clearance is not None and cfg.clearance != cfg.hole_clearance:
            raise ValueError("clearance and legacy hole_clearance must agree when both are set.")
        values["clearance"] = cfg.hole_clearance
    shared = SocketCfg(**values)
    return spawn_connector_socket(prim_path, shared, translation=translation, orientation=orientation)


@configclass
class ProceduralFemalePortCfg(SocketCfg):
    """Compatibility field names; default dimensions come from the same profile."""
    func = spawn_procedural_female_port
    family: str | None = None
    port_type: str = "stacked_cuboid"
    hole_clearance: float | None = None
    """Legacy name for per-side clearance; None uses the family default."""
