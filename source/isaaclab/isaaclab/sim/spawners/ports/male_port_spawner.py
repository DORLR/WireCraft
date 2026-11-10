# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Legacy male-port configuration backed by the shared connector factory.

Geometry and coordinates follow connector_profiles, including the confirmed
printable housing/tip dimensions. ``DP`` maps to ``displayport_style``.
"""

from dataclasses import fields
import copy
from isaaclab.utils import configclass
from .connector_spawner import PortCfg, spawn_connector_port


def spawn_procedural_male_port(prim_path, cfg, translation=None, orientation=None):
    family = cfg.family or ("displayport_style" if cfg.port_type == "DP" else cfg.port_type)
    values = {f.name: getattr(cfg, f.name) for f in fields(PortCfg) if f.name != "func"}
    values["family"] = family
    # Old callers often supplied an empty rigid-body cfg. Preserve explicit
    # overrides while retaining the common port's finite speed limits.
    if cfg.rigid_props is not None:
        body = copy.deepcopy(cfg.rigid_props)
        defaults = PortCfg().rigid_props
        for name in ("max_depenetration_velocity", "max_linear_velocity", "max_angular_velocity"):
            if getattr(body, name) is None:
                setattr(body, name, getattr(defaults, name))
        values["rigid_props"] = body
    colors = dict(cfg.part_colors or {})
    legacy_colors = {
        "cylinder": {"housing": cfg.cylinder_color, "tip": cfg.cylinder_color},
        "cuboid": {"housing": cfg.cuboid_color, "tip": cfg.cuboid_color},
        "stacked_cuboid": {"housing": cfg.stacked_cuboid_base_color,
                           "tip": cfg.stacked_cuboid_tip_color, "latch": cfg.stacked_cuboid_latch_color},
        "displayport_style": {"housing": cfg.dp_housing_color, "tip": cfg.dp_tip_color},
        "usb_a": {"housing": cfg.usb_a_housing_color},
    }
    colors.update({k: v for k, v in legacy_colors.get(family, {}).items() if v is not None})
    values["part_colors"] = colors or None
    shared = PortCfg(**values)
    return spawn_connector_port(prim_path, shared, translation=translation, orientation=orientation)


@configclass
class ProceduralMalePortCfg(PortCfg):
    """Compatibility fields only; all mesh creation is in connector_spawner."""
    func = spawn_procedural_male_port
    family: str | None = None
    port_type: str = "stacked_cuboid"
    cylinder_color: tuple[float, float, float] | None = None
    cuboid_color: tuple[float, float, float] | None = None
    stacked_cuboid_base_color: tuple[float, float, float] | None = None
    stacked_cuboid_tip_color: tuple[float, float, float] | None = None
    stacked_cuboid_latch_color: tuple[float, float, float] | None = None
    dp_housing_color: tuple[float, float, float] | None = None
    dp_tip_color: tuple[float, float, float] | None = None
    usb_a_housing_color: tuple[float, float, float] | None = None
