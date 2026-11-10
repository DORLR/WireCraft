# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Connector selection and geometry shared by both wire backends.

Frames, all in meters:

* Port root: the tip's rear shoulder. The tip points along local +X, so the
  leading face is ``root + tip_length * X``.
* Socket root: ``socket_mouth_x`` behind the mouth plane. Insertion is along
  local +X and the cavity ends ``socket_depth`` past the mouth.

Geometry always comes from the shared spawner profile; nothing is copied here.
"""

from __future__ import annotations

from dataclasses import dataclass

from isaaclab.sim.spawners.ports import SocketCfg, connector_families, get_connector_profile
from isaaclab.sim.spawners.ports.connector_spawner import _offset_polygon
from isaaclab.sim.spawners.wires import WirePortCfg
from isaaclab.utils import configclass
from isaaclab_assets import WIRECRAFT_ASSET_DIR

_ALIASES = {"DP": "displayport_style", "dp": "displayport_style"}


def normalize_family(name: str) -> str:
    """Return the canonical profile name, rejecting unknown connector families."""
    family = _ALIASES.get(name, name)
    if family not in connector_families():
        raise ValueError(f"Unknown connector {name!r}; choose from {connector_families()}.")
    return family


def _asset(family: str, name: str) -> str:
    return f"{WIRECRAFT_ASSET_DIR}/connectors/{get_connector_profile(family).category}/{family}/{name}.usda"


@configclass
class ConnectorCfg:
    """User-facing connector inputs. Spawn configs are derived during resolve."""

    family: str = "stacked_cuboid"
    socket_clearance: float | None = None
    """Per-side clearance in meters; None keeps the family default."""
    port_mass: float = 0.01
    """Mass of each port in kg."""
    port_static_friction: float = 2.0
    port_dynamic_friction: float = 2.0
    socket_color: tuple[float, float, float] = (0.8, 0.0, 0.0)
    socket_inner_color: tuple[float, float, float] = (0.8, 0.0, 0.0)

    def port_cfg(self) -> WirePortCfg:
        """Endpoint config shared by the start and end port of both backends."""
        family = normalize_family(self.family)
        return WirePortCfg(
            family=family,
            visual_usd_path=_asset(family, "port"),
            mass=self.port_mass,
            static_friction=self.port_static_friction,
            dynamic_friction=self.port_dynamic_friction,
        )

    def socket_cfg(self, **kwargs) -> SocketCfg:
        """Paired socket built by the shared spawner."""
        family = normalize_family(self.family)
        return SocketCfg(
            family=family,
            visual_usd_path=_asset(family, "socket"),
            clearance=self.socket_clearance,
            color=tuple(self.socket_color),
            inner_color=tuple(self.socket_inner_color),
            **kwargs,
        )


@dataclass(frozen=True)
class ConnectorGeometry:
    """Scalar geometry read back from the profile for task logic and policies."""

    family: str
    category: str
    tip_length: float
    grasp_x: float
    socket_depth: float
    socket_mouth_x: float
    tip_outline: tuple[tuple[float, float], ...]
    """Leading-face cross-section (y, z), counterclockwise."""
    hole: tuple[tuple[float, float], ...]
    """Socket bore cross-section (y, z): the outline grown by the clearance, as the spawner cuts it."""

    @classmethod
    def from_family(cls, name: str, clearance: float | None = None) -> ConnectorGeometry:
        family = normalize_family(name)
        profile = get_connector_profile(family)
        clearance = profile.clearance if clearance is None else clearance
        return cls(
            family=family,
            category=profile.category,
            tip_length=profile.tip_length,
            grasp_x=profile.grasp_x,
            socket_depth=profile.socket_depth,
            socket_mouth_x=SocketCfg().mouth_x,
            tip_outline=tuple(profile.tip_outline),
            hole=tuple(_offset_polygon(list(profile.tip_outline), clearance)),
        )
