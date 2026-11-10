# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convenience builder that converts a WireCfg into a ready-to-use ArticulationCfg.

Note:
    This module uses deferred imports for ``isaaclab.assets`` and ``isaaclab.actuators``
    to avoid circular import issues (spawners are loaded during ``isaaclab.sim`` init,
    which happens before ``isaaclab.assets`` is fully initialized).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .wire_cfg import WireChainSpawnerCfg, WireCfg

if TYPE_CHECKING:
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg


@dataclass
class WireBuildResult:
    """Result of building a wire articulation configuration."""

    articulation_cfg: object  # ArticulationCfg at runtime
    """Ready-to-use ArticulationCfg for scene registration."""

    head_segment_name: str
    """Name of the first segment (e.g. 'seg_0')."""

    tail_segment_name: str
    """Name of the last segment (e.g. 'seg_19')."""


def build_wire_articulation_cfg(
    wire_cfg: WireCfg,
    prim_path: str,
):
    """Convert a :class:`WireCfg` into a ready-to-use :class:`ArticulationCfg`.

    This is the main convenience entry point. It creates a ``WireChainSpawnerCfg``
    from the given wire parameters, wraps it in an ``ArticulationCfg`` with passive
    spring-damper actuators, and returns a config that can be directly assigned to a
    scene attribute.

    Args:
        wire_cfg: High-level wire description.
        prim_path: Prim path expression (e.g. ``"{ENV_REGEX_NS}/wire"``).

    Returns:
        An ``ArticulationCfg`` ready for scene registration.
    """
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    spawner = WireChainSpawnerCfg(wire_cfg=wire_cfg)

    art_cfg = ArticulationCfg(
        prim_path=prim_path,
        spawn=spawner,
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "wire_passive": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=wire_cfg.stiffness,
                damping=wire_cfg.damping,
                velocity_limit_sim=wire_cfg.joint_velocity_limit,
                armature=wire_cfg.joint_armature,
            ),
        },
    )

    return art_cfg


def build_wire_articulation_cfg_full(
    wire_cfg: WireCfg,
    prim_path: str,
) -> WireBuildResult:
    """Same as :func:`build_wire_articulation_cfg` but also returns helper metadata.

    Args:
        wire_cfg: High-level wire description.
        prim_path: Prim path expression.

    Returns:
        A :class:`WireBuildResult` with the articulation config and segment names.
    """
    art_cfg = build_wire_articulation_cfg(wire_cfg, prim_path)

    return WireBuildResult(
        articulation_cfg=art_cfg,
        head_segment_name="seg_0",
        tail_segment_name=f"seg_{wire_cfg.num_segments - 1}",
    )
