# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for spawning procedural wire chain articulations.

This module provides a parametric wire generator that builds rigid-body articulation
chains (capsules + D6 joints) at runtime from configuration parameters. No pre-authored
USD files are needed.

Usage example::

    from isaaclab.sim.spawners.wires import WireCfg, build_wire_articulation_cfg, WirePresets

    # Option A: Full parametric control
    wire_art_cfg = build_wire_articulation_cfg(
        WireCfg(num_segments=25, total_length=0.35, radius=0.002, stiffness=2.0, damping=0.2),
        prim_path="{ENV_REGEX_NS}/wire",
    )

    # Option B: Use a preset
    wire_art_cfg = build_wire_articulation_cfg(
        WirePresets.THIN_FLEXIBLE,
        prim_path="{ENV_REGEX_NS}/wire",
    )
"""

from .wire_builder import build_wire_articulation_cfg, build_wire_articulation_cfg_full, WireBuildResult
from .wire_cfg import WireChainSpawnerCfg, WireCfg, WirePortCfg
from .wire_presets import WirePresets
from .wire_spawner import spawn_wire_chain
from .wire_urdf import export_wire_urdf, save_wire_urdf
from .deformable_wire_spawner import ProceduralDeformableWireCfg, spawn_procedural_deformable_wire

from .wire_assembly import WireAssemblyCfg, spawn_wire_assembly, wire_assembly_paths
