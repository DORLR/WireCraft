# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Step through the shared port/socket factory and its legacy adapters.

Run with the Isaac Lab Python environment, from the repository root:
    ./isaaclab.sh -p scripts/demos/debug_ports.py --family displayport_style

The four rows use new/legacy APIs, first without a USD visual, then with it.
Use --family displayport for the real socket enclosure, ethernet for the latch,
or all for all eight families. --pdb pauses before the first profile lookup.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--family", default="displayport_style", help="One connector family or all.")
parser.add_argument("--clearance-mm", type=float, default=None, help="Optional per-side override, in mm.")
parser.add_argument("--steps", type=int, default=4, help="Physics steps after creating the scene.")
parser.add_argument("--keep-open", action="store_true", help="Keep the GUI scene open after stepping.")
parser.add_argument("--pdb", action="store_true", help="Open the standard Python debugger before spawning.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 0:
    parser.error("--steps must be nonnegative.")
if args.keep_open and args.headless:
    parser.error("--keep-open requires the GUI; omit --headless.")
launcher = AppLauncher(args)
simulation_app = launcher.app

# Isaac Sim must be running before importing its USD and spawner modules.
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.ports import (
    PortCfg,
    ProceduralFemalePortCfg,
    ProceduralMalePortCfg,
    SocketCfg,
    connector_families,
    get_connector_profile,
)
from isaaclab_assets import WIRECRAFT_ASSET_DIR


def review_family(family, family_index, clearance):
    """Set the first breakpoint here, then inspect the four API/visual cases."""
    profile = get_connector_profile(family)  # Breakpoint 1: F11 opens the profile lookup.
    legacy_name = "DP" if family == "displayport_style" else family
    assets = f"{WIRECRAFT_ASSET_DIR}/connectors/{profile.category}/{profile.family}"
    cases = (
        ("new_geometry", False, False),
        ("legacy_geometry", True, False),
        ("new_visual", False, True),
        ("legacy_visual", True, True),
    )
    spawned = []
    for case_index, (case_name, legacy, with_visual) in enumerate(cases):
        port_visual = f"{assets}/port.usda" if with_visual else None
        socket_visual = f"{assets}/socket.usda" if with_visual else None
        if legacy:
            port_cfg = ProceduralMalePortCfg(port_type=legacy_name, visual_usd_path=port_visual)
            socket_cfg = ProceduralFemalePortCfg(
                port_type=legacy_name, hole_clearance=clearance, visual_usd_path=socket_visual,
            )
        else:
            port_cfg = PortCfg(family=family, visual_usd_path=port_visual)
            socket_cfg = SocketCfg(family=family, clearance=clearance, visual_usd_path=socket_visual)

        # Hold the parts still for inspection while retaining the port's defaults.
        port_cfg.rigid_props.kinematic_enabled = True
        port_cfg.rigid_props.disable_gravity = True
        position = (family_index * 0.25, case_index * 0.10, 0.05)
        root = f"/World/PortsDebug/{family}/{case_name}"
        port_path, socket_path = root + "/port", root + "/socket"
        print(f"[ports debug] {family}: {case_name}, legacy_name={legacy_name}", flush=True)

        # Breakpoint 2: inspect cfg.func and step into the male adapter/factory.
        port_prim = port_cfg.func(port_path, port_cfg, translation=position)
        # Breakpoint 3: inspect mouth_x/clearance and step into the socket code.
        socket_position = (position[0] + 0.10, position[1], position[2])
        socket_prim = socket_cfg.func(socket_path, socket_cfg, translation=socket_position)
        spawned.extend((port_prim, socket_prim))
    return spawned


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args.device))
    sim.set_camera_view((0.40, 0.55, 0.38), (0.04, 0.15, 0.04))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/Ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2000)
    light.func("/World/Light", light)
    families = connector_families() if args.family == "all" else (args.family,)
    clearance = None if args.clearance_mm is None else args.clearance_mm / 1000
    if args.pdb:
        import pdb

        pdb.set_trace()
    spawned = []
    for index, family in enumerate(families):
        spawned.extend(review_family(family, index, clearance))
    sim.reset()
    for _ in range(args.steps):
        sim.step()
    print(f"Ports debug complete: {len(families)} families, {len(spawned)} root prims, {args.steps} steps.")
    if args.keep_open:
        while simulation_app.is_running():
            sim.step()


try:
    main()
finally:
    sim = sim_utils.SimulationContext.instance()
    if sim is not None:
        sim.clear_all_callbacks()
        sim.clear_instance()
    simulation_app.close(wait_for_replicator=False)
