# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Step through the existing clip spawner, with and without retaining lips.

Run from the repository root in the Isaac Lab environment::
    ./isaaclab.sh -p scripts/demos/debug_clips.py --headless --device cpu

Set a breakpoint on cfg.func in review_clip, then step into the existing
spawn_procedural_clip, add_block and apply_collision functions.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--keep-open", action="store_true", help="Keep the GUI open after the finite steps.")
parser.add_argument("--pdb", action="store_true", help="Pause before creating the clips.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 0:
    parser.error("--steps must be nonnegative.")
if args.keep_open and args.headless:
    parser.error("--keep-open requires the GUI; omit --headless.")
launcher = AppLauncher(args)
simulation_app = launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.ports import ProceduralClipCfg
from pxr import Usd, UsdPhysics


def review_clip(name, cfg, row):
    """Break at cfg.func, inspect the dimensions, then enter the real factory."""
    path = f"/World/ClipsDebug/{name}"
    # First breakpoint: F11 enters the clone wrapper and then the clip factory.
    prim = cfg.func(path, cfg, translation=(0.0, row * 0.06, 0.0), orientation=(1.0, 0.0, 0.0, 0.0))
    colliders = [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)]
    top_gap = cfg.opening_width - 2 * cfg.lip_overhang
    print(f"[clips debug] {name}: inner width={cfg.opening_width * 1000:g} mm, "
          f"top gap={top_gap * 1000:g} mm, colliders={len(colliders)}", flush=True)
    return len(colliders)


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args.device))
    sim.set_camera_view((0.12, -0.16, 0.13), (0.0, 0.03, 0.008))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/Ground", ground, translation=(0.0, 0.0, -0.001))
    light = sim_utils.DomeLightCfg(intensity=2000)
    light.func("/World/Light", light)
    cases = (("with_lips", ProceduralClipCfg()),
             ("open_u", ProceduralClipCfg(lip_overhang=0.0)))
    if args.pdb:
        import pdb

        pdb.set_trace()
    collider_count = sum(review_clip(name, cfg, row) for row, (name, cfg) in enumerate(cases))
    sim.reset()
    for _ in range(args.steps):
        sim.step()
    print(f"Clips debug complete: {len(cases)} clips, {collider_count} colliders, {args.steps} steps.", flush=True)
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
