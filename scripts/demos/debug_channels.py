# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Debug the shared straight/curved channel spawner without a robot or wire.

Run from the repository root with the existing Isaac Sim environment::

    ./isaaclab.sh -p scripts/demos/debug_channels.py --headless --steps 240
    ./isaaclab.sh -p scripts/demos/debug_channels.py --steps 0
    ./isaaclab.sh -p scripts/demos/debug_channels.py --channel-type curved --length 0.15 --bend-angle-deg 30

Zero steps keeps a GUI session open; headless mode always uses a finite run.
The joined example preserves three independent modules. A nominal 10-degree
arc sampled by 40 chords has flat end-face headings 9.75 degrees apart;
placement uses those actual faces instead of a nominal 10-degree rotation.
"""

import argparse
import math
import struct

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--channel-type", choices=("both", "straight", "curved"), default="both")
parser.add_argument("--length", type=float, default=None, help="Meters; None uses the module's 0.110/0.066 m default.")
parser.add_argument("--slot-width", type=float, default=0.010, help="Inner slot width in meters.")
parser.add_argument("--slot-depth", type=float, default=0.010, help="Slot depth above the floor in meters.")
parser.add_argument("--wall-thickness", type=float, default=0.005)
parser.add_argument("--bend-angle-deg", type=float, default=10.0)
parser.add_argument("--curve-segments", type=int, default=40)
parser.add_argument("--chain-modules", type=int, default=3, help="Also join this many independent curved modules at their actual flat end faces; 1 disables it.")
parser.add_argument("--steps", type=int, default=240)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.channels import ChannelCfg
from pxr import Gf, UsdGeom


def _sections(stage, path, cfg, end):
    """Read the actual authored float32 end-face vertices in world space."""
    segment = (cfg.curve_segments - 1 if cfg.channel_type == "curved" else 0) if end else 0
    indices = (3, 2, 7, 6) if end else (0, 1, 4, 5)
    cache = UsdGeom.XformCache()
    result = {}
    for side in ("floor", "left", "right"):
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"{path}/colliders/{side}_{segment}"))
        points = mesh.GetPointsAttr().Get()
        transform = cache.GetLocalToWorldTransform(mesh.GetPrim())
        result[side] = [transform.Transform(Gf.Vec3d(points[i])) for i in indices]
    return result


def _frame(section):
    center = (section[0] + section[1]) / 2
    span = section[0] - section[1]
    return center, math.atan2(-span[0], span[1])


def _set_gap(a, b):
    return max(max(min((p - q).GetLength() for q in b) for p in a),
               max(min((p - q).GetLength() for q in a) for p in b))


def _float32_points(points):
    return [Gf.Vec3d(*struct.unpack("<fff", struct.pack("<fff", *p))) for p in points]


def _spawn_joined_modules(stage, cfg, count, y):
    """Join unchanged module solids using their flat caps, not nominal arc tangents."""
    root = "/World/Channels/curved_modules"
    UsdGeom.Xform.Define(stage, root)
    previous = None
    worst, worst_f32 = 0.0, 0.0
    start_heading = None
    root_yaws = []
    for i in range(count):
        path = f"{root}/module_{i}"
        if previous is None:
            cfg.func(path, cfg, translation=(0.0, y, 0.0))
            start_heading = _frame(_sections(stage, path, cfg, False)["floor"])[1]
            yaw = 0.0
        else:
            end_faces = _sections(stage, previous, cfg, True)
            target, end_heading = _frame(end_faces["floor"])
            yaw = end_heading - start_heading
            cfg.func(path, cfg, orientation=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))
            start = _frame(_sections(stage, path, cfg, False)["floor"])[0]
            stage.GetPrimAtPath(path).GetAttribute("xformOp:translate").Set(target - start)
            start_faces = _sections(stage, path, cfg, False)
            for side in ("floor", "left", "right"):
                worst = max(worst, _set_gap(end_faces[side], start_faces[side]))
                worst_f32 = max(worst_f32, _set_gap(_float32_points(end_faces[side]),
                                                  _float32_points(start_faces[side])))
        previous = path
        root_yaws.append(math.degrees(yaw))
    if worst > 1e-7 or worst_f32 > 1e-7:
        raise AssertionError(f"Independent channel module seam exceeds float32 tolerance: {worst}, {worst_f32} m")
    print(f"CHANNEL_MODULE_SEAMS modules={count}; root yaw degrees={root_yaws}; "
          f"max end-face gap={worst:.9g} m; world-float32 gap={worst_f32:.9g} m.", flush=True)


def main():
    if args.steps < 0:
        raise ValueError("steps must be >= 0.")
    if args.chain_modules < 1:
        raise ValueError("chain-modules must be >= 1.")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args.device))
    sim.set_camera_view((0.30, -0.34, 0.30), (0.0, 0.08, 0.0))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/Ground", ground, translation=(0.0, 0.0, -0.001))
    light = sim_utils.DomeLightCfg(intensity=2000)
    light.func("/World/Light", light)
    types = ("straight", "curved") if args.channel_type == "both" else (args.channel_type,)
    examples = []
    for channel_type in types:
        cfg = ChannelCfg(
            channel_type=channel_type, length=args.length, width=args.slot_width, depth=args.slot_depth,
            wall_thickness=args.wall_thickness, bend_angle_deg=args.bend_angle_deg,
            curve_segments=args.curve_segments,
        )
        examples.append((channel_type, cfg))
    for row, (name, cfg) in enumerate(examples):
        # Break here, then inside spawn_channel/_channel_parts to inspect geometry.
        cfg.func(f"/World/Channels/{name}", cfg, translation=(0.0, row * 0.08, 0.0))
        print(f"Spawned {name} from ChannelCfg.", flush=True)
    joined = "curved" in types and args.chain_modules > 1
    if joined:
        curved_cfg = next(cfg for name, cfg in examples if name == "curved")
        _spawn_joined_modules(sim.stage, curved_cfg, args.chain_modules, len(examples) * 0.08)
    sim.reset()
    limit = args.steps or (240 if args.headless else None)
    steps = 0
    while simulation_app.is_running() and (limit is None or steps < limit):
        sim.step()
        steps += 1
    channel_count = len(examples) + (args.chain_modules if joined else 0)
    print(f"CHANNEL_DEBUG_COMPLETE channels={channel_count} scenes={len(examples) + int(joined)} steps={steps}", flush=True)


try:
    main()
finally:
    sim = sim_utils.SimulationContext.instance()
    if sim is not None:
        sim.clear_all_callbacks()
        sim.clear_instance()
    simulation_app.close(wait_for_replicator=False)
