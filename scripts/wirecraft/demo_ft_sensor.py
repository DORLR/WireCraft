# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Demo: a UR5 with a wrist force/torque sensor, read during a scripted insertion.

    python scripts/wirecraft/demo_ft_sensor.py --headless --output out/ft_demo
    python scripts/wirecraft/demo_ft_sensor.py --headless --connector HDMI --wire Deformable --output out/ft_hdmi

The sensor is the joint reaction wrench the flange passes to the gripper (``robot.ft_sensor_body:
gripper_base_link``, see ``scripts/wirecraft/experiments/ur5_ft_sensor.yaml``), in the wrist_3_link frame.
The script writes ``wrench.csv`` (step, phase, raw fx fy fz tx ty tz, and the same tared) and
``wrench.png`` (tared force and torque norms with the scripted phases shaded).

Two tares: the open gripper at rest (end of SETTLE) and the closed gripper holding the port in the
air (end of LIFT). While the jaws are closed the reading carries a steady offset of about 8-9 N from
the grasp itself (the same at ``--body wrist_3_link``), so contact during the insertion only shows
against the held tare; ``*_tared`` uses the held tare from CLOSE to HOLD and the open tare otherwise.

This is a starting point, not a finished sensor model: no noise, bias drift, filtering or saturation.
"""

import argparse
import csv
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--connector", default="StackedCuboid", help="Connector in the Gym ID, e.g. HDMI, Ethernet.")
parser.add_argument("--wire", default="Articulation", choices=("Articulation", "Deformable"))
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--body", default=None, help="Override robot.ft_sensor_body, e.g. wrist_3_link.")
parser.add_argument("--output", default="out/ft_demo")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.direct.wirecraft.single_arm.common import compose  # noqa: E402
from isaaclab_tasks.direct.wirecraft.single_arm.common.registry import SPECS  # noqa: E402
from isaaclab_tasks.direct.wirecraft.single_arm.common.scripted import make_policy  # noqa: E402

EXPERIMENT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "ur5_ft_sensor.yaml")
AXES = ("fx", "fy", "fz", "tx", "ty", "tz")


HELD = ("CLOSE", "LIFT", "PREALIGN", "ALIGN", "INSERT", "HOLD")


def mean(readings):
    readings = readings[-20:]
    return [sum(w[i] for w in readings) / max(1, len(readings)) for i in range(6)]


def main():
    task = f"WireCraft-UR5-{args_cli.wire}-{args_cli.connector}-Insertion-v0"
    cfg = compose.compose_env_cfg(SPECS[task], "scripted", [EXPERIMENT])
    cfg.scene.num_envs, cfg.seed = 1, args_cli.seed
    if args_cli.body:
        cfg.robot.ft_sensor_body = args_cli.body
    env = gym.make(task, cfg=cfg).unwrapped
    env.reset()
    policy = make_policy(env)
    rows = []
    with torch.inference_mode():
        for step in range(policy.max_steps):
            env.step(policy.act())
            policy.observe(env.extras)
            rows.append((step, policy.PHASES[int(policy.phase[0])], env.arm.wrench[0].tolist()))
            if policy.finished:
                break
    open_tare = mean([w for _, phase, w in rows if phase == "SETTLE"])
    held_tare = mean([w for _, phase, w in rows if phase == "LIFT"]) if any(r[1] == "LIFT" for r in rows) else open_tare
    tared = [[v - t for v, t in zip(w, held_tare if phase in HELD else open_tare)] for _, phase, w in rows]
    os.makedirs(args_cli.output, exist_ok=True)
    with open(os.path.join(args_cli.output, "wrench.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "phase", *AXES, *(f"{a}_tared" for a in AXES)])
        for (step, phase, wrench), t in zip(rows, tared):
            writer.writerow([step, phase, *(f"{v:.5f}" for v in wrench), *(f"{v:.5f}" for v in t)])
    success = bool(policy.success[0])
    plot(rows, tared, os.path.join(args_cli.output, "wrench.png"), task, success)
    print(f"[ft_demo] {task}: {len(rows)} steps, success={success}")
    print(f"[ft_demo] open tare {[round(v, 3) for v in open_tare]}, held tare {[round(v, 3) for v in held_tare]}")
    print(f"[ft_demo] wrote {args_cli.output}/wrench.csv and wrench.png")
    env.close()


def plot(rows, tared, path, task, success):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r[0] for r in rows]
    force = [sum(v * v for v in w[:3]) ** 0.5 for w in tared]
    torque = [sum(v * v for v in w[3:]) ** 0.5 for w in tared]
    figure, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(steps, force, lw=1)
    axes[0].set_ylabel("|F| tared (N)")
    axes[1].plot(steps, torque, lw=1, color="tab:orange")
    axes[1].set_ylabel("|T| tared (N m)")
    axes[1].set_xlabel("control step (60 Hz)")
    # Shade every other phase and label where each one starts.
    start, segment = 0, 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i][1] != rows[start][1]:
            if segment % 2:
                for axis in axes:
                    axis.axvspan(steps[start], steps[i - 1], color="grey", alpha=0.12, lw=0)
            axes[0].text(steps[start], max(force) * 0.98, rows[start][1], fontsize=6, rotation=90, va="top")
            start, segment = i, segment + 1
    figure.suptitle(f"{task}  wrist F/T, success={success}", fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=130)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        # SimulationApp.close() ends the process with status 0, so a failure exits before it.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
