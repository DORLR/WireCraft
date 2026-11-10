# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an RSL-RL checkpoint on a WireCraft task: one deterministic episode per env.

    python scripts/wirecraft/eval_checkpoint.py --headless \
        --task WireCraft-UR5-Articulation-StackedCuboid-Insertion-v0 \
        --checkpoint model_8900.pt --num_envs 100 --seed 42 --output eval.json [--video env0.mp4]

The env is composed in ``--mode`` (train by default, so resets match training) and the
policy's mean action is used. Each env is scored over its first episode only. The JSON reports
the success rate (the env's own success test: for insertion, the leading face 10 mm into the bore
at some step), how episodes ended (timeout, termination, simulation fault) and per-second means
of the logged metrics; for insertion also how often the port was seated at the timeout, the share of
the episode it was seated (when the +50 bonus is paid), and the median grasp point in the port frame.
``env.*`` and ``agent.*`` overrides and ``--experiment`` files apply as in training.
``--video`` records env 0's first episode from the ``side2`` workstation camera, up to one second
after it succeeds.
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

ORIGINAL_ARGV = list(sys.argv[1:])

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
# Script defaults sit below --experiment files; only flags actually typed override an experiment.
parser.add_argument("--num_envs", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--mode", default="train", choices=("train", "random", "scripted"))
parser.add_argument("--experiment", action="append", default=[], help="Experiment YAML; repeat in priority order.")
parser.add_argument("--output", required=True, help="JSON result path.")
parser.add_argument("--video", default=None, help="MP4 path for env 0's first episode.")
AppLauncher.add_app_launcher_args(parser)
args_cli, rest = parser.parse_known_args()
SCRIPT_DEFAULTS = {"seed": parser.get_default("seed"), "scene": {"num_envs": parser.get_default("num_envs")}}
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + rest
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.utils.math import quat_apply, quat_inv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.direct.wirecraft.single_arm.common import compose  # noqa: E402
from isaaclab_tasks.direct.wirecraft.single_arm.common.registry import SPECS  # noqa: E402


def main():
    spec = SPECS[args_cli.task]
    env_cfg = compose.compose_env_cfg(spec, args_cli.mode, args_cli.experiment,
                                      script_defaults=SCRIPT_DEFAULTS)
    if args_cli.video:
        env_cfg.enable_cameras, env_cfg.cameras, env_cfg.camera_resolution = True, ["side2"], 512
    agent_cfg = compose.compose_agent_cfg(spec, args_cli.experiment)
    # Typed flags (--num_envs, --seed, --device, ...) and env.*/agent.* tokens, exactly as in training.
    compose.apply_cli(env_cfg, agent_cfg, compose.parse_cli_overrides(rest),
                      compose.legacy_overrides(args_cli, ORIGINAL_ARGV))
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=base.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=base.device)
    # The observation normalizer must be the checkpoint's own; a fresh one would feed the actor raw values.
    saved = torch.load(args_cli.checkpoint, map_location=base.device, weights_only=False).get("obs_norm_state_dict")
    loaded = runner.obs_normalizer.state_dict() if hasattr(runner.obs_normalizer, "state_dict") else {}
    normalizer = {
        "in_checkpoint": saved is not None,
        "matches_checkpoint": saved is not None and set(saved) <= set(loaded)
        and all(torch.equal(loaded[k].to(saved[k].device), saved[k]) for k in saved),
        "frozen": not runner.obs_normalizer.training if hasattr(runner.obs_normalizer, "training") else True,
    }
    insertion = spec.task == "insertion"

    writer = None
    if args_cli.video:
        import imageio.v2 as imageio

        writer = imageio.get_writer(args_cli.video, fps=30, codec="libx264", macro_block_size=16,
                                    ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])

    count, device = base.num_envs, base.device
    alive = torch.ones(count, dtype=torch.bool, device=device)
    seated_steps = torch.zeros(count, device=device)
    grasped_ever = torch.zeros(count, dtype=torch.bool, device=device)
    success_ever = torch.zeros(count, dtype=torch.bool, device=device)
    final_seated = torch.zeros(count, dtype=torch.bool, device=device)
    length = torch.zeros(count, device=device)
    ended_by = ["running"] * count
    logs, grasp_points = {}, []
    video_tail = int(1.0 / base.step_dt)
    obs, _ = wrapped.get_observations()
    with torch.inference_mode():
        for step in range(int(base.max_episode_length) + 5):
            obs, _, dones, extras = wrapped.step(policy(obs))
            success_ever |= base.extras["is_success"].bool() & alive
            if insertion:
                task = base.task
                _, _, _, grasped = task._grasp_terms()
                seated_steps += (task.seated & alive).float()
                grasped_ever |= grasped & alive
                chosen = grasped & alive
                if chosen.any():
                    state = base.wire_state
                    local = quat_apply(quat_inv(state.insertion_end_quat), base.arm.ee_pos - state.insertion_end_pos)
                    grasp_points.append(local[chosen].cpu())
            for key, value in base.extras.get("log", {}).items():
                # Per-step means of the scalar metrics; success is counted per episode above.
                if key.startswith("metrics/") and torch.as_tensor(value).numel() == 1:
                    logs.setdefault(key, []).append(float(value))
            length += alive.float()
            if bool(success_ever[0]):
                video_tail -= 1
            if writer is not None and bool(alive[0]) and video_tail >= 0 and step % 2 == 0:
                frame = base.tiled_cameras["side2"].data.output["rgb"][0, ..., :3]
                writer.append_data(frame.to(torch.uint8).cpu().numpy())
            done = dones.bool() & alive
            timeout = extras.get("time_outs", torch.zeros_like(done)).bool()
            for i in done.nonzero().flatten().tolist():
                ended_by[i] = "timeout" if timeout[i] else ("fault" if base.faulted[i] else "terminated")
            if insertion:
                final_seated |= done & timeout & task.seated
            alive &= ~done
            if not alive.any():
                break
    if writer is not None:
        writer.close()
    result = {
        "task": args_cli.task, "checkpoint": args_cli.checkpoint, "mode": args_cli.mode,
        "num_envs": count, "seed": base.cfg.seed, "episode_steps": int(base.max_episode_length),
        "success_rate": float(success_ever.float().mean()),
        "ended_by": {key: ended_by.count(key) for key in sorted(set(ended_by))},
        "mean_episode_steps": float(length.mean()),
        "observation_normalizer": normalizer,
        "metrics_by_second": {k: [round(sum(v[i:i + 60]) / len(v[i:i + 60]), 4) for i in range(0, len(v), 60)]
                              for k, v in logs.items()},
    }
    if insertion:
        result.update(
            seated_at_timeout=float(final_seated.float().mean()),
            seated_fraction_of_episode=float((seated_steps / length.clamp_min(1)).mean()),
            grasped_ever=float(grasped_ever.float().mean()),
            grasp_point_in_port_mm_median=(torch.cat(grasp_points).median(0).values * 1000).tolist() if grasp_points else None,
        )
    with open(args_cli.output, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "metrics_by_second"}, indent=1))
    env.close()


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
