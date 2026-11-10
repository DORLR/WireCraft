# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Run any WireCraft combination in random or scripted mode and record the outcome.

    python scripts/wirecraft/run.py --headless --robot franka --job clip_routing \
        --wire articulation --connector stacked_cuboid --variant 2clip \
        --mode scripted --seed 7 --num_envs 5 --output results/franka_2clip

``--task <Gym ID>`` selects a combination directly. Extra ``env.*`` and
``agent.*`` arguments and ``--experiment file.yaml`` follow the usual priority.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

ORIGINAL_ARGV = list(sys.argv[1:])

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default=None, help="Gym ID; overrides the selectors below.")
parser.add_argument("--robot", default="ur5", choices=("ur5", "franka", "trossen"))
parser.add_argument("--job", default="insertion", choices=("insertion", "clip_routing", "lay_in_channel"))
parser.add_argument("--wire", default="articulation", choices=("articulation", "deformable"))
parser.add_argument("--connector", default="stacked_cuboid")
parser.add_argument("--variant", default="", help="1clip/2clip/3clip or straight/curved.")
parser.add_argument("--mode", default="random", choices=("random", "scripted"))
parser.add_argument("--experiment", action="append", default=[], help="Experiment YAML; repeat in priority order.")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--steps", type=int, default=0, help="Control steps; zero derives them from the episode length.")
parser.add_argument("--output", default=None, help="Directory for result.json, resolved config and video.")
parser.add_argument("--video", action="store_true", help="Record env 0 from the side camera.")
parser.add_argument("--video_env", type=int, default=0, help="Env to record.")
parser.add_argument("--stop_after_successes", type=int, default=0,
                    help="Scripted: end the first rollout once this many envs succeeded and finished recording (0: never).")
parser.add_argument("--video_all", action="store_true",
                    help="Record every env of the first rollout, one rollout_env<i>.mp4 each (scripted mode).")
parser.add_argument("--list", default=None, help="Write every registered WireCraft ID and its combination to this JSON, then exit.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video_all:
    args_cli.video = True
if args_cli.video:
    args_cli.enable_cameras = True
    hydra_args += ["env.enable_cameras=true", "env.cameras=[side]", "env.camera_resolution=512", "env.render_stride=4"]
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import dataclasses  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.direct.wirecraft.single_arm.common.compose import (  # noqa: E402
    save_resolved,
    wirecraft_task_config,
)
from isaaclab_tasks.direct.wirecraft.single_arm.common.registry import SPECS  # noqa: E402
from isaaclab_tasks.direct.wirecraft.single_arm.common.scripted import make_policy  # noqa: E402


def select_task() -> str:
    if args_cli.task:
        return args_cli.task
    for task_id, spec in SPECS.items():
        wanted = (args_cli.robot, args_cli.job, args_cli.wire, args_cli.connector, args_cli.variant)
        if (spec.robot, spec.task, spec.wire, spec.connector, spec.variant) == wanted and task_id.startswith("WireCraft"):
            return task_id
    raise SystemExit(f"No registered combination for {args_cli}.")


if args_cli.list:
    registered = {k: dataclasses.asdict(v) for k, v in SPECS.items() if k in gym.registry}
    for spec in registered.values():
        spec["variant_overrides"] = dict(spec["variant_overrides"])
    with open(args_cli.list, "w", encoding="utf-8") as file:
        json.dump(registered, file, indent=2)
    simulation_app.close()
    sys.exit(0)

TASK = select_task()


class Video:
    """``rollout.mp4`` of ``--video_env``, or with ``--video_all`` one ``rollout_env<i>.mp4`` per env."""

    def __init__(self, env, output):
        import imageio.v2 as imageio

        self.env = env
        # One frame every render_stride control steps: play back at the simulated rate.
        fps = 1.0 / (env.step_dt * env.cfg.render_stride)
        self.envs = list(range(env.num_envs)) if args_cli.video_all else [args_cli.video_env]
        names = [f"rollout_env{i}.mp4" for i in self.envs] if args_cli.video_all else ["rollout.mp4"]
        self.writers = [imageio.get_writer(os.path.join(output, n), fps=fps, macro_block_size=1) for n in names]

    def add(self, active=None):
        """Append the current frame for the recorded envs; ``active`` (bool per env) limits it."""
        frames = self.env.tiled_cameras["side"].data.output["rgb"][..., :3].to(torch.uint8).cpu().numpy()
        for env_id, writer in zip(self.envs, self.writers):
            if active is None or bool(active[env_id]):
                writer.append_data(frames[env_id])

    def close(self):
        for writer in self.writers:
            writer.close()


MAX_ATTEMPTS = 4


def check(env, obs, reward, action, report):
    tensors = {"obs": obs["policy"], "reward": reward, "action": action}
    for name, tensor in tensors.items():
        if not torch.isfinite(tensor).all():
            report["nonfinite"].append(name)
        if tensor.device.type != torch.device(env.device).type:
            report["errors"].append(f"{name} is on {tensor.device}")
    if obs["policy"].shape != (env.num_envs, env.cfg.observation_space):
        report["errors"].append(f"obs shape {tuple(obs['policy'].shape)}")
    if not torch.isfinite(env.wire_state.centerline).all():
        report["nonfinite"].append("wire")


def run_random(env, report, video):
    steps = args_cli.steps or int(env.max_episode_length * 2 + 20)
    obs, _ = env.reset()
    motion, last = 0.0, env.arm.joint_pos.clone()
    for step in range(steps):
        action = 2.0 * torch.rand(env.num_envs, env.cfg.action_space, device=env.device) - 1.0
        obs, reward, terminated, truncated, _ = env.step(action)
        check(env, obs, reward, action, report)
        done = terminated | truncated
        report["episodes"] += int(done.sum())
        report["partial_resets"] += int(done.any() and not done.all())
        motion += float((env.arm.joint_pos - last).abs().sum())
        last = env.arm.joint_pos.clone()
        report["steps"] = step + 1
        if video and step % env.cfg.render_stride == 0:
            video.add()
        if step == steps // 3 and env.num_envs > 1:
            # A forced partial reset must leave the other envs running.
            before = env.episode_length_buf[1:].clone()
            env._reset_idx(torch.tensor([0], device=env.device))
            report["partial_resets"] += 1
            if not torch.equal(before, env.episode_length_buf[1:]):
                report["errors"].append("partial reset touched another env")
        if step == 2 * steps // 3:
            obs, _ = env.reset()
            report["full_resets"] += 1
    report["joint_motion"] = motion
    report["passed"] = not report["nonfinite"] and not report["errors"] and motion > 1e-3 and report["episodes"] > 0


def run_scripted(env, report, video):
    """Roll out until every env has given a rollout the simulator didn't break.

    A NaN or runaway wire is a simulator fault, not a policy result: as in training the
    env guard catches it and resets the env. That rollout is dropped and listed.
    """
    valid, dropped = [], []
    for attempt in range(MAX_ATTEMPTS):
        obs, _ = env.reset()
        policy = make_policy(env)
        steps = args_cli.steps or policy.max_steps
        # The recorded env's rollout ends at DONE; one more second of video, then stop.
        tail = int(1.0 / env.step_dt)
        tails = torch.full((env.num_envs,), tail, dtype=torch.long, device=env.device)
        for step in range(steps):
            action = policy.act()
            obs, reward, terminated, truncated, extras = env.step(action)
            check(env, obs, reward, action, report)
            policy.observe(extras)
            report["steps"] += 1
            if step % 300 == 0:
                progress(env, policy, attempt, step, steps)
            if policy.phase[args_cli.video_env] == policy.done_phase:
                tail -= 1
            tails -= (policy.phase == policy.done_phase).long()
            if video and step % env.cfg.render_stride == 0:
                if args_cli.video_all:
                    if attempt == 0:
                        video.add(tails >= 0)
                elif tail >= 0:
                    video.add()
            wanted = args_cli.stop_after_successes
            if wanted and attempt == 0:
                won = policy.success & ~policy.faulted
                if int(won.sum()) >= wanted and bool((tails[won] < 0).all()):
                    report["stopped_after_successes"] = wanted
                    break
            if policy.finished:
                break
        for rollout in policy.summary():
            rollout["attempt"] = attempt
            (dropped if rollout["faulted"] else valid).append(rollout)
        if len(valid) >= env.num_envs or report.get("stopped_after_successes"):
            break
    report["rollouts"] = valid
    report["dropped_rollouts"] = dropped
    report["success_count"] = sum(r["success"] for r in valid)
    report["failure_count"] = len(valid) - report["success_count"]
    enough = len(valid) >= env.num_envs
    report["passed"] = enough and report["failure_count"] == 0 and not report["nonfinite"] and not report["errors"]


def progress(env, policy, attempt: int, step: int, steps: int) -> None:
    """One line per call, so a stuck or failing rollout shows in the log long before it ends."""
    from collections import Counter

    phases = Counter(policy.PHASES[int(p)].split("_RETRY")[0] for p in policy.phase)
    holder = getattr(env, "wire_holder", None)
    held = f"{int(holder.holding.sum())}/{int(holder.engaged.sum())}" if holder is not None else "-"
    done = int((policy.phase == policy.done_phase).sum())
    seated = getattr(env.task, "seated", None)
    seated = int(seated().all(dim=1).sum()) if callable(seated) else "-"
    detail = ""
    if hasattr(env.task, "slot_fit"):
        # Per env: lowest wire height over the slot and its distance from the slot centre (mm, clip frame),
        # then the tool point's height above the table (mm).
        z, y = env.task.slot_fit(0)
        tool = (env.arm.ee_pos[:, 2] - env.scene.env_origins[:, 2] - env.cfg.table_top_z) * 1000
        detail = " slot_z " + str([round(v, 1) for v in (z * 1000).tolist()]) + " slot_y " + str(
            [round(v, 1) for v in (y * 1000).tolist()]) + " tool_z " + str([round(v) for v in tool.tolist()])
    if env.fault_log:
        last = env.fault_log[-1]
        e = last["envs"][0]
        detail += (f" last_fault step {last['step']} env {e} joint {last['joint_speed'][e]:.1f}"
                   f" ({last.get('fastest_joint', ['?'] * (e + 1))[e]}) port {last['port_speed'][e]:.1f}")
    print(f"[progress] attempt {attempt} step {step}/{steps} done {done}/{env.num_envs} "
          f"success {int(policy.success.sum())} faulted {int(policy.faulted.sum())} wire_held {held} seated {seated} "
          f"phases {dict(phases)}{detail}", flush=True)


@wirecraft_task_config(TASK, "rsl_rl_cfg_entry_point", args_cli, ORIGINAL_ARGV)
def main(env_cfg, agent_cfg):
    output = os.path.abspath(args_cli.output or os.path.join("logs", "wirecraft", TASK, args_cli.mode))
    os.makedirs(output, exist_ok=True)
    save_resolved(output, env_cfg, agent_cfg)
    report = {
        "task": TASK, "mode": args_cli.mode, "seed": env_cfg.seed, "num_envs": env_cfg.scene.num_envs, "steps": 0,
        "episodes": 0, "partial_resets": 0, "full_resets": 0, "nonfinite": [], "errors": [], "passed": False,
        "composition": env_cfg.composition, "argv": ORIGINAL_ARGV,
    }
    start = time.time()
    video = None
    try:
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        if args_cli.video:
            video = Video(env, output)
        (run_random if args_cli.mode == "random" else run_scripted)(env, report, video)
        report["faults"] = env.fault_count
        report["fault_log"] = env.fault_log[:20]
        env.close()
    except Exception:
        report["errors"].append(traceback.format_exc()[-1800:])
    finally:
        if video:
            video.close()
        report["wall_time_s"] = round(time.time() - start, 1)
        with open(os.path.join(output, "result.json"), "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, default=str)
    print(f"[wirecraft] {TASK} {args_cli.mode}: passed={report['passed']} errors={report['errors'][:1]}")
    RESULT["passed"] = bool(report["passed"])
    return report


RESULT = {"passed": False}


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
        # 0 only for a passing run; the details are in result.json.
        exit_code = 0 if RESULT["passed"] else 1
    except SystemExit as error:
        exit_code = error.code if isinstance(error.code, int) else 1
    except Exception:
        traceback.print_exc()
        exit_code = 1
    if exit_code:
        # SimulationApp.close() ends the process with status 0, so a failure exits before it.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    simulation_app.close()
