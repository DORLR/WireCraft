# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Record scripted or RL rollouts as raw episodes for imitation learning.

    # scripted policy, two wrist/side cameras
    python scripts/wirecraft/collect_demos.py --headless --task WireCraft-UR5-Articulation-StackedCuboid-Insertion-v0 \
        --policy scripted --num_episodes 20 --num_envs 5 --cameras side2 color --output demos/ur5_insertion
    # trained RSL-RL checkpoint
    python scripts/wirecraft/collect_demos.py --headless --task WireCraft-UR5-Articulation-StackedCuboid-Insertion-v0 \
        --policy rl --checkpoint model.pt --num_episodes 20 --output demos/ur5_insertion_rl

Each episode is one ``episode_XXXXXX.npz`` holding, per control step (60 Hz):
``state`` (arm and gripper joints, tool position and quaternion), ``environment_state`` (the task's policy
observation), ``action`` (the env action actually applied) and ``image_<camera>`` (uint8 HxWx3), plus
``success`` and ``task``. ``scripts/wirecraft/to_lerobot.py`` turns the folder into a LeRobot v3.0 dataset.
Rollouts the simulator broke (fault guard) are never written. ``env.*``/``agent.*`` overrides and
``--experiment`` files apply as in training. A folder that already holds episodes is refused unless
``--append`` is given, which continues the numbering only when the new rollouts produce the same data: task,
policy, robot, fps, cameras and resolution, state names, state/observation/action shapes and the resolved env
config (seed and env count aside) must all match the folder. Every episode stores its ``run`` index and a
``provenance`` JSON (checkpoint, seed, mode, experiments, overrides, config hash); ``info.json`` keeps one entry
per collection run under ``runs``. The exit code is non-zero when fewer than ``--num_episodes`` episodes could be
written.
"""

import argparse
import datetime
import glob
import hashlib
import json
import os
import sys

from isaaclab.app import AppLauncher

ORIGINAL_ARGV = list(sys.argv[1:])

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", required=True)
parser.add_argument("--policy", required=True, choices=("scripted", "rl"))
parser.add_argument("--checkpoint", default=None, help="RSL-RL checkpoint for --policy rl.")
parser.add_argument("--mode", default=None, help="Config mode; scripted for --policy scripted, train for rl.")
parser.add_argument("--experiment", action="append", default=[], help="Experiment YAML; repeat in priority order.")
parser.add_argument("--append", action="store_true", help="Add episodes to a folder that already holds some.")
parser.add_argument("--num_episodes", type=int, default=10)
# Script defaults sit below --experiment files; only flags actually typed override an experiment.
parser.add_argument("--num_envs", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--cameras", nargs="*", default=[], help="Any of side, side2 and the robot's wrist cameras.")
parser.add_argument("--camera_resolution", type=int, default=224)
parser.add_argument("--only_success", action="store_true", help="Keep successful episodes only.")
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, rest = parser.parse_known_args()
SCRIPT_DEFAULTS = {"seed": parser.get_default("seed"), "scene": {"num_envs": parser.get_default("num_envs")}}
if args_cli.cameras:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + rest
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.direct.wirecraft.single_arm.common import compose  # noqa: E402
from isaaclab_tasks.direct.wirecraft.single_arm.common.registry import SPECS  # noqa: E402
from isaaclab_tasks.direct.wirecraft.single_arm.common.scripted import make_policy  # noqa: E402

TASK_TEXT = {
    "insertion": "insert the {connector} connector into its socket",
    "clip_routing": "route the wire through the clip",
    "lay_in_channel": "lay the wire into the channel",
}


class Recorder:
    """Per-env step buffers; an env's buffer is written once its rollout ends."""

    def __init__(self, env, cameras):
        self.env, self.cameras = env, cameras
        self.buffers = [self._empty() for _ in range(env.num_envs)]

    def _empty(self):
        return {"state": [], "environment_state": [], "action": [], **{f"image_{c}": [] for c in self.cameras}}

    def state(self) -> torch.Tensor:
        arm = self.env.arm
        return torch.cat((arm.joint_pos, self.env.ee_local_pos, self.env.ee_local_quat), dim=-1)

    def add(self, observation: torch.Tensor, action: torch.Tensor, recording: torch.Tensor) -> None:
        """Store the observation seen before ``action`` was applied, for the envs still recording."""
        state, images = self.state().cpu().numpy(), {}
        for camera in self.cameras:
            images[camera] = self.env.tiled_cameras[camera].data.output["rgb"][..., :3].to(torch.uint8).cpu().numpy()
        observation, action = observation.cpu().numpy(), action.cpu().numpy()
        for i in recording.nonzero().flatten().tolist():
            buffer = self.buffers[i]
            buffer["state"].append(state[i])
            buffer["environment_state"].append(observation[i])
            buffer["action"].append(action[i])
            for camera in self.cameras:
                buffer[f"image_{camera}"].append(images[camera][i])

    def take(self, env_id: int) -> dict:
        buffer, self.buffers[env_id] = self.buffers[env_id], self._empty()
        return {k: np.asarray(v, dtype=np.uint8 if k.startswith("image_") else np.float32) for k, v in buffer.items()}


def main():
    spec = SPECS[args_cli.task]
    mode = args_cli.mode or ("scripted" if args_cli.policy == "scripted" else "train")
    env_cfg = compose.compose_env_cfg(spec, mode, args_cli.experiment,
                                      script_defaults=SCRIPT_DEFAULTS)
    if args_cli.cameras:
        env_cfg.enable_cameras, env_cfg.cameras = True, list(args_cli.cameras)
        env_cfg.camera_resolution, env_cfg.render_stride = args_cli.camera_resolution, 1
    agent_cfg = compose.compose_agent_cfg(spec, args_cli.experiment)
    # Typed flags (--num_envs, --seed, --device, ...) and env.*/agent.* tokens, exactly as in training.
    compose.apply_cli(env_cfg, agent_cfg, compose.parse_cli_overrides(rest),
                      compose.legacy_overrides(args_cli, ORIGINAL_ARGV))
    first, previous = existing_episodes()
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    task_text = TASK_TEXT[spec.task].format(connector=spec.connector.replace("_", " "))
    schema = data_schema(base, spec, mode, task_text)
    check_append(previous, schema)
    run = len(previous.get("runs", []))
    provenance = {
        "run": run, "checkpoint": os.path.abspath(args_cli.checkpoint) if args_cli.checkpoint else None,
        "seed": base.cfg.seed, "num_envs": base.num_envs, "mode": mode,
        "experiments": [os.path.abspath(p) for p in args_cli.experiment], "cli": list(env_cfg.composition["cli"]),
        "config_hash": schema["config_hash"], "argv": ORIGINAL_ARGV,
        "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    rl_policy = wrapped = None
    if args_cli.policy == "rl":
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        if not args_cli.checkpoint:
            raise SystemExit("--policy rl needs --checkpoint.")
        wrapped = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=base.device)
        runner.load(args_cli.checkpoint)
        rl_policy = runner.get_inference_policy(device=base.device)

    saved, attempted, successes = 0, 0, 0
    while saved < args_cli.num_episodes and attempted < 10 * args_cli.num_episodes:
        recorder = Recorder(base, args_cli.cameras)
        count = base.num_envs
        success = torch.zeros(count, dtype=torch.bool, device=base.device)
        faulted = torch.zeros_like(success)
        with torch.inference_mode():
            if rl_policy is None:
                base.reset()
                policy = make_policy(base)
                recording = torch.ones_like(success)
                for _ in range(policy.max_steps):
                    observation = base.task.observations()
                    action = policy.act()
                    recorder.add(observation, action.clamp(-1.0, 1.0), recording)
                    _, _, _, _, extras = base.step(action)
                    policy.observe(extras)
                    recording &= policy.phase != policy.done_phase
                    if policy.finished:
                        break
                for rollout in policy.summary():
                    success[rollout["env"]] = rollout["success"]
                    faulted[rollout["env"]] = rollout["faulted"]
            else:
                observation, _ = wrapped.reset()
                recording = torch.ones_like(success)
                for _ in range(int(base.max_episode_length) + 5):
                    action = rl_policy(observation)
                    recorder.add(observation, action.clamp(-1.0, 1.0), recording)
                    observation, _, dones, _ = wrapped.step(action)
                    success |= base.extras["is_success"].bool() & recording
                    faulted |= base.faulted & recording
                    recording &= ~dones.bool()
                    if not recording.any():
                        break
        for i in range(count):
            attempted += 1
            episode = recorder.take(i)
            if faulted[i] or len(episode["action"]) == 0 or (args_cli.only_success and not success[i]):
                continue
            if saved >= args_cli.num_episodes:
                break
            np.savez_compressed(os.path.join(args_cli.output, f"episode_{first + saved:06d}.npz"), success=bool(success[i]),
                                task=task_text, run=run, provenance=json.dumps(provenance), **episode)
            saved += 1
            successes += int(success[i])
            print(f"[collect] episode {saved}/{args_cli.num_episodes}: {len(episode['action'])} steps, "
                  f"success={bool(success[i])}")
    provenance.update(first_episode=first, episodes=saved, successes=successes, rollouts_tried=attempted)
    runs = previous.get("runs", []) + [provenance]
    info = {
        **schema,
        "episodes": sum(r["episodes"] for r in runs), "successes": sum(r["successes"] for r in runs),
        "rollouts_tried": sum(r["rollouts_tried"] for r in runs), "runs": runs,
    }
    if saved:
        # The resolved env config of this run, next to the episodes it produced.
        from isaaclab.utils.io import dump_yaml

        dump_yaml(os.path.join(args_cli.output, "runs", f"run_{run:03d}_env.yaml"), base.cfg)
    if saved or not previous:
        with open(os.path.join(args_cli.output, "info.json"), "w", encoding="utf-8") as file:
            json.dump(info, file, indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "runs"} | {"this_run": provenance}, indent=1))
    env.close()
    if saved < args_cli.num_episodes:
        print(f"[collect] only {saved} of {args_cli.num_episodes} episodes after {attempted} rollouts", file=sys.stderr)
        return 1
    return 0


# Fields of the resolved env config that may differ between appended runs.
FREE_FIELDS = ("seed", "composition", ("scene", "num_envs"), ("sim", "device"))


def config_hash(cfg) -> str:
    """Hash of the resolved env config without the fields that are free to change between runs."""
    from isaaclab.utils import class_to_dict

    data = class_to_dict(cfg)
    for field in FREE_FIELDS:
        *parents, leaf = (field,) if isinstance(field, str) else field
        node = data
        for key in parents:
            node = node.get(key, {})
        node.pop(leaf, None)
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:16]


def data_schema(base, spec, mode: str, task_text: str) -> dict:
    """What every episode in the folder has in common: the format and the env that produced it."""
    cameras = list(args_cli.cameras)
    return {
        "format": "wirecraft-episodes-v1", "task": args_cli.task, "task_text": task_text, "policy": args_cli.policy,
        "mode": mode, "robot": spec.robot, "fps": round(1.0 / base.step_dt), "cameras": cameras,
        "camera_resolution": [base.cfg.camera_resolution] * 2 if cameras else None,
        "state_names": [f"joint_{i}" for i in range(base.arm.joint_pos.shape[1])]
        + ["ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz"],
        "environment_state_dim": int(base.cfg.observation_space), "action_dim": int(base.cfg.action_space),
        "config_hash": config_hash(base.cfg),
    }


def existing_episodes() -> tuple[int, dict]:
    """Index of the next episode file and the folder's previous info; refuses to overwrite."""
    os.makedirs(args_cli.output, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args_cli.output, "episode_*.npz")))
    if not paths:
        return 0, {}
    if not args_cli.append:
        raise SystemExit(f"{args_cli.output} already holds {len(paths)} episodes; pass --append or pick a new folder.")
    with open(os.path.join(args_cli.output, "info.json"), encoding="utf-8") as file:
        previous = json.load(file)
    previous["last_episode"] = paths[-1]
    return int(os.path.basename(paths[-1])[8:14]) + 1, previous


def check_append(previous: dict, schema: dict) -> None:
    """Refuse to append rollouts whose format or env differs from the episodes already in the folder."""
    if not previous:
        return
    if previous.get("format") != schema["format"] or "runs" not in previous:
        raise SystemExit(f"{args_cli.output} was written by an older collect_demos.py without provenance; "
                         "collect into a new folder.")
    differ = {k: (previous.get(k), v) for k, v in schema.items() if previous.get(k) != v}
    # The stored arrays themselves, not only info.json: shapes of the last episode on disk.
    with np.load(previous["last_episode"]) as last:
        stored = {"environment_state_dim": last["environment_state"].shape[1], "action_dim": last["action"].shape[1],
                  "state_dim": last["state"].shape[1]}
        for camera in schema["cameras"]:
            stored[f"image_{camera}"] = list(last[f"image_{camera}"].shape[1:3]) if f"image_{camera}" in last else None
    wanted = {"environment_state_dim": schema["environment_state_dim"], "action_dim": schema["action_dim"],
              "state_dim": len(schema["state_names"]),
              **{f"image_{c}": schema["camera_resolution"] for c in schema["cameras"]}}
    differ.update({f"stored {k}": (stored[k], v) for k, v in wanted.items() if stored[k] != v})
    if differ:
        lines = "\n".join(f"  {k}: folder {old!r}, this run {new!r}" for k, (old, new) in differ.items())
        raise SystemExit(f"--append refused, {args_cli.output} holds different data:\n{lines}\n"
                         "Collect into a new folder (seed and --num_envs may differ; nothing else may).")


if __name__ == "__main__":
    try:
        code = main()
    except SystemExit as error:
        print(f"[collect] {error}", file=sys.stderr)
        code = 1
    if code:
        # SimulationApp.close() ends the process with status 0, so a failure exits before it.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    simulation_app.close()
