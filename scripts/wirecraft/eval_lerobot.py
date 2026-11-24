# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Closed-loop evaluation of LeRobot policies (e.g. ACT) in WireCraft.

Spawns ``lerobot_policy_server.py`` inside the separate LeRobot Python environment and drives
WireCraft in joint-target mode (``control_mode="joint"``, ``action_scale=1.0``), converting
predicted absolute joint targets into joint deltas and remapping the gripper target at its
midpoint (0.3925 rad for UR5 Robotiq 2F-85: <0.3925 -> open, >=0.3925 -> close).

Example (CoRL 2026 connector insertion, 64 rollouts = 4 seeds x 16 envs, 800 steps):

    python scripts/wirecraft/eval_lerobot.py --headless \
        --task WireCraft-UR5-Deformable-StackedCuboid-Insertion-v0 \
        --experiment scripts/wirecraft/experiments/corl2026_insertion.yaml \
        --checkpoint /path/to/pretrained_model \
        --lerobot_python /path/to/lerobot/bin/python \
        --num_envs 16 --seeds 1000,2000,3000,4000 --episode_length 800 \
        --output eval_insertion.json

Example (CoRL 2026 1-clip routing, 900 steps):

    python scripts/wirecraft/eval_lerobot.py --headless \
        --task WireCraft-UR5-Deformable-StackedCuboid-1Clip-Routing-v0 \
        --experiment scripts/wirecraft/experiments/corl2026_clip_routing.yaml \
        --checkpoint /path/to/clip_route_act_100k \
        --lerobot_python /path/to/lerobot/bin/python \
        --num_envs 16 --seeds 1000,2000,3000,4000 --episode_length 900 \
        --output eval_clip_routing.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import socket
import struct
import subprocess
import sys
import time

from isaaclab.app import AppLauncher

ORIGINAL_ARGV = list(sys.argv[1:])

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", required=True, help="Registered WireCraft Gym task ID.")
parser.add_argument("--checkpoint", required=True, help="Local pretrained_model path or HuggingFace Hub repo ID.")
parser.add_argument("--lerobot_python", required=True, help="Path to Python executable in the LeRobot environment.")
parser.add_argument("--experiment", action="append", default=[], help="Experiment YAML file(s) applied in order.")
parser.add_argument("--mode", default="scripted", help="Base config mode (default: scripted).")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seeds", default="1000,2000,3000,4000", help="Comma-separated evaluation seeds.")
parser.add_argument("--episode_length", type=int, default=800,
                    help="Rollout horizon in env control steps (60 Hz: 800 for insertion, 900 for clip routing).")
parser.add_argument("--cameras", nargs="+", default=["side2", "color"], help="WireCraft camera names.")
parser.add_argument("--camera_prefix", default="cam_",
                    help="Prefix mapping env camera name -> LeRobot feature observation.images.<prefix><cam>.")
parser.add_argument("--camera_resolution", type=int, default=512)
parser.add_argument("--n_action_steps", type=int, default=None, help="Override policy.n_action_steps.")
parser.add_argument("--task_prompt", default=None, help="Optional language instruction for VLA policies.")
parser.add_argument("--save_snapshots", default=None, help="Prefix path to save env-0 PNGs at steps 0/200/400.")
parser.add_argument("--output", default=None, help="Optional JSON path to write per-seed and aggregate metrics.")
AppLauncher.add_app_launcher_args(parser)
args_cli, rest = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + rest

SCRIPT_DEFAULTS = {"scene": {"num_envs": parser.get_default("num_envs")}}


def _enc(obj):
    import numpy as np

    if isinstance(obj, np.ndarray):
        return {"__nd__": obj.tobytes(), "dt": str(obj.dtype), "sh": obj.shape}
    if isinstance(obj, dict):
        return {k: _enc(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_enc(x) for x in obj)
    return obj


def _dec(obj):
    import numpy as np

    if isinstance(obj, dict):
        if "__nd__" in obj:
            return np.frombuffer(obj["__nd__"], dtype=obj["dt"]).reshape(obj["sh"]).copy()
        return {k: _dec(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_dec(x) for x in obj)
    return obj


def _send(sock: socket.socket, obj) -> None:
    data = pickle.dumps(_enc(obj), protocol=4)
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 22))
        if not chunk:
            raise ConnectionError("lerobot_policy_server closed unexpectedly")
        buf.extend(chunk)
    return bytes(buf)


def _recv(sock: socket.socket):
    n = struct.unpack(">I", _recv_exact(sock, 4))[0]
    return _dec(pickle.loads(_recv_exact(sock, n)))


def start_policy_server(checkpoint: str, lerobot_py: str, n_action_steps: int | None):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lerobot_policy_server.py")
    cmd = [lerobot_py, script, "--checkpoint", checkpoint, "--port", str(port)]
    if n_action_steps is not None:
        cmd += ["--n_action_steps", str(n_action_steps)]
    env = {
        k: v for k, v in os.environ.items()
        if k not in ("PYTHONPATH", "LD_LIBRARY_PATH") and not k.startswith(("CARB_", "OMNI_", "ISAAC"))
    }
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    ready = False
    t0 = time.monotonic()
    while time.monotonic() - t0 < 180:
        if proc.poll() is not None:
            raise RuntimeError(f"lerobot_policy_server exited ({proc.returncode}):\n{proc.stdout.read()}")
        line = proc.stdout.readline()
        if line:
            print(f"[policy_server] {line.rstrip()}", flush=True)
            if line.startswith("READY:"):
                ready = True
                break
    if not ready:
        proc.terminate()
        raise RuntimeError("Timed out waiting for lerobot_policy_server")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 24)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 24)
    sock.connect(("127.0.0.1", port))
    return proc, sock


# Start policy server before Isaac Sim claims GPU memory
_server_proc, _server_sock = start_policy_server(args_cli.checkpoint, args_cli.lerobot_python, args_cli.n_action_steps)

simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.direct.wirecraft.single_arm.common import compose  # noqa: E402
from isaaclab_tasks.direct.wirecraft.single_arm.common.registry import SPECS  # noqa: E402


def main() -> None:
    seeds = [int(s) for s in args_cli.seeds.split(",") if s.strip()]
    spec = SPECS[args_cli.task]
    env_cfg = compose.compose_env_cfg(spec, args_cli.mode, args_cli.experiment, script_defaults=SCRIPT_DEFAULTS)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.control_mode = "joint"
    env_cfg.fix_ee_orient = False
    env_cfg.action_scale = 1.0
    env_cfg.enable_cameras = True
    env_cfg.cameras = list(args_cli.cameras)
    env_cfg.camera_resolution = args_cli.camera_resolution
    env_cfg.render_stride = 1
    env_cfg.rerender_on_reset = True
    env_cfg.enable_terminations = False
    env_cfg.terminate_on_success = False
    env_cfg.terminate_on_time_out = False
    env_cfg.episode_length_s = 1e6

    agent_cfg = compose.compose_agent_cfg(spec, args_cli.experiment)
    compose.apply_cli(env_cfg, agent_cfg, compose.parse_cli_overrides(rest),
                      compose.legacy_overrides(args_cli, ORIGINAL_ARGV))

    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    arm_dim = len(base.arm.arm_ids)
    grip_mid = 0.5 * (float(base.cfg.robot.gripper_open_pos) + float(base.cfg.robot.gripper_closed_pos))
    task_kind = spec.task  # "insertion", "clip_routing", or "lay_in_channel"

    # Warm up renderer once
    base.reset()
    zero = torch.zeros(args_cli.num_envs, arm_dim + 1, device=base.device)
    for _ in range(3):
        base.step(zero)

    def read_obs():
        state = base.arm.joint_pos.cpu().numpy().astype(np.float32)
        images = {}
        for cam in args_cli.cameras:
            rgb = base.tiled_cameras[cam].data.output["rgb"][..., :3].to(torch.uint8).cpu().numpy()
            images[f"observation.images.{args_cli.camera_prefix}{cam}"] = rgb
        return state, images

    def step_metrics():
        if task_kind == "insertion":
            tip = base.task.tip_in_socket()
            grasped = base.arm.is_grasping

            def in_box(off, half):
                o = torch.tensor(off, device=base.device)
                h = torch.tensor(half, device=base.device)
                return ((tip - o).abs() <= h).all(dim=-1)

            # Exact CoRL 2026 multi-tier zones (Appendix E / eval_multi_tier)
            t0_raw = in_box((0.0, 0.0, 0.0), (0.010, 0.005, 0.005))
            t1_raw = in_box((-0.005, 0.0, 0.0), (0.015, 0.005, 0.005))
            t2_raw = in_box((-0.005, 0.0, 0.0), (0.020, 0.010, 0.010))
            return {
                "T0_insert": t0_raw & grasped,
                "T1": t1_raw & grasped,
                "T2_reach": t2_raw & grasped,
                "T0_raw": t0_raw,
                "T2_raw": t2_raw,
                "wc_is_seated": base.task.is_seated(),
                "grasped": grasped,
                "faulted": base.faulted,
            }
        elif task_kind == "clip_routing":
            in_slot = base.task.seated().all(dim=1)
            is_open = base.arm.is_released
            return {
                "in_slot": in_slot,
                "in_slot_open": in_slot & is_open,
                "wc_success": base.task.success(),
                "faulted": base.faulted,
            }
        else:
            frac = base.task.in_channel_fraction
            # Paper lay_in_channel_task normalized by geometric max (channel_length / wire_length = 0.33 / 0.40 = 0.825)
            max_frac = min(1.0, 0.33 / max(float(base.cfg.wire.length), 1e-6))
            return {
                "paper_ratio_75": (frac / max_frac) >= 0.75,
                "wc_success": base.task.success(),
                "faulted": base.faulted,
                "frac": frac,
            }

    all_seeds = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        base.reset()
        _send(_server_sock, {"cmd": "reset"})
        meta = _recv(_server_sock)
        chunk_len = int(meta["n_action_steps"]) if meta["chunk_mode"] else 1

        if task_kind == "insertion":
            ever = {k: torch.zeros(args_cli.num_envs, dtype=torch.bool, device=base.device)
                    for k in ("T0_insert", "T1", "T2_reach", "T0_raw", "T2_raw", "wc_is_seated", "grasped", "faulted")}
        elif task_kind == "clip_routing":
            ever = {k: torch.zeros(args_cli.num_envs, dtype=torch.bool, device=base.device)
                    for k in ("in_slot", "in_slot_open", "paper_sustain20_open", "wc_success", "faulted")}
            sustain = torch.zeros(args_cli.num_envs, dtype=torch.long, device=base.device)
        else:
            ever = {k: torch.zeros(args_cli.num_envs, dtype=torch.bool, device=base.device)
                    for k in ("paper_ratio_75", "wc_success", "faulted")}
            best_frac = torch.zeros(args_cli.num_envs, device=base.device)

        t0 = time.monotonic()
        chunk = None
        for step in range(args_cli.episode_length):
            k_in = step % chunk_len
            if k_in == 0:
                state, images = read_obs()
                if args_cli.save_snapshots and seed == seeds[0] and step in (0, 200, 400):
                    for cam in args_cli.cameras:
                        img = images[f"observation.images.{args_cli.camera_prefix}{cam}"][0]
                        Image.fromarray(img).save(f"{args_cli.save_snapshots}_s{step:03d}_{cam}.png")
                payload = {"cmd": "infer", "state": state, "images": images}
                if args_cli.task_prompt is not None:
                    payload["task"] = [args_cli.task_prompt] * args_cli.num_envs
                _send(_server_sock, payload)
                chunk = _recv(_server_sock)["actions"]  # (N, chunk_len, D)
                if step > 0 and step % 200 == 0:
                    print(f"[eval] seed={seed} step={step}/{args_cli.episode_length} dt={time.monotonic()-t0:.1f}s", flush=True)
            else:
                state = base.arm.joint_pos.cpu().numpy().astype(np.float32)

            pred = torch.from_numpy(chunk[:, k_in, :]).to(base.device)
            cur = torch.from_numpy(state).to(base.device)
            env_act = torch.empty(args_cli.num_envs, arm_dim + 1, device=base.device)
            env_act[:, :arm_dim] = pred[:, :arm_dim] - cur[:, :arm_dim]
            env_act[:, arm_dim] = torch.where(
                pred[:, arm_dim] < grip_mid,
                torch.tensor(-1.0, device=base.device),
                torch.tensor(1.0, device=base.device),
            )
            base.step(env_act)
            m = step_metrics()
            if task_kind == "insertion":
                for k in ever:
                    ever[k] |= m[k]
            elif task_kind == "clip_routing":
                ever["in_slot"] |= m["in_slot"]
                ever["in_slot_open"] |= m["in_slot_open"]
                ever["wc_success"] |= m["wc_success"]
                ever["faulted"] |= m["faulted"]
                sustain = torch.where(m["in_slot_open"], sustain + 1, torch.zeros_like(sustain))
                ever["paper_sustain20_open"] |= sustain >= 20
            else:
                ever["paper_ratio_75"] |= m["paper_ratio_75"]
                ever["wc_success"] |= m["wc_success"]
                ever["faulted"] |= m["faulted"]
                best_frac = torch.maximum(best_frac, m["frac"])

        row = {"seed": seed, "n_episodes": args_cli.num_envs, "elapsed_s": round(time.monotonic() - t0, 1)}
        for k, v in ever.items():
            row[k] = round(float(v.float().mean().item()) * 100.0, 2)
        if task_kind == "lay_in_channel":
            row["mean_best_fraction"] = round(float(best_frac.mean().item()) * 100.0, 2)
        all_seeds.append(row)
        print(f"[SEED] {json.dumps(row)}", flush=True)

    metric_keys = [k for k in all_seeds[0] if k not in ("seed", "n_episodes", "elapsed_s")]
    summary = {
        "task": args_cli.task,
        "checkpoint": args_cli.checkpoint,
        "experiments": args_cli.experiment,
        "episode_length": args_cli.episode_length,
        "total_episodes": args_cli.num_envs * len(seeds),
        "seeds": all_seeds,
        "mean_pct": {k: round(float(np.mean([r[k] for r in all_seeds])), 2) for k in metric_keys},
        "std_pct": {k: round(float(np.std([r[k] for r in all_seeds])), 2) for k in metric_keys},
    }
    print(f"[SUMMARY] {json.dumps(summary, indent=2)}", flush=True)
    if args_cli.output:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
        with open(args_cli.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    try:
        _send(_server_sock, {"cmd": "close"})
        _recv(_server_sock)
    except Exception:
        pass
    _server_sock.close()
    _server_proc.wait(timeout=10)
    env.close()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
