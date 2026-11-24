# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Localhost TCP inference server for LeRobot policies (runs inside the lerobot conda env).

Spawned automatically by ``scripts/wirecraft/eval_lerobot.py`` so Isaac Sim (Python 3.11 / NumPy 1.x)
and LeRobot (Python 3.12 / NumPy 2.x) stay cleanly isolated in separate environments.
"""

from __future__ import annotations

import argparse
import pickle
import socket
import struct

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors


def _enc(obj):
    if isinstance(obj, np.ndarray):
        return {"__nd__": obj.tobytes(), "dt": str(obj.dtype), "sh": obj.shape}
    if isinstance(obj, dict):
        return {k: _enc(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_enc(x) for x in obj)
    return obj


def _dec(obj):
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


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 22))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _recv(sock: socket.socket):
    hdr = _recv_exact(sock, 4)
    if hdr is None:
        return None
    n = struct.unpack(">I", hdr)[0]
    body = _recv_exact(sock, n)
    return None if body is None else _dec(pickle.loads(body))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Local pretrained_model directory or HF Hub repo ID.")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_action_steps", type=int, default=None, help="Override policy.n_action_steps.")
    args = parser.parse_args()

    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(args.checkpoint).to(args.device).eval()
    if args.n_action_steps is not None and hasattr(policy.config, "n_action_steps"):
        policy.config.n_action_steps = args.n_action_steps
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.checkpoint)

    cam_keys = [k for k in policy.config.input_features if k.startswith("observation.images.")]
    n_act = int(getattr(policy.config, "n_action_steps", 1))
    use_chunk = hasattr(policy, "predict_action_chunk") and n_act > 1 and int(getattr(policy.config, "n_obs_steps", 1)) == 1

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 24)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 24)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(1)
    cams_str = ",".join(cam_keys)
    print(f"READY:{args.port} type={cfg.type} n_act={n_act} chunk_mode={use_chunk} cams={cams_str}", flush=True)

    conn, _ = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 24)
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 24)
    try:
        while True:
            msg = _recv(conn)
            if msg is None or msg["cmd"] == "close":
                _send(conn, {"ok": True})
                break
            if msg["cmd"] == "reset":
                policy.reset()
                _send(conn, {"ok": True, "n_action_steps": n_act, "chunk_mode": use_chunk, "camera_keys": cam_keys})
            elif msg["cmd"] == "infer":
                batch = {"observation.state": torch.from_numpy(msg["state"]).float()}
                for key, img in msg["images"].items():
                    batch[key] = torch.from_numpy(img).permute(0, 3, 1, 2).float() / 255.0
                if msg.get("task") is not None:
                    batch["task"] = msg["task"]
                batch = pre(batch)
                with torch.inference_mode():
                    if use_chunk:
                        chunk = policy.predict_action_chunk(batch)[:, :n_act]
                        steps = [post(chunk[:, t]).cpu().numpy().astype(np.float32) for t in range(chunk.shape[1])]
                        out = np.stack(steps, axis=1)  # (B, n_act, D)
                    else:
                        act = post(policy.select_action(batch)).cpu().numpy().astype(np.float32)
                        out = act[:, None, :]  # (B, 1, D)
                _send(conn, {"ok": True, "actions": out})
    finally:
        conn.close()
        srv.close()


if __name__ == "__main__":
    main()
