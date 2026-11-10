# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Run every registered combination through ``run.py`` and collect the results.

    python scripts/wirecraft/verify_matrix.py --output results --mode random scripted --jobs 3

The list comes from the real Gym registry, so a combination can't drop out of
the check by accident. Finished cells are kept; delete a cell's folder to redo it.
Extra arguments after ``--`` go to every ``run.py`` call.

Scripted policies are only expected to finish Insertion, and 1-clip routing and straight
channels with the stacked cuboid. The other variants are listed as ``not_expected`` in the summary and the
report instead of being run; ``--all_scripted`` runs them anyway.
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time

RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py")
SCRIPTED_NOT_EXPECTED = ("2Clip", "3Clip", "CurvedChannel", "USBA-1Clip", "USBA-StraightChannel")
"""Routing and channels hinge on the wire, not the port, so one port covers them."""


def launch(task: str, mode: str, args, extra: list[str]) -> dict:
    cell = os.path.join(args.output, mode, task)
    result = os.path.join(cell, "result.json")
    if os.path.exists(result) and not args.rerun:
        with open(result, encoding="utf-8") as file:
            data = json.load(file)
        if "exit_code" in data and (data["passed"] or not args.retry_failed):
            return data
    os.makedirs(cell, exist_ok=True)
    envs = args.random_envs if mode == "random" else args.scripted_envs
    video = ["--video", "--enable_cameras"] if mode == "scripted" and task in args.video_tasks else []
    command = [
        sys.executable, RUN, "--headless", "--task", task, "--mode", mode, "--seed", str(args.seed),
        "--num_envs", str(envs), "--output", cell, *video, *extra,
    ]
    start = time.time()
    with open(os.path.join(cell, "run.log"), "w", encoding="utf-8") as log:
        try:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout).returncode
        except subprocess.TimeoutExpired:
            code = 124
    data = {"task": task, "mode": mode, "passed": False, "errors": ["no result written"]}
    if os.path.exists(result):
        with open(result, encoding="utf-8") as file:
            data = json.load(file)
    data["exit_code"] = code
    data["command"] = command
    data["total_time_s"] = round(time.time() - start, 1)
    data["passed"] = bool(data.get("passed")) and code == 0
    with open(result, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    print(f"{mode:8s} {task:68s} passed={data['passed']} exit={code} {data['total_time_s']}s", flush=True)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", nargs="+", default=["random", "scripted"], choices=("random", "scripted"))
    parser.add_argument("--jobs", type=int, default=2, help="Isaac processes at once.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--random_envs", type=int, default=3)
    parser.add_argument("--scripted_envs", type=int, default=5, help="Each env is one rollout from its own reset.")
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--filter", default="", help="Only IDs containing this text.")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--retry_failed", action="store_true")
    parser.add_argument("--video_tasks", nargs="*", default=[])
    parser.add_argument("--all_scripted", action="store_true", help="Also run the variants with no finished scripted policy.")
    args, extra = parser.parse_known_args()
    extra = [a for a in extra if a != "--"]
    args.output = os.path.abspath(args.output)
    os.makedirs(args.output, exist_ok=True)
    registry = os.path.join(args.output, "registry.json")
    subprocess.run([sys.executable, RUN, "--headless", "--list", registry, *extra], check=True, stdout=subprocess.DEVNULL)
    with open(registry, encoding="utf-8") as file:
        tasks = [t for t in json.load(file) if t.startswith("WireCraft-") and args.filter in t]
    cells = [(task, mode) for mode in args.mode for task in tasks]
    not_expected = [
        (task, mode)
        for task, mode in cells
        if mode == "scripted" and not args.all_scripted and any(name in task for name in SCRIPTED_NOT_EXPECTED)
    ]
    cells = [cell for cell in cells if cell not in not_expected]
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(lambda cell: launch(*cell, args, extra), cells))
    with open(os.path.join(args.output, "summary.json"), "w", encoding="utf-8") as file:
        json.dump({"cells": results, "not_expected": [task for task, _ in not_expected]}, file, indent=2)
    failed = [f"{r['mode']} {r['task']}" for r in results if not r.get("passed")]
    print(f"{len(results) - len(failed)}/{len(results)} cells passed; {len(not_expected)} scripted cells not expected")
    print("\n".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
