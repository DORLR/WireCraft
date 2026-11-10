# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Convert episodes from ``collect_demos.py`` into a LeRobot v3.0 dataset.

    python scripts/wirecraft/to_lerobot.py demos/ur5_insertion --repo_id <hf_user>/wirecraft_ur5_insertion \
        --root datasets/ur5_insertion [--only_success] [--push_to_hub [--private]]

Runs in any Python with ``lerobot`` >= 0.4 (dataset format v3.0; tested with 0.5.2) and needs no Isaac Sim.
``--push_to_hub`` uploads to the Hugging Face Hub as ``repo_id`` (log in first with ``huggingface-cli login``). Features:
``observation.state``, ``observation.environment_state``, ``action`` and one video
``observation.images.<camera>`` per recorded camera; the episode's ``task`` string is kept. Every episode must
have the same array shapes. Where each LeRobot episode came from (source file, collection run, checkpoint, seed,
config hash) is written to ``meta/wirecraft_provenance.json`` in the dataset.
"""

import argparse
import glob
import json
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episodes", help="Folder written by collect_demos.py.")
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--root", required=True, help="Output folder; must not exist yet.")
    parser.add_argument("--only_success", action="store_true")
    parser.add_argument("--vcodec", default="libsvtav1", help="Video codec; h264 is faster to encode.")
    parser.add_argument("--image_writer_threads", type=int, default=4, help="Threads writing frames before encoding.")
    parser.add_argument("--push_to_hub", action="store_true", help="Upload the dataset to the Hugging Face Hub.")
    parser.add_argument("--private", action="store_true", help="Make the uploaded dataset private.")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    with open(os.path.join(args.episodes, "info.json"), encoding="utf-8") as file:
        info = json.load(file)
    paths = sorted(glob.glob(os.path.join(args.episodes, "episode_*.npz")))
    if not paths:
        raise SystemExit(f"No episodes in {args.episodes}.")
    first = np.load(paths[0])
    action_names = [f"ee_d{a}" for a in ("x", "y", "z", "rx", "ry", "rz")][: first["action"].shape[1] - 1] + ["gripper"]
    features = {
        "observation.state": {"dtype": "float32", "shape": first["state"].shape[1:], "names": info["state_names"]},
        "observation.environment_state": {"dtype": "float32", "shape": first["environment_state"].shape[1:],
                                          "names": None},
        "action": {"dtype": "float32", "shape": first["action"].shape[1:], "names": action_names},
    }
    for camera in info["cameras"]:
        features[f"observation.images.{camera}"] = {
            "dtype": "video", "shape": first[f"image_{camera}"].shape[1:], "names": ["height", "width", "channels"],
        }
    dataset = LeRobotDataset.create(repo_id=args.repo_id, fps=info["fps"], features=features, root=args.root,
                                    robot_type=info["robot"], use_videos=bool(info["cameras"]), vcodec=args.vcodec,
                                    image_writer_threads=args.image_writer_threads)
    shapes = {k: first[k].shape[1:] for k in ("state", "environment_state", "action")}
    shapes.update({f"image_{c}": first[f"image_{c}"].shape[1:] for c in info["cameras"]})
    for path in paths:
        with np.load(path) as episode:
            wrong = {k: episode[k].shape[1:] for k in shapes if k not in episode or episode[k].shape[1:] != shapes[k]}
        if wrong:
            raise SystemExit(f"{path} has shapes {wrong}, the first episode {shapes}; the folder mixes datasets.")
    kept, provenance = 0, []
    for path in paths:
        episode = np.load(path)
        if args.only_success and not bool(episode["success"]):
            continue
        source = json.loads(str(episode["provenance"])) if "provenance" in episode else {}
        provenance.append({"episode_index": kept, "source": os.path.basename(path),
                           "success": bool(episode["success"]), **source})
        task = str(episode["task"])
        for t in range(len(episode["action"])):
            frame = {
                "observation.state": episode["state"][t],
                "observation.environment_state": episode["environment_state"][t],
                "action": episode["action"][t],
                "task": task,
            }
            for camera in info["cameras"]:
                frame[f"observation.images.{camera}"] = episode[f"image_{camera}"][t]
            dataset.add_frame(frame)
        dataset.save_episode()
        kept += 1
    dataset.finalize()
    with open(os.path.join(args.root, "meta", "wirecraft_provenance.json"), "w", encoding="utf-8") as file:
        json.dump({"collection": {k: v for k, v in info.items() if k != "runs"}, "runs": info.get("runs", []),
                   "episodes": provenance}, file, indent=1)
    print(f"[to_lerobot] {kept} episodes -> {args.root} ({args.repo_id}, {info['fps']} fps)")
    if args.push_to_hub:
        dataset.push_to_hub(private=args.private, tags=["wirecraft", "isaac-lab"])
        print(f"[to_lerobot] pushed to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
