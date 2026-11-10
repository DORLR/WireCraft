# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default='Isaac-UR5Gripper-Wire-Direct-v1', help="Name of the task.")
parser.add_argument(
    "--random_actions",
    action="store_true",
    default=False,
    help="If set, the agent will take random actions instead of zero actions.",
)
# NOTE: Add argument for camera activation list
parser.add_argument(
    "--cameras_activation_list",
    nargs="+",
    default=None,
    help="List of cameras to enable. Choices: side, side2, color, left, right, depth.",
)
parser.add_argument("--save_images", action="store_true", default=False, help="Save camera images to disk.")
parser.add_argument("--action_scale", type=float, default=None, help="Override the action scale for the environment from the command line.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Launch omniverse app
# This will start the Isaac Sim application without loading any specific stage (just an empty simulator).
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device, 
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        enable_cameras=bool(args_cli.enable_cameras) if hasattr(args_cli, "enable_cameras") else None,
        cameras_activation_list=args_cli.cameras_activation_list,
        save_images=args_cli.save_images,
    )

    if hasattr(args_cli, "action_scale") and args_cli.action_scale is not None:
        env_cfg.action_scale = args_cli.action_scale
        print(f"[INFO] Overriding action_scale with CLI argument: {args_cli.action_scale}")

    # Enable terminations to validate reset and randomization logic during development
    if hasattr(env_cfg, "enable_terminations"):
        env_cfg.enable_terminations = True
    if hasattr(env_cfg, "terminate_on_time_out"):
        env_cfg.terminate_on_time_out = True

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print(f"[INFO]: Zero Actions: {not args_cli.random_actions}")
    print(f"[INFO]: Random Actions: {args_cli.random_actions}")
    # reset environment
    env.reset()
    # env.close()
    # exit()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            if args_cli.random_actions:
                actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            else:
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
