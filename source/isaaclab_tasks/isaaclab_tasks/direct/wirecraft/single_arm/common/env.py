# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""The one DirectRLEnv for single-arm wire tasks.

Control flow per step: ``_pre_physics_step`` stores the action, ``_apply_action``
runs the robot adapter, then ``_get_dones``, ``_get_rewards``, ``_reset_idx`` and
``_get_observations``. The task object owns reward, success and task resets; the
wire backend owns wire resets; the robot adapter owns control.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera
from isaaclab.utils.math import quat_from_euler_xyz

from ...wire_base.articulation import ArticulationWire
from ...wire_base.connectors import ConnectorGeometry
from ...wire_base.deformable import DeformableWire
from .env_cfg import SingleArmWireEnvCfg, _import, resolve_env_cfg
from .wire_holder import WireHolder


class SingleArmWireEnv(DirectRLEnv):
    cfg: SingleArmWireEnvCfg

    def __init__(self, cfg: SingleArmWireEnvCfg, render_mode: str | None = None, **kwargs):
        resolve_env_cfg(cfg)
        # DirectRLEnv.__init__ calls _setup_scene, which needs these collaborators.
        self.cfg = cfg
        self.connector = ConnectorGeometry.from_family(cfg.connector.family, cfg.connector.socket_clearance)
        self.arm = _import(cfg.robot.adapter)(self)
        self.wire_backend = (ArticulationWire if cfg.wire.kind == "articulation" else DeformableWire)(self)
        self.task = cfg.task.task_class(self)
        self.wire_holder = WireHolder(self) if cfg.wire_holder.enabled else None
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self.actions = torch.zeros(self.num_envs, cfg.action_space, device=self.device)
        self.safe_low = torch.tensor(cfg.ee_safe_zone[0::2], device=self.device)
        self.safe_high = torch.tensor(cfg.ee_safe_zone[1::2], device=self.device)
        zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.out_of_bounds, self.joint_limit, self.port_drop = zeros.clone(), zeros.clone(), zeros.clone()
        self.faulted, self.episode_done = zeros.clone(), zeros.clone()
        self.fault_count = 0
        self.fault_log = []
        self.arm.initialize()
        self.wire_backend.initialize()
        self.task.initialize()
        self._update_state()
        if self.wire_holder is not None:
            self.wire_holder.initialize()

    # ---- scene -------------------------------------------------------------
    def _setup_scene(self):
        self.arm.spawn()
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.table = RigidObject(self.cfg.table_cfg)
        self.scene.rigid_objects["table"] = self.table
        self.wire_backend.spawn()
        self.task.spawn()
        if self.wire_holder is not None:
            self.wire_holder.spawn()
        self.scene.clone_environments(copy_from_source=False)
        self.arm.post_clone()
        self.task.post_clone()
        if self.wire_holder is not None:
            self.wire_holder.post_clone()
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.8, 0.8, 0.8))
        light.func("/World/Light", light)
        self.tiled_cameras = {}
        for name, cfg in self.cfg.camera_cfgs.items():
            self.tiled_cameras[name] = self.scene.sensors[name] = TiledCamera(cfg)

    # ---- shared state ------------------------------------------------------
    def sample_wire_pose(self, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-reset wire XY offset, rotation about the wire centre and lift; both backends use the same draw."""
        wire, device = self.cfg.wire, self.device
        uniform = lambda half: (torch.rand(count, device=device) * 2.0 - 1.0) * half  # noqa: E731
        xy = torch.zeros(count, 2, device=device)
        if wire.randomize_xy:
            half = torch.tensor(wire.randomize_xy_range, device=device)
            xy = (torch.rand(count, 2, device=device) * 2.0 - 1.0) * half
        yaw = uniform(wire.randomize_yaw)
        # Tilt and lift draw only when enabled, so tasks without them keep their random streams.
        zero = torch.zeros(count, device=device)
        roll, pitch = (uniform(wire.randomize_tilt), uniform(wire.randomize_tilt)) if wire.randomize_tilt else (zero, zero)
        low, high = wire.randomize_lift
        lift = low + torch.rand(count, device=device) * (high - low) if high > 0 else zero
        return xy, quat_from_euler_xyz(roll, pitch, yaw), lift

    def _update_state(self) -> None:
        self.arm.update()
        self.wire_state = self.wire_backend.state()
        origins = self.scene.env_origins
        self.ee_local_pos = self.arm.ee_pos - origins
        self.ee_local_quat = self.arm.ee_quat
        self.port_local_pos = self.wire_state.insertion_end_pos - origins
        self.tip_local_pos = self.wire_state.tip_pos - origins
        self.task.update()

    def _fault_mask(self) -> torch.Tensor:
        """Envs the simulation has lost, isolated from healthy envs.

        A wire that diverges reaches its ports, so the ports and the arm joints are enough
        to watch, and their speed gives it away well before any state turns NaN.
        """
        # Written as a passing test so that a NaN speed fails it.
        joints = self.arm.joint_speed() < self.cfg.fault_joint_speed
        return ~(joints & (self.wire_backend.port_speed() < self.cfg.fault_port_speed))

    # ---- RL interface ------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)
        self.arm.latch_control_state()
        if self.wire_holder is not None:
            self.wire_holder.engage(self.actions)

    def _apply_action(self) -> None:
        self.wire_backend.hold_far_end()
        if self.wire_holder is not None:
            self.wire_holder.track()
        self.arm.apply_action(self.actions, self.cfg.action_scale, self.cfg.fix_ee_orient, self.cfg.control_mode)

    def _get_observations(self) -> dict:
        self._update_state()
        observations = {"policy": torch.nan_to_num(self.task.observations(), nan=0.0, posinf=0.0, neginf=0.0)}
        if self.arm.ft_id is not None:
            # Kept out of "policy" so trained checkpoints keep their observation size.
            observations["ft"] = self.arm.wrench.clone()
        for name, camera in self.tiled_cameras.items():
            for data_type, data in camera.data.output.items():
                observations[name] = data / 255.0 if data_type == "rgb" else data.clone()
        return observations

    def _get_rewards(self) -> torch.Tensor:
        reward = self.task.rewards()
        # Tasks may add their own terms under "log". Success is logged as counts of the episodes that ended this
        # step and of those that succeeded; WireCraftOnPolicyRunner sums them over an iteration into
        # metrics/success_rate (skipped when no episode ended) instead of averaging per-step ratios.
        log = self.extras.get("log", {})
        log["counts/episodes_ended"] = self.episode_done.sum().float()
        log["counts/episodes_succeeded"] = (self.extras["is_success"].bool() & self.episode_done).sum().float()
        log["metrics/fault_rate"] = self.faulted.float().mean()
        self.extras["log"] = log
        reward = torch.where(self.out_of_bounds, reward + self.cfg.out_of_bound_penalty, reward)
        reward = torch.where(self.joint_limit, reward + self.cfg.joint_limit_penalty, reward)
        reward = torch.where(self.port_drop, reward + self.cfg.male_port_drop_penalty, reward)
        return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_state()
        self.faulted = self._fault_mask()
        if self.faulted.any():
            self.fault_count += int(self.faulted.sum())
            self.fault_log.append({
                "step": int(self.common_step_counter), "envs": self.faulted.nonzero().flatten().tolist(),
                "joint_speed": self.arm.joint_speed().tolist(), "port_speed": self.wire_backend.port_speed().tolist(),
                "fastest_joint": [self.arm.robot.joint_names[self.arm.arm_ids[int(i)]] for i in
                                  self.arm.robot.data.joint_vel[:, self.arm.arm_ids].abs().argmax(dim=-1).tolist()],
            })
        success = self.task.success() & ~self.faulted
        self.extras["is_success"] = success
        self.extras["faulted"] = self.faulted
        terminated = self.faulted.clone()
        if self.cfg.enable_terminations:
            xy = self.ee_local_pos[:, :2]
            self.out_of_bounds = ((xy < self.safe_low[:2]) | (xy > self.safe_high[:2])).any(dim=-1)
            self.joint_limit = self.arm.joint_limit_mask()
            low = self.cfg.port_drop_height
            self.port_drop = (self.wire_state.insertion_end_pos[:, 2] < low) | (self.wire_state.far_end_pos[:, 2] < low)
            terminated |= self.out_of_bounds | self.joint_limit | self.port_drop
        if self.cfg.terminate_on_success:
            terminated |= success
        if self.cfg.terminate_on_time_out:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
        else:
            time_out = torch.zeros_like(terminated)
        self.episode_done = terminated | time_out
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        self.arm.reset(env_ids)
        # A task may tie the far port to its own fixture, so the wire is placed after it.
        anchor = self.task.reset(env_ids)
        self.wire_backend.reset(env_ids, anchor)
        if self.wire_holder is not None and hasattr(self.wire_holder, "engaged"):
            self.wire_holder.reset(env_ids)
        # Physics state is written; refresh buffers before the first observation.
        self.scene.update(dt=0.0)
        self._update_state()
