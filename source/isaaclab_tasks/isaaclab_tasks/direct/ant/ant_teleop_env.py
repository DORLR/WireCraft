# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Teleoperation-friendly Ant environment config.

Provides a config class with a dummy `terminations` attribute so existing
teleoperation script (`teleop_se3_agent.py`) that expects manager-based
fields can run without modification.

Usage (example):
    ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
        --task isaaclab_tasks.direct.ant.ant_teleop_env:AntTeleopEnvCfg
"""
from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_tasks.direct.ant.ant_env import AntEnv, AntEnvCfg


class _DummyTerminations:
    """Minimal terminations container expected by teleop script.

    The script sets `env_cfg.terminations.time_out = None`.
    We provide the attribute so that assignment succeeds.
    """
    def __init__(self) -> None:
        # default episode timeout (seconds). Will be overwritten to None by teleop script.
        self.time_out: float | None = 15.0


@configclass
class AntTeleopEnvCfg(AntEnvCfg):
    """Ant config extended with dummy terminations for teleoperation script compatibility."""
    # add a dummy terminations namespace so teleop script doesn't error
    terminations: _DummyTerminations = _DummyTerminations()


class AntTeleopEnv(AntEnv):
    """Environment class reusing AntEnv implementation with teleop config."""
    cfg: AntTeleopEnvCfg

    def __init__(self, cfg: AntTeleopEnvCfg, render_mode: str | None = None, **kwargs):  # noqa: D401
        super().__init__(cfg, render_mode, **kwargs)
