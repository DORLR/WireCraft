# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Layered env/agent configuration.

Lowest to highest priority:

* env: defaults, robot, task, wire, connector, mode, experiment YAML files in
  the order given, explicit ``env.*`` on the CLI.
* agent: shared defaults, task, robot/wire, experiment YAML files, explicit
  ``agent.*`` on the CLI.

Merge rules, the same for every layer:

* Mappings merge key by key; lists and tuples are replaced whole.
* An unknown key raises ``KeyError``; a wrong type raises ``ValueError``.
  An int is accepted for a float field, never the other way round, and a bool
  is never accepted as a number.
* Each preset may only touch the top-level keys listed in ``ENV_LAYER_KEYS``.
* Experiment YAML holds ``env:`` and ``agent:`` sections. Neither side is ever
  written from the other's section.
* Fields derived by :func:`resolve_env_cfg` can't be set by any layer.
"""

from __future__ import annotations

import copy
import functools
import importlib
import os
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import yaml

from isaaclab.utils.string import string_to_callable

from .agents.rsl_rl_ppo_cfg import WireCraftPPORunnerCfg
from .env_cfg import MODES, SingleArmWireEnvCfg, resolve_env_cfg

PRESET_DIR = os.path.join(os.path.dirname(__file__), "presets")

_SCENE_KEYS = ("episode_length_s", "decimation", "control_mode", "fix_ee_orient", "action_scale")
ENV_LAYER_KEYS = {
    "robot": ("robot",),
    "task": (
        "task", "wire", "wire_holder", "gripper_close_on_wire", "enable_terminations", "terminate_on_time_out", "terminate_on_success", "ee_safe_zone",
        "fault_joint_speed", "fault_port_speed", *_SCENE_KEYS,
    ),
    "wire": ("wire", "task", "connector", "episode_length_s", "fix_ee_orient", "action_scale"),
    "connector": ("connector",),
    "mode": (
        "mode", "wire", "task", "scene", "cameras", "enable_cameras", "camera_resolution", "render_stride",
        "enable_terminations", "terminate_on_time_out", "terminate_on_success", *_SCENE_KEYS,
    ),
}
DERIVED_ENV_FIELDS = (
    "resolved", "wire_prim_path", "wire_spawn", "table_cfg", "camera_cfgs", "action_space", "observation_space",
    "state_space", "composition", "task.socket_cfg", "task.backboard_cfg", "task.pillar_height_resolved",
    "task.clip_cfgs", "task.module_cfgs", "task.module_poses", "task.line_xy", "task.task_class",
)


@dataclass(frozen=True)
class TaskSpec:
    """One registered combination; the Gym ID maps to exactly one of these."""

    robot: str
    robot_cfg: str
    """``module:function`` returning a fresh :class:`RobotCfg`."""
    task: str
    wire: str
    connector: str
    variant: str = ""
    variant_overrides: Mapping = field(default_factory=dict)
    leaf_dir: str = ""
    """Robot/task folder that may carry ``task_cfg/`` and ``agents/`` files."""


def _coerce(current, value, where: str):
    if value is None or current is None:
        return value
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{where}: expected bool, got {value!r}.")
    elif isinstance(current, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where}: expected float, got {value!r}.")
        return float(value)
    elif isinstance(current, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{where}: expected int, got {value!r}.")
    elif isinstance(current, str):
        if not isinstance(value, str):
            raise ValueError(f"{where}: expected str, got {value!r}.")
    elif isinstance(current, (list, tuple)):
        if isinstance(value, (str, Mapping)) or not isinstance(value, Sequence):
            raise ValueError(f"{where}: expected a list, got {value!r}.")
        if isinstance(current, tuple):
            if len(value) != len(current):
                raise ValueError(f"{where}: expected {len(current)} items, got {len(value)}.")
            return tuple(_coerce(c, v, f"{where}[{i}]") for i, (c, v) in enumerate(zip(current, value)))
        return list(value)
    elif isinstance(current, dict) or hasattr(current, "__dataclass_fields__"):
        raise ValueError(f"{where}: expected a mapping, got {value!r}.")
    elif callable(current):
        return string_to_callable(value) if isinstance(value, str) else value
    return value


def apply_overrides(target, data: Mapping, path: str = "", derived: Sequence[str] = ()) -> None:
    """Merge ``data`` into a configclass or dict under the module's merge rules."""
    for key, value in data.items():
        where = f"{path}.{key}" if path else key
        if where in derived:
            raise ValueError(f"{where} is derived during resolve and can't be overridden.")
        if isinstance(target, dict):
            current = target.get(key)
        elif hasattr(target, key):
            current = getattr(target, key)
        else:
            raise KeyError(f"Unknown config field {where!r}.")
        if isinstance(value, Mapping) and current is not None:
            if isinstance(current, (bool, int, float, str, list, tuple)):
                raise ValueError(f"{where}: expected a scalar or list, got a mapping.")
            apply_overrides(current, value, where, derived)
            continue
        value = _coerce(current, value, where)
        if isinstance(target, dict):
            target[key] = value
        else:
            setattr(target, key, value)


def read_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a mapping.")
    return data


def apply_env_layer(cfg, layer: str, data: Mapping) -> None:
    allowed = ENV_LAYER_KEYS[layer]
    stray = sorted(set(data) - set(allowed))
    if stray:
        raise ValueError(f"The {layer} preset can't set {stray}; allowed keys are {list(allowed)}.")
    apply_overrides(cfg, data, derived=DERIVED_ENV_FIELDS)


def _files(spec: TaskSpec, mode: str, layer: str) -> list[str]:
    """Common preset first, then the robot/task folder's small difference if any."""
    name = {"task": spec.task, "wire": spec.wire, "connector": spec.connector, "mode": mode}[layer]
    paths = [os.path.join(PRESET_DIR, layer, f"{name}.yaml")]
    if layer == "mode":
        paths.append(os.path.join(PRESET_DIR, layer, f"{name}_{spec.task}.yaml"))
    if spec.leaf_dir:
        paths.append(os.path.join(spec.leaf_dir, "task_cfg", f"{layer}_{name}.yaml"))
    return [p for p in paths if os.path.exists(p)]


def _experiment_sections(paths: Sequence[str]) -> list[tuple[str, dict]]:
    sections = []
    for path in paths:
        data = read_yaml(path)
        stray = sorted(set(data) - {"env", "agent"})
        if stray:
            raise ValueError(f"{path}: top-level keys must be 'env' and 'agent', found {stray}.")
        sections.append((path, data))
    return sections


def _load_callable(path: str) -> Callable:
    module, name = path.split(":")
    return getattr(importlib.import_module(module), name)


def compose_env_cfg(spec: TaskSpec, mode: str = "train", experiments: Sequence[str] = (),
                    script_defaults: Mapping | None = None) -> SingleArmWireEnvCfg:
    """Build unresolved env inputs for one spec. Every call returns independent objects.

    ``script_defaults`` (nested, e.g. ``{"seed": 42, "scene": {"num_envs": 50}}``) is a calling script's own
    defaults: applied after the presets and before the experiment files, so an experiment still wins over them.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}.")
    cfg = SingleArmWireEnvCfg()
    record = [f"defaults:{type(cfg).__name__}"]
    cfg.robot = _load_callable(spec.robot_cfg)()
    record.append(f"robot:{spec.robot_cfg}")
    cfg.task = _task_cfg(spec.task)
    cfg.wire.kind = spec.wire
    cfg.connector.family = spec.connector
    cfg.variant = spec.variant
    for layer in ("task", "wire", "connector", "mode"):
        for path in _files(spec, mode, layer):
            apply_env_layer(cfg, layer, read_yaml(path))
            record.append(f"{layer}:{os.path.relpath(path, os.path.dirname(PRESET_DIR))}")
        if layer == "task" and spec.variant_overrides:
            apply_env_layer(cfg, "task", copy.deepcopy(dict(spec.variant_overrides)))
            record.append(f"task:variant={spec.variant}")
    cfg.mode = mode
    if script_defaults:
        apply_overrides(cfg, copy.deepcopy(dict(script_defaults)), derived=DERIVED_ENV_FIELDS)
        record.append(f"script_defaults:{dict(script_defaults)}")
    for path, data in _experiment_sections(experiments):
        if "env" in data:
            apply_overrides(cfg, data["env"], derived=DERIVED_ENV_FIELDS)
            record.append(f"experiment:{path}")
    cfg.composition = {"spec": _spec_record(spec), "mode": mode, "env_layers": record, "cli": []}
    return cfg


def _task_cfg(name: str):
    from .tasks import ClipRoutingCfg, InsertionCfg, LayInChannelCfg

    return {"insertion": InsertionCfg, "clip_routing": ClipRoutingCfg, "lay_in_channel": LayInChannelCfg}[name]()


def _spec_record(spec: TaskSpec) -> dict:
    return {k: getattr(spec, k) for k in ("robot", "task", "wire", "connector", "variant")}


def compose_agent_cfg(spec: TaskSpec, experiments: Sequence[str] = ()) -> WireCraftPPORunnerCfg:
    cfg = WireCraftPPORunnerCfg()
    parts = [spec.robot, spec.wire, spec.connector, spec.task, spec.variant]
    cfg.experiment_name = "wirecraft_" + "_".join(p for p in parts if p)
    record = [f"defaults:{type(cfg).__name__}"]
    paths = []
    if spec.leaf_dir:
        paths.append(("task", os.path.join(spec.leaf_dir, "agents", "rsl_rl_ppo.yaml")))
        paths.append(("robot_wire", os.path.join(spec.leaf_dir, "agents", f"rsl_rl_ppo_{spec.wire}.yaml")))
    for layer, path in paths:
        if os.path.exists(path):
            apply_overrides(cfg, read_yaml(path))
            record.append(f"{layer}:{path}")
    for path, data in _experiment_sections(experiments):
        if "agent" in data:
            apply_overrides(cfg, data["agent"])
            record.append(f"experiment:{path}")
    cfg.composition = {"agent_layers": record, "cli": []}
    return cfg


def resolve_agent_cfg(cfg: WireCraftPPORunnerCfg) -> WireCraftPPORunnerCfg:
    if cfg.max_iterations < 1 or cfg.num_steps_per_env < 1:
        raise ValueError("agent.max_iterations and agent.num_steps_per_env must be positive.")
    if not cfg.experiment_name:
        raise ValueError("agent.experiment_name must not be empty.")
    return cfg


# ---- Gym entry points ------------------------------------------------------
_context = {"mode": "train", "experiments": ()}


def configure(mode: str = "train", experiments: Sequence[str] = ()) -> None:
    """Select the mode and experiment files read by the Gym cfg entry points."""
    _context.update(mode=mode, experiments=tuple(experiments))


def env_entry_point(spec: TaskSpec) -> Callable:
    def build_env_cfg():
        return compose_env_cfg(spec, _context["mode"], _context["experiments"])

    return build_env_cfg


def agent_entry_point(spec: TaskSpec) -> Callable:
    def build_agent_cfg():
        return compose_agent_cfg(spec, _context["experiments"])

    return build_agent_cfg


# ---- Legacy flags ----------------------------------------------------------
LEGACY_FLAGS = {
    "num_envs": ("env.scene.num_envs",),
    "seed": ("env.seed", "agent.seed"),
    "max_iterations": ("agent.max_iterations",),
    "resume": ("agent.resume",),
    "load_run": ("agent.load_run",),
    "checkpoint": ("agent.load_checkpoint",),
    "run_name": ("agent.run_name",),
    "logger": ("agent.logger",),
    "experiment_name": ("agent.experiment_name",),
    "action_scale": ("env.action_scale",),
    "device": ("env.sim.device", "agent.device"),
}


def legacy_overrides(args_cli, argv: Sequence[str]) -> dict[str, object]:
    """Map only the flags actually typed; an argparse default never reaches a config."""
    typed = {a.split("=")[0].lstrip("-") for a in argv if a.startswith("--")}
    result = {}
    for flag, paths in LEGACY_FLAGS.items():
        value = getattr(args_cli, flag, None)
        if flag not in typed or value is None:
            continue
        if flag == "seed" and value == -1:
            value = random.randint(0, 10000)
        result.update({path: value for path in paths})
    return result


def _changed(old, new, prefix: str = "") -> dict:
    """Leaves that differ, as ``dotted.path -> value``."""
    if isinstance(new, dict) and isinstance(old, dict):
        result = {}
        for key, value in new.items():
            result.update(_changed(old.get(key), value, f"{prefix}.{key}" if prefix else str(key)))
        return result
    normal = lambda v: list(v) if isinstance(v, tuple) else v  # noqa: E731
    return {} if normal(old) == normal(new) else {prefix: new}


def _nested(path: str, value) -> dict:
    for key in reversed(path.split(".")):
        value = {key: value}
    return value


def parse_cli_overrides(tokens: Sequence[str]) -> dict[str, object]:
    """``env.a.b=value`` / ``agent.a=value`` tokens as ``dotted.path -> value``, values parsed as YAML.

    For scripts that compose configs without Hydra; anything else on the command line is an error.
    """
    result = {}
    for token in tokens:
        path, sep, text = token.partition("=")
        if not sep or not path.startswith(("env.", "agent.")):
            raise ValueError(f"Unrecognized argument {token!r}; overrides look like env.a.b=value or agent.a=value.")
        result[path] = yaml.safe_load(text)
    return result


def apply_cli(env_cfg, agent_cfg, explicit: Mapping, legacy: Mapping) -> None:
    """Apply CLI values, refusing a legacy flag that disagrees with ``env.*``/``agent.*``."""
    for path, value in legacy.items():
        if path in explicit and explicit[path] != value:
            raise ValueError(f"Conflicting values for {path}: {explicit[path]!r} from Hydra, {value!r} from a legacy flag.")
    for path, value in {**legacy, **explicit}.items():
        side, _, rest = path.partition(".")
        if side == "env":
            apply_overrides(env_cfg, _nested(rest, value), derived=DERIVED_ENV_FIELDS)
            env_cfg.composition["cli"].append(f"{path}={value!r}")
        else:
            apply_overrides(agent_cfg, _nested(rest, value))
            agent_cfg.composition["cli"].append(f"{path}={value!r}")


def wirecraft_task_config(task_name: str, agent_entry: str, args_cli, argv: Sequence[str]) -> Callable:
    """``hydra_task_config`` for WireCraft: composes layers, then applies only explicit CLI values."""
    import hydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig, OmegaConf

    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            configure(getattr(args_cli, "mode", None) or "train", getattr(args_cli, "experiment", None) or ())
            env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
            agent_cfg = load_cfg_from_registry(task_name, agent_entry)
            base = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
            ConfigStore.instance().store(name=task_name, node=copy.deepcopy(base))
            sys.argv += ["hydra.output_subdir=null", "hydra.run.dir=."]

            @hydra.main(config_path=None, config_name=task_name, version_base="1.3")
            def hydra_main(merged: DictConfig):
                explicit = _changed(base, OmegaConf.to_container(merged, resolve=True))
                apply_cli(env_cfg, agent_cfg, explicit, legacy_overrides(args_cli, argv))
                func(resolve_env_cfg(env_cfg), resolve_agent_cfg(agent_cfg), *args, **kwargs)

            hydra_main()

        return wrapper

    return decorator


def is_wirecraft(task_name: str | None) -> bool:
    """True for every ID registered through this package, including legacy aliases."""
    import gymnasium as gym

    if not task_name:
        return False
    name = task_name.split(":")[-1]
    return name in gym.registry and "wirecraft" in str(gym.registry[name].entry_point)


def save_resolved(log_dir: str, env_cfg, agent_cfg) -> None:
    """Write the values that really reach the scene and runner, with presets and CLI."""
    from isaaclab.utils.io import dump_yaml

    params = os.path.join(log_dir, "params")
    dump_yaml(os.path.join(params, "env.yaml"), env_cfg)
    selection = {"env": env_cfg.composition}
    if agent_cfg is not None:
        dump_yaml(os.path.join(params, "agent.yaml"), agent_cfg)
        selection["agent"] = agent_cfg.composition
    selection["argv"] = list(sys.orig_argv)
    dump_yaml(os.path.join(params, "selection.yaml"), selection)
