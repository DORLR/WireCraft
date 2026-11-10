# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Layer priority, CLI handling and final spawn values."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import argparse  # noqa: E402

import pytest  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.direct.wirecraft.single_arm.common import compose  # noqa: E402
from isaaclab_tasks.direct.wirecraft.single_arm.common.compose import (  # noqa: E402
    apply_cli,
    apply_env_layer,
    apply_overrides,
    compose_agent_cfg,
    compose_env_cfg,
    legacy_overrides,
    resolve_agent_cfg,
)
from isaaclab_tasks.direct.wirecraft.single_arm.common.env_cfg import resolve_env_cfg  # noqa: E402
from isaaclab_tasks.direct.wirecraft.single_arm.common.registry import SPECS  # noqa: E402

UR5 = SPECS["WireCraft-UR5-Articulation-StackedCuboid-Insertion-v0"]
FRANKA_FEM = SPECS["WireCraft-Franka-Deformable-USBA-1Clip-Routing-v0"]


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_mode_presets_differ():
    train, scripted = compose_env_cfg(UR5, "train"), compose_env_cfg(UR5, "scripted")
    assert train.enable_terminations and train.terminate_on_time_out and train.episode_length_s == 8.0
    routing = compose_env_cfg(SPECS["WireCraft-UR5-Articulation-StackedCuboid-1Clip-Routing-v0"], "train")
    assert routing.fix_ee_orient and not train.fix_ee_orient  # the insertion reward scores the grasp orientation
    assert not scripted.enable_terminations and not scripted.fix_ee_orient and scripted.action_scale == train.action_scale


def test_layer_order_and_wire_preset():
    cfg = compose_env_cfg(FRANKA_FEM, "train")
    assert cfg.robot.name == "franka" and cfg.wire.kind == "deformable"
    assert cfg.task.num_clips == 1 and cfg.connector.family == "usb_a"
    assert cfg.action_scale == 0.01  # wire preset wins over the task preset's 0.5


def test_experiment_order_and_cli_priority(tmp_path):
    first = write(tmp_path, "a.yaml", "env:\n  action_scale: 0.2\n  wire:\n    stiffness: 7\nagent:\n  max_iterations: 11\n")
    second = write(tmp_path, "b.yaml", "env:\n  action_scale: 0.3\n")
    env = compose_env_cfg(UR5, "train", [first, second])
    agent = compose_agent_cfg(UR5, [first, second])
    assert env.action_scale == 0.3 and env.wire.stiffness == 7.0 and agent.max_iterations == 11
    apply_cli(env, agent, {"env.action_scale": 0.4, "agent.max_iterations": 12}, {})
    assert env.action_scale == 0.4 and agent.max_iterations == 12
    assert env.composition["cli"] == ["env.action_scale=0.4"]


def test_sections_stay_separate(tmp_path):
    env_only = write(tmp_path, "a.yaml", "env:\n  action_scale: 0.2\n")
    before = compose_agent_cfg(UR5).to_dict()
    after = compose_agent_cfg(UR5, [env_only]).to_dict()
    before.pop("composition"), after.pop("composition")
    assert before == after
    with pytest.raises(ValueError, match="top-level"):
        compose_env_cfg(UR5, "train", [write(tmp_path, "b.yaml", "action_scale: 0.2\n")])


def test_preset_namespace():
    with pytest.raises(ValueError, match="connector preset"):
        apply_env_layer(compose_env_cfg(UR5), "connector", {"wire": {"radius": 0.1}})


def test_merge_rules():
    cfg = compose_env_cfg(UR5)
    apply_overrides(cfg, {"ee_safe_zone": [0, 1, 2, 3, 4, 5], "wire": {"root_pos": [0, 0, 1]}, "connector": {"socket_clearance": 0.001}})
    assert cfg.ee_safe_zone == [0, 1, 2, 3, 4, 5]  # lists replace whole
    assert cfg.wire.root_pos == (0.0, 0.0, 1.0) and isinstance(cfg.wire.root_pos[0], float)
    assert cfg.connector.socket_clearance == 0.001  # None default takes a value
    with pytest.raises(KeyError, match="wire.nope"):
        apply_overrides(cfg, {"wire": {"nope": 1}})
    for bad in ({"decimation": 1.5}, {"enable_terminations": 1}, {"action_scale": True}, {"wire": {"root_pos": [1, 2]}}, {"wire": 3}):
        with pytest.raises(ValueError):
            apply_overrides(cfg, bad)
    with pytest.raises(ValueError, match="derived"):
        apply_overrides(cfg, {"wire_spawn": 1}, derived=compose.DERIVED_ENV_FIELDS)


def test_legacy_flags_only_when_typed():
    args = argparse.Namespace(num_envs=None, seed=5, max_iterations=None, resume=None, device="cuda:0")
    assert legacy_overrides(args, ["--seed", "5"]) == {"env.seed": 5, "agent.seed": 5}
    assert legacy_overrides(args, []) == {}
    args = argparse.Namespace(resume=True)
    assert legacy_overrides(args, ["--resume"]) == {"agent.resume": True}


def test_legacy_conflict_raises():
    env, agent = compose_env_cfg(UR5), compose_agent_cfg(UR5)
    with pytest.raises(ValueError, match="Conflicting"):
        apply_cli(env, agent, {"env.seed": 1}, {"env.seed": 2, "agent.seed": 2})
    apply_cli(env, agent, {"env.seed": 2}, {"env.seed": 2})
    assert env.seed == 2


def test_agent_resume_survives_absent_flag():
    env, agent = compose_env_cfg(UR5), compose_agent_cfg(UR5)
    apply_cli(env, agent, {"agent.resume": True}, legacy_overrides(argparse.Namespace(resume=None), []))
    assert resolve_agent_cfg(agent).resume is True


def test_resolve_reaches_spawn_after_late_override():
    cfg = compose_env_cfg(UR5, "scripted")
    apply_overrides(cfg, {"wire": {"radius": 0.003, "stiffness": 9, "length": 0.5}, "connector": {"family": "hdmi", "port_mass": 0.02}})
    resolve_env_cfg(cfg)
    wire = cfg.wire_spawn.wire
    assert (wire.radius, wire.stiffness, wire.total_length) == (0.003, 9.0, 0.5)
    assert cfg.wire_spawn.end_port.family == "hdmi" and cfg.wire_spawn.end_port.mass == 0.02
    assert cfg.task.socket_cfg.spawn.family == "hdmi"
    assert cfg.action_space == 7


def test_action_and_observation_follow_robot():
    ur5 = resolve_env_cfg(compose_env_cfg(UR5, "train"))
    franka = resolve_env_cfg(compose_env_cfg(SPECS["WireCraft-Franka-Articulation-StackedCuboid-Insertion-v0"], "train"))
    assert ur5.action_space == franka.action_space == 7
    assert ur5.observation_space == 42  # the v14.5 insertion observation, so trained checkpoints load
    assert franka.observation_space == ur5.observation_space + 2


def test_instances_do_not_share_state():
    first, second = compose_env_cfg(UR5), compose_env_cfg(UR5)
    first.wire.stiffness = 99.0
    first.robot.arm_joints.append("x")
    first.ee_safe_zone[0] = -9.0
    first.robot.articulation.actuators["gripper"].stiffness = 123.0
    assert second.wire.stiffness == pytest.approx(0.3015928947446202) and "x" not in second.robot.arm_joints
    assert second.ee_safe_zone[0] == 0.4 and second.robot.articulation.actuators["gripper"].stiffness == 1.7
    agent = compose_agent_cfg(UR5)
    agent.policy.actor_hidden_dims.append(1)
    assert compose_agent_cfg(UR5).policy.actor_hidden_dims == [256, 128, 64]
    assert compose_agent_cfg(FRANKA_FEM).policy.actor_hidden_dims == [512, 256, 128]


def test_trossen_layout_scales_ur5():
    for task, mode in (("Insertion", "random"), ("1Clip-Routing", "scripted"), ("StraightChannel", "scripted")):
        ur5 = resolve_env_cfg(compose_env_cfg(SPECS[f"WireCraft-UR5-Articulation-StackedCuboid-{task}-v0"], mode))
        trossen = resolve_env_cfg(compose_env_cfg(SPECS[f"WireCraft-Trossen-Articulation-StackedCuboid-{task}-v0"], mode))
        assert trossen.wire.length == pytest.approx(0.55 * ur5.wire.length)
        assert trossen.wire.num_segments == ur5.wire.num_segments
        if task != "1Clip-Routing":  # the port keeps the UR5's 5 cm to the first clip
            assert trossen.wire.root_pos[0] == pytest.approx(0.18 + 0.55 * ur5.wire.root_pos[0])
        assert trossen.wire.randomize_xy_range == pytest.approx(tuple(0.55 * v for v in ur5.wire.randomize_xy_range))
        assert trossen.wire.radius == ur5.wire.radius and trossen.connector == ur5.connector
    channel = trossen.task
    assert channel.module_cfgs[0].spawn.length == pytest.approx(0.55 * 0.110)
    assert channel.start_xy[0] == pytest.approx(0.18 + 0.55 * 0.255)
    assert trossen.action_space == ur5.action_space


def test_scripted_swing_override_is_per_wire():
    def swing(robot, wire):
        spec = SPECS[f"WireCraft-{robot}-{wire}-StackedCuboid-StraightChannel-v0"]
        return resolve_env_cfg(compose_env_cfg(spec, "scripted")).task.scripted_swing

    assert swing("UR5", "Deformable") == {}
    assert swing("Trossen", "Deformable")["deformable"] == {"slack": 0.0, "height": 0.10, "set_slack": 0.0}
    # Trossen's articulation chain carries lower: the UR5's 10 cm height leaves its 0.22 m chain no reach.
    assert swing("Trossen", "Articulation")["articulation"] == {"height": 0.055, "slack": 0.03}
    assert "articulation" not in swing("UR5", "Articulation")


def _seat_env(family: str, depth: float, y: float = 0.0, yaw: float = 0.0):
    """One port posed in front of a socket at the origin, its leading face ``depth`` past the mouth."""
    import math
    from types import SimpleNamespace

    import torch

    from isaaclab_tasks.direct.wirecraft.single_arm.common.tasks.insertion import InsertionCfg, InsertionTask
    from isaaclab_tasks.direct.wirecraft.wire_base.connectors import ConnectorGeometry

    geometry = ConnectorGeometry.from_family(family)
    root = torch.tensor([[geometry.socket_mouth_x + depth - geometry.tip_length, y, 0.0]])
    quat = torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]])
    env = SimpleNamespace(device="cpu", num_envs=1, connector=geometry,
                          wire_state=SimpleNamespace(insertion_end_pos=root, insertion_end_quat=quat))
    task = InsertionTask.__new__(InsertionTask)
    task.env, task.cfg = env, InsertionCfg()
    task.seat_depth = min(task.cfg.seat_depth, geometry.socket_depth - 0.002, geometry.tip_length - 0.002)
    task.face = torch.tensor([(geometry.tip_length, a, b) for a, b in geometry.tip_outline])
    task.hole = torch.tensor(geometry.hole)
    task.x_axis = torch.tensor([[1.0, 0.0, 0.0]])
    task.socket_pos, task.socket_quat = torch.zeros(1, 3), torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    return task


@pytest.mark.parametrize("family", ["cylinder", "cuboid", "stacked_cuboid", "displayport_style",
                                    "ethernet", "usb_a", "displayport", "hdmi"])
def test_seat_needs_the_whole_face_in_the_bore(family):
    seat = _seat_env(family, 0.0).seat_depth
    assert bool(_seat_env(family, seat + 0.0005).is_seated())
    assert not bool(_seat_env(family, seat - 0.0005).is_seated())
    assert not bool(_seat_env(family, seat + 0.0005, y=0.004).is_seated())  # rim, not bore
    assert not bool(_seat_env(family, seat + 0.0005, yaw=3.14159).is_seated())  # facing out


def test_insertion_modes_share_the_training_resets():
    cfgs = {mode: compose_env_cfg(UR5, mode) for mode in ("train", "random", "scripted")}
    for cfg in cfgs.values():
        wire = cfg.wire
        assert wire.randomize_xy and tuple(wire.randomize_xy_range) == (0.05, 0.15)
        assert (wire.randomize_yaw, wire.randomize_tilt, tuple(wire.randomize_lift)) == (0.6283, 0.3142, (0.06, 0.11))
        assert cfg.task.randomize_socket and cfg.action_scale == 0.01
    assert not cfgs["scripted"].enable_terminations  # the scripted policy outlasts an 8 s episode
    assert not cfgs["train"].terminate_on_success
    routing = compose_env_cfg(SPECS["WireCraft-UR5-Articulation-StackedCuboid-1Clip-Routing-v0"], "scripted")
    assert routing.wire.randomize_tilt == 0.0 and routing.action_scale == 1.0


def test_articulation_wire_is_the_trained_cable():
    cfg = resolve_env_cfg(compose_env_cfg(SPECS["WireCraft-Franka-Articulation-StackedCuboid-1Clip-Routing-v0"], "scripted"))
    wire = cfg.wire_spawn.wire
    assert (wire.radius, wire.density, wire.damping, wire.capsule_overlap) == (0.003, 800.0, 0.005, 0.008)
    assert wire.stiffness == pytest.approx(0.3015928947446202) and wire.max_depenetration_velocity == 0.1
    assert (wire.linear_damping, wire.angular_damping, wire.joint_velocity_limit) == (5.0, 5.0, 40.0)
    assert (cfg.wire_spawn.end_port.linear_damping, cfg.wire_spawn.end_port.angular_damping) == (5.0, 0.5)
    assert cfg.sim.physx.max_velocity_iteration_count == 1
    fem = resolve_env_cfg(compose_env_cfg(FRANKA_FEM, "scripted"))
    assert fem.wire_spawn.end_port.linear_damping is None  # the FEM ports keep their own settings
    assert (cfg.wire_spawn.end_port.cable_embed_depth, fem.wire_spawn.end_port.cable_embed_depth) == (0.015, 0.008)


def test_wrist_ft_sensor_is_opt_in():
    import os

    assert compose_env_cfg(UR5, "scripted").robot.ft_sensor_body is None
    experiment = os.path.join(os.path.dirname(__file__), "../../../../scripts/wirecraft/experiments/ur5_ft_sensor.yaml")
    cfg = resolve_env_cfg(compose_env_cfg(UR5, "scripted", [os.path.abspath(experiment)]))
    assert cfg.robot.ft_sensor_body == "gripper_base_link"
    assert cfg.observation_space == 42  # the wrench is a separate observation, not part of "policy"


def test_script_defaults_sit_below_experiments(tmp_path):
    """eval/collect defaults apply without an experiment, an experiment beats them, a typed flag beats both."""
    defaults = {"seed": 42, "scene": {"num_envs": 50}}
    cfg = compose_env_cfg(UR5, "train", script_defaults=defaults)
    assert (cfg.seed, cfg.scene.num_envs) == (42, 50)
    experiment = tmp_path / "exp.yaml"
    experiment.write_text("env:\n  seed: 99\n  scene:\n    num_envs: 7\n")
    cfg = compose_env_cfg(UR5, "train", [str(experiment)], script_defaults=defaults)
    assert (cfg.seed, cfg.scene.num_envs) == (99, 7)
    args = argparse.Namespace(num_envs=50, seed=3)
    apply_cli(cfg, compose_agent_cfg(UR5), {}, legacy_overrides(args, ["--seed", "3"]))
    assert (cfg.seed, cfg.scene.num_envs) == (3, 7)
