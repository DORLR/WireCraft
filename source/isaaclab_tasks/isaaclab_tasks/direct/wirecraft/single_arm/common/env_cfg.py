# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Inputs for the single-arm wire environment and the one resolve step.

Everything a user may override is a plain input on these configclasses.
Spawn configs, spaces and derived sizes are produced only by
:func:`resolve_env_cfg`, after every override layer has been merged, so a
late override cannot leave a stale copy behind.
"""

from __future__ import annotations

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_assets import WIRECRAFT_ASSET_DIR

from ...wire_base import ConnectorCfg, WireSceneCfg
from ...wire_base.connectors import normalize_family

PRIM_PATH = "/World/envs/env_.*"
MODES = ("train", "random", "scripted", "teleop")
ASSET_DIR_ENV = "WIRECRAFT_ROBOT_ASSET_DIR"


def robot_asset(name: str) -> str:
    """A robot/table USD from ``$WIRECRAFT_ROBOT_ASSET_DIR`` if set, else the copy packaged with isaaclab_assets."""
    root = os.environ.get(ASSET_DIR_ENV) or os.path.join(WIRECRAFT_ASSET_DIR, "robots")
    path = os.path.join(root, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist; check {ASSET_DIR_ENV}.")
    return path


@configclass
class CameraCfg:
    """Pinhole intrinsics shared by the two workstation cameras."""

    focal_length: float = 24.0
    focus_distance: float = 400.0
    horizontal_aperture: float = 20.955
    clipping_range: tuple[float, float] = (0.1, 2.0)


@configclass
class RobotCfg:
    """Everything robot-specific; shared tasks only read the adapter built from it."""

    name: str = MISSING
    adapter: str = MISSING
    """``module:Class`` of the robot adapter."""
    usd_path: str = ""
    """Empty uses the robot's own default asset."""
    articulation: ArticulationCfg = MISSING
    arm_joints: list[str] = MISSING
    gripper_joint: str = MISSING
    ee_body: str = MISSING
    ee_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Grasp point in the ``ee_body`` frame."""
    tool_parent: str = ""
    """Link to hang a ``grasp_center`` body on after the asset loads, for assets without one.
    Set ``ee_body`` to ``grasp_center`` to track it."""
    tool_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """``grasp_center`` in the ``tool_parent`` frame: midway between the finger pads."""
    gripper_open_pos: float = MISSING
    gripper_closed_pos: float = MISSING
    gripper_max_step: float | None = None
    """Largest closing change of the finger target per physics step; None commands it directly."""
    gripper_release_step: float | None = None
    """Same limit while opening. Letting go of a loaded wire at once can diverge its solve."""
    grasp_opening_range: tuple[float, float] = MISSING
    """Normalized opening (1 = fully open) in which a port counts as held."""
    release_opening: float = MISSING
    """Normalized opening above which the gripper counts as released."""
    fully_closed_opening: float = MISSING
    finger_bodies: tuple[str, str] = MISSING
    """Regexes of the left and right finger bodies under the robot prim."""
    pad_static_friction: float = 5.0
    pad_dynamic_friction: float = 5.0
    pad_friction_combine_mode: str = "max"
    grasp_rest_quat: tuple[float, float, float, float] = MISSING
    """World orientation of the tool for a top-down pinch across the port's Y axis."""
    grasp_z_offset: float = 0.0
    """Height of the grasp point above the port origin."""
    grasp_tilt: float = 0.0
    """Pitch about the robot base Y axis of the grasp pose the insertion reward asks for, radians."""
    wire_pinch_clearance: float = 0.02
    """Clip routing: the wire is pinched this far past the clip's end, so the fingers stay clear of the clip."""
    wire_gripper_closed_pos: float | None = None
    """Gripper target when ``gripper_close_on_wire`` is set: closed just short of the wire's width, so a
    deformable wire is held, not crushed flat. ``None`` keeps ``gripper_closed_pos`` (a soft drive needs no limit)."""
    wire_pinch_opening: tuple[float, float] = (0.03, 0.5)
    """Gripper opening (0 closed, 1 open) of jaws resting on the wire: below it they shut on nothing."""
    wire_grasp_z_offset: float = 0.0
    """Tool point above the wire's centre for a top-down pinch on the cable itself."""
    press_z_offset: float = 0.0
    """Raises every press target of the closed, empty gripper, for jaws that reach further below the grasp point."""
    ik_tracking_step: float = 0.0015
    """Largest per-control translation the scripted controller commands, meters."""
    fine_tracking_step: float | None = None
    """Same limit while the insertion policy reaches for the port and places it, to a millimetre; None keeps ``ik_tracking_step``."""
    ik_joint_reference: str = "control"
    """Joint angles the IK adds its step to on each physics substep: ``control`` keeps the angles read at the
    start of the control step (the controller the released policies were trained with, so a control step
    moves the tool by about one action whatever the decimation); ``live`` re-reads them every substep."""
    ft_sensor_body: str | None = None
    """Body whose incoming joint wrench is read as a wrist force/torque sensor; None disables it. On the UR5,
    ``gripper_base_link`` gives the wrench the flange passes to the gripper, like an F/T sensor between them."""
    max_joint_step: float | None = 0.05
    """Largest arm joint target change per physics substep, rad. Near a singularity the IK asks for far more,
    which whips the arm. None leaves the IK step unclipped."""
    joint_lower: list[float] | None = None
    joint_upper: list[float] | None = None
    """Arm limits; None reads the soft limits from the articulation."""


@configclass
class WireHolderCfg:
    enabled: bool = False
    clearance: float = 0.004
    """Gap between a centred wire and each inner wall, meters: room for the wire to sit off the pinch centre."""
    wall_thickness: float = 0.008
    """Thick walls: a deformable wire pressed against a thin wall can be pushed through it."""
    floor_thickness: float = 0.02
    """Bottom wall, the one the hanging wire is pulled against, meters."""
    length: float = 0.02
    """Ring length along the wire, meters."""
    trigger_distance: float = 0.010
    """The wire centerline must pass this close to the tool point for the ring to be placed, meters."""
    trigger_opening: float = 0.5
    """Gripper opening (0 closed, 1 open) below which the jaws count as closed on the wire."""
    port_clearance: float = 0.025
    """No ring while the tool point is this close to a port's root: the jaws hold the port, meters."""
    trigger_opening_rate: float = 0.002
    """Largest change of the opening per control step for the jaws to count as at rest."""
    sustain_steps: int = 10
    """Control steps the close command, the closed jaws and the nearby wire must last together."""
    visible: bool = False
    """Render the ring, for debugging only."""


@configclass
class SingleArmWireEnvCfg(DirectRLEnvCfg):
    """Shared workstation inputs. Robot, task, wire and connector compose below."""

    episode_length_s: float = 5.0
    decimation: int = 2
    action_space: int = 7
    observation_space: int = 0
    state_space: int = 0
    seed: int = 42

    mode: str = "train"
    variant: str = ""
    """Registered variant name, e.g. ``1clip`` or ``curved``."""

    robot: RobotCfg = MISSING
    task: object = MISSING
    wire: WireSceneCfg = WireSceneCfg()
    gripper_close_on_wire: bool = False
    """Close the gripper to ``robot.wire_gripper_closed_pos`` instead of all the way (tasks that pinch the wire)."""
    wire_holder: WireHolderCfg = WireHolderCfg()
    """Invisible ring that keeps a pinched wire between the fingers; see ``wire_holder.py``."""
    connector: ConnectorCfg = ConnectorCfg()

    control_mode: str = "cartesian"
    """``cartesian`` (delta pose in the robot base frame) or ``joint``."""
    fix_ee_orient: bool = False
    action_scale: float = 1.0
    dof_velocity_scale: float = 0.1

    enable_terminations: bool = False
    terminate_on_time_out: bool = False
    terminate_on_success: bool = True
    ee_safe_zone: list[float] = [0.4, 0.8, -0.35, 0.35, 0.75, 1.2]
    out_of_bound_penalty: float = -50.0
    joint_limit_penalty: float = -50.0
    male_port_drop_penalty: float = -50.0
    port_drop_height: float = 0.77
    fault_port_speed: float = 1000.0
    """Port angular speed in rad/s beyond which the simulation has lost the env. A held port
    already turns at 10-20 rad/s, and early contact spikes reach a few hundred."""
    fault_joint_speed: float = 20.0
    """Arm joint speed in rad/s beyond which the simulation has lost the env; both arms are limited to about pi."""

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            gpu_max_rigid_contact_count=1048576,
            gpu_max_rigid_patch_count=1048576,
            gpu_found_lost_pairs_capacity=1048576,
            gpu_found_lost_aggregate_pairs_capacity=1048576,
            gpu_total_aggregate_pairs_capacity=1048576,
            gpu_max_soft_body_contacts=1048576,
            gpu_max_particle_contacts=1048576,
            gpu_heap_capacity=134217728,
            gpu_temp_buffer_capacity=67108864,
            gpu_max_num_partitions=8,
            # One velocity iteration, as the v14.5 insertion runs used; it also caps every actor's own count.
            max_velocity_iteration_count=1,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=3.0, replicate_physics=False, clone_in_fabric=False
    )

    table_usd: str = ""
    """Empty resolves ``table.usd`` in the robot asset directory."""
    table_pos: tuple[float, float, float] = (0.0, 0.0, 0.795)
    table_top_z: float = 0.7825
    """Calibrated top of the table mesh; a 4 mm clip base sank at 0.780 and floated at 0.785."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    enable_cameras: bool = False
    cameras: list[str] = []
    """Any of ``side``, ``side2`` and the robot's wrist cameras."""
    camera_resolution: int = 224
    render_stride: int = 1
    """Control steps per rendered frame; raise it to record long rollouts cheaply."""
    camera: CameraCfg = CameraCfg()
    side_camera_pose: tuple[float, ...] = (0.7, 0.7, 1.4, 0.56, 0.12, 0.25, -0.76)
    side2_camera_pose: tuple[float, ...] = (0.0, 0.6, 1.2, 0.9183, 0.0705, 0.1906, -0.3397)
    """World position then wxyz rotation; identical for every robot."""

    # Derived by resolve_env_cfg. Never set these from a preset or the CLI.
    resolved: bool = False
    wire_prim_path: str = f"{PRIM_PATH}/wire"
    wire_spawn: object = None
    table_cfg: object = None
    camera_cfgs: dict = {}
    composition: dict = {}
    """Layers and CLI values that produced this config."""


def build_camera(cfg: SingleArmWireEnvCfg, name: str, pose: tuple[float, ...]) -> TiledCameraCfg:
    """World-fixed camera; the same transform and intrinsics for every robot."""
    return TiledCameraCfg(
        prim_path=f"/World/envs/env_.*/Camera_{name}",
        offset=TiledCameraCfg.OffsetCfg(pos=tuple(pose[:3]), rot=tuple(pose[3:]), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=cfg.camera.focal_length,
            focus_distance=cfg.camera.focus_distance,
            horizontal_aperture=cfg.camera.horizontal_aperture,
            clipping_range=tuple(cfg.camera.clipping_range),
        ),
        width=cfg.camera_resolution,
        height=cfg.camera_resolution,
    )


def _import(path: str):
    import importlib

    module, name = path.split(":")
    return getattr(importlib.import_module(module), name)


def resolve_env_cfg(cfg: SingleArmWireEnvCfg) -> SingleArmWireEnvCfg:
    """Validate the merged inputs and derive every spawn config exactly once."""
    if cfg.resolved:
        return cfg
    if cfg.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {cfg.mode!r}.")
    if cfg.control_mode not in ("cartesian", "joint"):
        raise ValueError(f"control_mode must be 'cartesian' or 'joint', got {cfg.control_mode!r}.")
    if cfg.fix_ee_orient and cfg.control_mode != "cartesian":
        raise ValueError("fix_ee_orient requires control_mode='cartesian'.")
    if cfg.decimation < 1 or cfg.episode_length_s <= 0 or cfg.action_scale <= 0:
        raise ValueError("decimation, episode_length_s and action_scale must be positive.")
    if len(cfg.ee_safe_zone) != 6:
        raise ValueError("ee_safe_zone must be [x_min, x_max, y_min, y_max, z_min, z_max].")
    if cfg.scene.num_envs < 1:
        raise ValueError("scene.num_envs must be positive.")
    cfg.connector.family = normalize_family(cfg.connector.family)
    if cfg.wire.kind == "deformable" and "cuda" not in cfg.sim.device:
        raise ValueError("Deformable wires require a CUDA device.")

    cfg.sim.render_interval = cfg.decimation * cfg.render_stride
    # replicate_physics must stay off: every env owns its own reset geometry.
    cfg.scene.replicate_physics = False

    adapter = _import(cfg.robot.adapter)
    cfg.robot.articulation.prim_path = f"{PRIM_PATH}/{adapter.prim_name}"
    cfg.robot.articulation.spawn.usd_path = cfg.robot.usd_path or adapter.resolve_usd_path(cfg.robot)
    arm_dof = len(cfg.robot.arm_joints)
    if cfg.control_mode == "joint":
        cfg.action_space = arm_dof + 1
    else:
        cfg.action_space = 4 if cfg.fix_ee_orient else 7
    cfg.wire_spawn = cfg.wire.assembly_cfg(cfg.connector)
    cfg.task.resolve(cfg)
    cfg.observation_space = cfg.task.observation_dim(arm_dof + 1)

    table = cfg.table_usd or robot_asset("table.usd")
    cfg.table_cfg = RigidObjectCfg(
        prim_path=f"{PRIM_PATH}/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=table,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(cfg.table_pos), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    cfg.camera_cfgs = {}
    if cfg.enable_cameras:
        wrist = adapter.camera_cfgs(cfg)
        fixed = {"side": cfg.side_camera_pose, "side2": cfg.side2_camera_pose}
        for name in cfg.cameras:
            if name in fixed:
                cfg.camera_cfgs[name] = build_camera(cfg, name, fixed[name])
            elif name in wrist:
                cfg.camera_cfgs[name] = wrist[name]
            else:
                raise ValueError(f"Unknown camera {name!r}; choose from {sorted(fixed) + sorted(wrist)}.")
    cfg.resolved = True
    return cfg
