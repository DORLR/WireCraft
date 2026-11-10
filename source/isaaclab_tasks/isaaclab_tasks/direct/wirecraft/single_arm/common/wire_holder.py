# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Invisible ring that keeps a wire pinched between the fingers.

Two fingers closing on a 6 mm wire hold it badly: the wire rolls out between the
pads as soon as it is pulled. When the gripper has closed on the wire, a small
closed ring (an "O": four walls around the wire, open along it) is placed around
the wire at the pinch and then follows the tool rigidly; opening the gripper puts
it away again. The ring collides with the wire only (pairs with the robot and every
other rigid body in the env are filtered) and is never rendered, so it changes
nothing but how firmly a closed gripper holds the wire. It is part of the env, so
scripted and learned policies see the same physics.

Enabled per task with ``wire_holder.enabled`` (on for clip routing).
"""

from __future__ import annotations

import torch
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import schemas
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat, normalize, quat_apply, quat_from_matrix, quat_inv, quat_mul

from .env_cfg import PRIM_PATH, WireHolderCfg

IDLE_HEIGHT = 0.3
"""Ring height above the env origin while unused: inside the (kinematic) table, far from the wire."""


@clone
def spawn_wire_holder(prim_path: str, cfg: WireHolderSpawnCfg, translation=None, orientation=None) -> Usd.Prim:
    """Four boxes around the local X axis, one rigid body: the wire runs along X through the middle."""
    stage = get_current_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    if translation is not None:
        xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    if orientation is not None:
        xform.AddOrientOp().Set(Gf.Quatf(*orientation))
    inner, wall, length = cfg.inner_size, cfg.wall_thickness, cfg.length
    middle, outer = (inner + wall) / 2.0, inner + 2.0 * wall
    # The wire hangs on the bottom wall and the pull of the cable presses it there: that wall is the thickest,
    # since a deformable mesh pushed past a box's mid plane is pushed out on the far side.
    floor = cfg.floor_thickness
    walls = {
        "bottom": ((0.0, 0.0, -(inner + floor) / 2.0), (length, inner + 2.0 * wall, floor)),
        "top": ((0.0, 0.0, middle), (length, outer, wall)),
        "left": ((0.0, -middle, 0.0), (length, wall, inner)),
        "right": ((0.0, middle, 0.0), (length, wall, inner)),
    }
    for name, (position, size) in walls.items():
        cube = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*position))
        cube.AddScaleOp().Set(Gf.Vec3d(*size))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        if cfg.collision_props is not None:
            schemas.define_collision_properties(cube.GetPrim().GetPath().pathString, cfg.collision_props)
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props)
    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props)
    if not cfg.visible:
        UsdGeom.Imageable(xform.GetPrim()).MakeInvisible()
    return xform.GetPrim()


@configclass
class WireHolderSpawnCfg(RigidObjectSpawnerCfg):
    func: callable = spawn_wire_holder
    inner_size: float = 0.010
    wall_thickness: float = 0.003
    floor_thickness: float = 0.003
    length: float = 0.012
    visible: bool = False


class WireHolder:
    def __init__(self, env):
        self.env = env
        self.cfg: WireHolderCfg = env.cfg.wire_holder

    # ---- scene -------------------------------------------------------------
    def spawn(self) -> None:
        env, cfg = self.env, self.cfg
        radius = env.cfg.wire.radius if env.cfg.wire.kind == "articulation" else env.cfg.wire.fem_radius
        self.inner_half = radius + cfg.clearance
        self.body = RigidObject(
            RigidObjectCfg(
                prim_path=f"{PRIM_PATH}/wire_holder",
                spawn=WireHolderSpawnCfg(
                    inner_size=2.0 * (radius + cfg.clearance),
                    wall_thickness=cfg.wall_thickness,
                    floor_thickness=cfg.floor_thickness,
                    length=cfg.length,
                    visible=cfg.visible,
                    # Dynamic, weightless and heavy, steered by velocity every physics step (GPU PhysX can't set
                    # kinematic targets): the solver then knows how fast the ring moves and drags the wire with it.
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=True, linear_damping=0.0, angular_damping=0.0, max_depenetration_velocity=1.0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    # The wire's own contact settings, so the ring neither pushes it from afar nor lets it sink in.
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, IDLE_HEIGHT)),
            )
        )
        env.scene.rigid_objects["wire_holder"] = self.body

    def post_clone(self) -> None:
        """Filter the ring against every rigid body of its env except the cable links."""
        stage = get_current_stage()
        for env_id in range(self.env.scene.cfg.num_envs):
            root = f"/World/envs/env_{env_id}"
            holder = stage.GetPrimAtPath(f"{root}/wire_holder")
            targets = []
            for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
                path = prim.GetPath().pathString
                if path.startswith(holder.GetPath().pathString) or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                if prim.GetName().startswith("seg_"):
                    continue  # articulated cable links: the ring is there to hold them
                targets.append(prim.GetPath())
            UsdPhysics.FilteredPairsAPI.Apply(holder).CreateFilteredPairsRel().SetTargets(targets)

    # ---- runtime -----------------------------------------------------------
    def initialize(self) -> None:
        count, device = self.env.num_envs, self.env.device
        self.engaged = torch.zeros(count, dtype=torch.bool, device=device)
        self.counter = torch.zeros(count, dtype=torch.long, device=device)
        self.local_pos = torch.zeros(count, 3, device=device)
        self.local_quat = torch.zeros(count, 4, device=device)
        self.local_quat[:, 0] = 1.0
        self.engagements = torch.zeros(count, dtype=torch.long, device=device)
        """Times the ring was placed in each env since its last reset."""
        self.last_opening = self.env.arm.opening.clone()
        self.armed = torch.ones(count, dtype=torch.bool, device=device)
        self.holding = torch.zeros(count, dtype=torch.bool, device=device)
        """Engaged and the wire is actually inside the ring."""
        self.jump = torch.ones(count, dtype=torch.bool, device=device)
        """The ring is placed or put away this step: teleport it instead of moving it to a kinematic target."""
        """The jaws have opened since they last came to rest, so their next close may place the ring."""
        self.track()

    def engage(self, actions: torch.Tensor) -> None:
        """Place or put away the ring from this step's gripper command; call once per control step."""
        env, cfg, arm = self.env, self.cfg, self.env.arm
        closing = actions[:, -1] >= 0.0
        points = env.wire_state.centerline
        nearest = torch.linalg.vector_norm(points - arm.ee_pos.unsqueeze(1), dim=-1).argmin(dim=1)
        near = self._distance_to_wire(points, arm.ee_pos) <= cfg.trigger_distance
        # The fingers must have come to rest on the wire: a ring placed while they still move would hold
        # the wire where the closing fingers push it, and the two would fight.
        still = (arm.opening - self.last_opening).abs() < cfg.trigger_opening_rate
        self.last_opening = arm.opening.clone()
        # One decision per close: where the jaws come to rest. Jaws that closed in the air and then push on the
        # wire (a press with the empty gripper) don't place the ring; they must open again first.
        self.armed |= arm.opening > cfg.trigger_opening
        ready = closing & still & (arm.opening <= cfg.trigger_opening) & self.armed & ~self.engaged
        self.counter = torch.where(ready, self.counter + 1, torch.zeros_like(self.counter))
        rested = self.counter >= cfg.sustain_steps
        # Jaws resting on a wire stop at the wire's width; jaws that shut all the way hold nothing.
        low, high = env.cfg.robot.wire_pinch_opening
        # Jaws on a port hold the port, not the wire leaving it.
        state = env.wire_state
        clear = ((arm.ee_pos - state.insertion_end_pos).norm(dim=-1) > cfg.port_clearance) & (
            (arm.ee_pos - state.far_end_pos).norm(dim=-1) > cfg.port_clearance
        )
        newly = rested & near & clear & (arm.opening > low) & (arm.opening < high)
        self.armed &= ~rested
        if newly.any():
            rows = torch.arange(points.shape[0], device=points.device)
            last = points.shape[1] - 1
            ahead, behind = points[rows, (nearest + 1).clamp(max=last)], points[rows, (nearest - 1).clamp(min=0)]
            frame = self._wire_frame(ahead - behind)
            # The ring is centred on the wire where it lies now that the jaws are at rest: between the pads when
            # they hold it, and below them when they shut above it (Trossen's fingertips stop above a 6 mm wire
            # on the table). It then moves with the tool.
            point = self._closest_on_wire(points, arm.ee_pos)
            offset = quat_apply(quat_inv(arm.ee_quat), point - arm.ee_pos)
            local_quat = quat_mul(quat_inv(arm.ee_quat), frame)
            self.local_pos = torch.where(newly.unsqueeze(-1), offset, self.local_pos)
            self.local_quat = torch.where(newly.unsqueeze(-1), local_quat, self.local_quat)
            self.engagements += newly.long()
        before = self.engaged.clone()
        self.engaged |= newly
        self.counter = torch.where(newly, torch.zeros_like(self.counter), self.counter)
        # Opening the gripper lets go of the wire at once.
        self.engaged &= closing
        self.jump |= self.engaged != before
        # Whether the wire is really in the ring, measured from where the ring is now.
        inside = self._distance_to_wire(points, self.body.data.root_pos_w) < self.inner_half + 0.002
        self.holding = self.engaged & ~self.jump & inside | (newly & near)

    @staticmethod
    def _closest_on_wire(points: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """Point of the centerline polyline closest to ``position``."""
        start, segment = points[:, :-1], points[:, 1:] - points[:, :-1]
        share = (((position.unsqueeze(1) - start) * segment).sum(-1) / (segment * segment).sum(-1).clamp_min(1e-12))
        closest = start + share.clamp(0.0, 1.0).unsqueeze(-1) * segment
        index = (closest - position.unsqueeze(1)).norm(dim=-1).argmin(dim=1)
        return closest[torch.arange(points.shape[0], device=points.device), index]

    @staticmethod
    def _distance_to_wire(points: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """Distance from ``position`` to the centerline polyline (its segments, not just its sample points)."""
        start, end = points[:, :-1], points[:, 1:]
        segment = end - start
        share = (((position.unsqueeze(1) - start) * segment).sum(-1) / (segment * segment).sum(-1).clamp_min(1e-12))
        closest = start + share.clamp(0.0, 1.0).unsqueeze(-1) * segment
        return (closest - position.unsqueeze(1)).norm(dim=-1).amin(dim=1)

    @staticmethod
    def _wire_frame(tangent: torch.Tensor) -> torch.Tensor:
        """World quaternion with +X along ``tangent`` and +Z as close to up as the tangent allows."""
        x = normalize(tangent)
        up = torch.zeros_like(x)
        up[:, 2] = 1.0
        y = torch.linalg.cross(up, x)
        # A vertical tangent has no "up" side; any perpendicular will do.
        flat = torch.linalg.vector_norm(y, dim=-1, keepdim=True) < 1e-6
        side = torch.zeros_like(x)
        side[:, 1] = 1.0
        y = normalize(torch.where(flat, torch.linalg.cross(side, x), y))
        z = torch.linalg.cross(x, y)
        return quat_from_matrix(torch.stack((x, y, z), dim=-1))

    def pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Ring pose in the world from the current tool pose; parked under the table when not engaged."""
        env, arm = self.env, self.env.arm
        data = arm.robot.data
        ee_quat = data.body_quat_w[:, arm.ee_id]
        ee_pos = data.body_pos_w[:, arm.ee_id] + quat_apply(ee_quat, arm.ee_offset)
        position = ee_pos + quat_apply(ee_quat, self.local_pos)
        quat = quat_mul(ee_quat, self.local_quat)
        idle = env.scene.env_origins.clone()
        idle[:, 2] += IDLE_HEIGHT
        identity = torch.zeros_like(quat)
        identity[:, 0] = 1.0
        held = self.engaged.unsqueeze(-1)
        return torch.where(held, position, idle), torch.where(held, quat, identity)

    def track(self) -> None:
        """Move the ring with the tool; call every physics step.

        The ring is given the velocity that brings it onto the tool's pose within one physics step, so the
        solver knows how fast it moves and pushes the wire along with it; a teleported ring would only push wire
        nodes out of its walls, either way. It is teleported only when it is placed or put away.
        """
        position, quat = self.pose()
        if self.jump.any():
            ids = self.jump.nonzero().flatten()
            self.body.write_root_pose_to_sim(torch.cat((position, quat), dim=-1)[ids], env_ids=ids)
            self.jump[:] = False
        data, dt = self.body.data, self.env.physics_dt
        linear = (position - data.root_pos_w) / dt
        angular = axis_angle_from_quat(quat_mul(quat, quat_inv(data.root_quat_w))) / dt
        self.body.write_root_velocity_to_sim(torch.cat((linear, angular), dim=-1))

    def reset(self, env_ids: torch.Tensor) -> None:
        self.engaged[env_ids] = False
        self.counter[env_ids] = 0
        self.engagements[env_ids] = 0
        self.armed[env_ids] = True
        self.holding[env_ids] = False
        self.jump[env_ids] = True
        self.track()
