# WireCraft tasks

Single-arm wire tasks for UR5, Franka and the Trossen AI Stationary kit's right arm: connector
insertion, clip routing and laying a wire in a channel, each with an articulated or an FEM wire.

## Layout

```
wirecraft/
  wire_base/            wire backends, connector geometry, WireState; no robot code
  single_arm/
    common/
      env.py            the one DirectRLEnv
      env_cfg.py        inputs and resolve_env_cfg
      compose.py        override layers, CLI handling, saved configs
      registry.py       Gym IDs
      robot.py          robot adapter and differential IK
      tasks/            reward, success, reset per task
      scripted/         scripted policies, through the normal action interface
      presets/          task, wire and mode YAML
      agents/           shared PPO defaults
    ur5/, franka/,
    trossen/            robot asset, joints, gripper, tool frame, cameras
      <task>/           Gym registration, plus task_cfg/ and agents/ for real differences
  dual_arm/             extension point; no dual-arm environment exists yet
```

Dependencies point one way: `ur5`/`franka`/`trossen` -> `common` -> `wire_base` -> `isaaclab`
spawners and `isaaclab_assets`.

## Assets

Official robot assets are loaded where they exist: Franka from Isaac Lab's Nucleus
(`Robots/FrankaEmika/panda_instanceable.usd`) and the Trossen AI Stationary kit from
`TrossenRobotics/trossen_ai_isaac` at a pinned commit (fetched by URL; that repo has no license file, so it is not
copied here). The UR5 with the Robotiq 2F-85 and camera, and the table, have no official copy and ship in
`isaaclab_assets/data/wirecraft/robots/` (see the `NOTICE.md` there). `WIRECRAFT_ROBOT_ASSET_DIR` optionally points
at another folder with the same file names.

Franka and Trossen get a `grasp_center` body after loading: no geometry, no collision, 0.1 g, fixed to the hand
(`robot.tool_parent`, `robot.tool_offset`) midway between the finger pads. `robot.ee_body` tracks it, so the
controller moves the point between the fingers.

## Cameras

`env.enable_cameras=true env.cameras=[side,side2,color]` spawns the two fixed workstation cameras and the
robot's wrist camera `color`. The Trossen one is the D405 on the right arm. The UR5 wrist camera is the RSD455
in its USD. The Franka one copies the wrist camera of Isaac Lab's Franka visuomotor stack task (`panda_hand`,
pos `(0.13, 0, -0.15)`, ROS convention, 24 mm focal length), so images match data collected there.

## Gym IDs

```
WireCraft-{UR5,Franka,Trossen}-{Articulation,Deformable}-<Connector>-Insertion-v0
WireCraft-{UR5,Franka,Trossen}-{Articulation,Deformable}-StackedCuboid-{1,2,3}Clip-Routing-v0
WireCraft-{UR5,Franka,Trossen}-{Articulation,Deformable}-USBA-1Clip-Routing-v0
WireCraft-{UR5,Franka,Trossen}-{Articulation,Deformable}-{StackedCuboid,USBA}-{Straight,Curved}Channel-v0
```

`<Connector>` is `Cylinder`, `Cuboid`, `StackedCuboid`, `DisplayPortStyle`, `Ethernet`,
`DisplayPort`, `HDMI` or `USBA`.

## Configuration

Lowest to highest priority:

| env | agent |
| --- | --- |
| `SingleArmWireEnvCfg` defaults | `WireCraftPPORunnerCfg` defaults |
| robot (`<robot>/robot_cfg.py`) | task (`<robot>/<task>/agents/rsl_rl_ppo.yaml`) |
| task (`presets/task`, then `<robot>/<task>/task_cfg/task_*.yaml`) | robot/wire (`.../agents/rsl_rl_ppo_<wire>.yaml`) |
| wire (`presets/wire`) | |
| connector (`presets/connector/<family>.yaml`, optional) | |
| mode: `train`, `random`, `scripted`, `teleop` (`presets/mode`) | |
| `--experiment file.yaml`, repeatable, in the order given | same files, `agent:` section |
| `env.*` on the command line | `agent.*` on the command line |

Experiment files hold an `env:` and an `agent:` section; neither touches the other side.

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --headless \
    --task WireCraft-UR5-Articulation-StackedCuboid-Insertion-v0 \
    --experiment base.yaml --experiment sweep.yaml \
    env.scene.num_envs=256 env.wire.stiffness=8 agent.max_iterations=2000
```

Rules, enforced in `compose.py` and covered by `source/isaaclab_tasks/test/wirecraft`:

- Mappings merge; lists and tuples are replaced whole. A tuple keeps its length.
- An unknown key raises `KeyError`. A wrong type raises `ValueError`. An int is
  accepted for a float, never the reverse; a bool is never a number.
- A preset may set only the top-level keys listed in `ENV_LAYER_KEYS`.
- Derived fields (`wire_spawn`, `table_cfg`, spaces and so on) can't be overridden.
- `--num_envs`, `--seed`, `--max_iterations`, `--resume`, `--device` and similar flags apply
  only when typed. A flag that disagrees with `env.*` or `agent.*` on the same field is an error.
- `train.py` and `play.py` no longer force terminations or copy `agent.seed` into
  `env.seed` for these tasks; the `train` mode preset sets the terminations.

`resolve_env_cfg` runs once after all layers and builds every spawn config, so nothing
applied late can leave a stale copy. Runs save `params/env.yaml`, `params/agent.yaml`
and `params/selection.yaml` (layers, CLI values, argv).

## Running and checking

One command selects any combination:

```bash
python scripts/wirecraft/run.py --headless --robot franka --job clip_routing --wire deformable \
    --connector stacked_cuboid --variant 1clip --mode scripted --seed 7 --num_envs 5 --output out/
```

In scripted mode each env is one rollout from its own reset. A rollout the simulator breaks
(NaN wire) is caught by the env's fault guard, dropped and rerun, as in training; `result.json`
lists it under `dropped_rollouts`. `verify_matrix.py` runs every ID in the Gym registry and
`build_report.py` turns its folder into a clickable page.

Scripted policies cover Insertion with every connector, and 1-clip routing and straight channels
with the stacked cuboid: those two tasks hinge on the wire rather than the port, so both wires are
checked with one port. 2/3-clip routing, curved channels and the other connectors of routing and
channels run in random mode; the report lists them, and their scripted policies are open for
contributions (`--all_scripted` runs them).

A trained policy is scored with:

```bash
python scripts/wirecraft/eval_checkpoint.py --headless --task WireCraft-UR5-Articulation-StackedCuboid-Insertion-v0 \
    --checkpoint model.pt --num_envs 100 --seed 42 --output eval.json --video env0.mp4
```

Each env plays one deterministic episode from the training reset distribution; the JSON gives the success rate,
how episodes ended (timeout, termination, simulation fault), how often the port was seated at the timeout, how
much of the episode it was seated (the +50 bonus), the median grasp point in the port frame, and per-second task
metrics. `env.*`/`agent.*` overrides and `--experiment` apply as in training.

## Debugging

`scripts/wirecraft/run.py` is the program for every task. Use `--num_envs 1`; `--steps 0` runs one whole scripted rollout.
First breakpoint: `main` in `run.py`. Then:

```
gym.make -> SingleArmWireEnv.__init__ (env.py) -> resolve_env_cfg (env_cfg.py) -> _setup_scene
env.step -> _apply_action -> RobotAdapter.apply_action (robot.py)
         -> _get_dones -> Task.success -> _get_rewards -> Task.rewards -> _reset_idx
policy   -> InsertionPolicy.act (scripted/insertion.py) or WireOpsPolicy.act (scripted/wire_ops.py)
```

Watch `env.wire_state`, `env.arm.ee_pos`, `env.arm.opening`, `env.arm.is_grasping`,
`env.task.tip_depth()`, `env.extras`, `policy.phase`.

| Task | Where to stop first |
| --- | --- |
| insertion | `InsertionTask.success`, `InsertionPolicy.act` |
| clip routing | `ClipRoutingTask.routed`, `RoutingPolicy.move_port` / `place_wire`, `WireHolder.engage` |
| lay in channel | `LayInChannelTask.update`, `ChannelPolicy.park` |
| UR5 | `UR5Adapter.apply_gripper_extras`, `RobotAdapter._jacobian_b` |
| Franka | `RobotAdapter._add_tool_body`, `FrankaAdapter.null_space_step`, `RobotAdapter._slew` |
| Trossen | `RobotAdapter._add_tool_body`, `RobotAdapter.apply_action` (joint-step clamp), `wire_ops.press` |

## Physics shared with the trained policies

The articulation cable is the one the v14.5 insertion policies were trained on: 12 capsules of radius 3 mm and
density 800 kg/m^3, overlapping 8 mm, bending drives of 0.3016 Nm/rad and 0.005 damping limited to 40 rad/s, segment
damping 5/5, contact offset 1 mm, rest offset 0, contact slop 0.0005 and depenetration 0.1 m/s; each port is a
separate 10 g body on a fixed joint, damped 5/0.5, with 15 mm of cable inside its housing so the cable ends at the
tip shoulder (the port origin; `wire.port_cable_embed_depth`). The FEM cable keeps 8 mm. PhysX runs one velocity
iteration. Clip routing and channels use the same cable.
For insertion the cable starts at x 0.1667-0.5667 m, 0.85 m up, and drops onto the table at reset.

## Insertion reward

`InsertionTask.rewards` is the v14.5 staged reward the released UR5 policies were trained on, plus a seated bonus.
Per control step, with `d_ee` the hand's distance to the grasp anchor (`connector.grasp_x` along the port) and
`d_seat` the port root's distance to the seated target (`task.seated_target_offset_x` = 5 mm into the socket frame):

| Term | Shape | Scale |
| --- | --- | --- |
| reach | `(1/(1+50 d_ee^2))^2`, doubled within 2 cm | 15 |
| grasp | closure x `(1 - d_ee/5cm)` inside 5 cm, `-0.2` x closure outside | 25 |
| hand orientation | `abs(q_ee . q_grasp)^4`; `q_grasp` = port orientation x `robot.grasp_tilt` x `robot.grasp_rest_quat` | 10 |
| bring | `1/(1+100 d_seat^2)`, halved when the port is skewed more than about 63 deg | 40 |
| two-point line-up | `1 - tanh(25 (mean + 4.5 max))` of the leading face's and body's off-axis errors, grasped | 20 |
| high-water depth | new forward progress along the bore, at most 2 mm a step, inside a 3.5/5 mm footprint with roll < 5 deg | 6000 |
| push in | `exp(-80 d_seat)` inside a 6/8 mm footprint, grasped | 40 |
| port orientation | `abs(q_port . q_socket)^4`, grasped and within 12 cm | 10 |
| roll | `max(0, y_port . y_socket)^16`, grasped | 10 |
| seated | the port is seated, grasped or not | +50 |
| port on table, empty fist, action^2 | | -5, -5, -0.01 |

"Grasped" is the pose test: within 2 cm of the anchor, orientation similarity > 0.95, and closure past 0.4 rad
of UR5 finger travel (`task.gripper_reward_travel` maps every robot's closure onto it). "Seated" is a pose test on the procedural geometry: every vertex of the leading face is `task.seat_depth`
(10 mm, capped 2 mm short of a shallow socket or a short tip) past the mouth and before the back, inside the bore
cross-section, with the port pointing in.

**Success** is one rule everywhere (env, scripted check, `eval_checkpoint.py`, logs): the port was seated at some
step of the episode. Grasping or releasing does not matter, and success does not end a training episode.

Logged each step under `reward/` (each term as it enters the reward, already scaled, plus `penalties` and
`total`) and `metrics/`: `fault_rate`, `seated_rate`, `grasped_rate`, `hand_to_grasp_point_m`, `port_to_seat_m` and
`best_insert_depth_m`. Success is counted per episode: `metrics/episodes_ended` and `metrics/episodes_succeeded`
are the episodes that ended in the iteration, and `metrics/success_rate` is their ratio, left out of an iteration in
which no episode ended (`WireCraftOnPolicyRunner` in `single_arm/common/rsl_runner.py`, used by
`scripts/reinforcement_learning/rsl_rl/train.py` for WireCraft tasks).

Resets follow the same runs, in train, random and scripted mode alike (`presets/mode/<mode>_insertion.yaml`):
the wire is turned about its centre by up to 18 deg of roll and pitch and 36 deg of yaw, raised 6-11 cm and
dropped, and shifted up to 5 cm in X and 15 cm in Y; the socket moves up to 14 cm in Y and 7.5 cm in Z. The
scripted policy also steps at the RL action scale (0.01). It keeps only what it can't share: no terminations and a
60 s episode, since a scripted insertion takes about 30 s.

The observation (42 values on a 6-joint arm) and the 6-DoF delta action match those runs, so their checkpoints
load. The FEM wire has no filtered cable contacts in PhysX, so its port share is always 100 %.

## Task notes

- Insertion, scripted: a close that ends without a grasp opens and reaches again, and a port that falls 4 mm behind
  the insertion reference backs out to align again, each up to twice; `result.json` counts both per rollout
  (`regrasps`, `reinserts`). Of the two opposite pinches the policy takes the one whose turn to the socket ends
  within 90 deg of the rest heading.
- Lay in channel: the far port starts just outside the chain's entry with the wire swung off the slot, and
  only the other port is picked. The deformable far port is held at its reset position and turns freely; a kinematic
  port would leave the wire's attachment behind on the GPU. The articulation's far port is held by a mass-scaled
  spring, since a rigid clamp and a pinch on the other end diverge the chain.
- Lay in channel, scripted: the picked port swings over the chain on an arc about the far port, a little inside the
  wire's reach (a chain can't stretch), is set down beyond the chain's end and let go. A pinch that closes on no
  port reopens and reaches again, up to twice. The held far port sits 17 mm above the table for the articulation
  cable (20 mm for the FEM one): higher, the soft cable's first stretch stays above the wall tops. Pressing then only visits
  stations while the in-channel fraction is below the success line plus a margin; pressing a seated chain diverges it.
- Simulation faults: an env whose port turns faster than `fault_port_speed` (1000 rad/s), whose arm joint turns
  faster than `fault_joint_speed` (20 rad/s; gripper joints are left out, an instantly switched jaw and the Robotiq's
  passive fingers snap past it), or that goes non-finite, is terminated and reset, as in training. A diverging
  wire reaches its ports, so the ports and the arm joints are the whole check. Scripted runs drop such rollouts, list
  them, and roll out again. Insertion raises both limits to 1e4 rad/s, the threshold its training runs reset at.
- Arm commands: the differential IK runs on every physics substep, but adds its step to the joint angles read at
  the start of the control step (`robot.ik_joint_reference: control`), so one control step moves the tool by about
  one action whatever the decimation. This is the controller the released policies were trained with; `live`
  re-reads the joints each substep and moves about twice as far. Franka and Trossen also clip each substep's joint
  target change to `robot.max_joint_step` (0.05 rad): near a singularity the IK asks for far more and whips the arm.
  The UR5 leaves it unclipped and switches its jaw target at once, as its policies were trained.
- Clip routing, scripted, in two stages. (1) The port is picked mid-housing, carried over the clip along its slot,
  a little past and back to let the cable's stress out, and set down beyond it, so the cable behind it lies across the
  clip; a port that leaves the fingers is picked up again (two port attempts). (2) The wire itself is picked 2 cm past
  the clip, lifted clear of the lips, centred on the slot, drawn down into it while still held and let go; a wire
  that is not seated is picked again, up to four times. There is no pressing with the empty gripper. Success also
  requires the wire to stay seated for 10 s after the arm lets go.
- Wire holder (`single_arm/common/wire_holder.py`, on for clip routing): an invisible closed ring ("O", a U-trough
  closed on top; 14 mm bore, 20 mm long) that collides with the wire only. When the jaws close and come to rest next
  to the wire (`wire_holder.trigger_distance`, 10 mm) away from a port, it is placed around the nearest point of the
  wire and then follows the tool at that offset, steered by velocity each physics step so the solver drags the wire
  with it; opening the jaws puts it away under the table. It keeps a pinched wire from sliding out between small
  pads (Franka, FEM cable) and carries the wire where the fingertips cannot reach it: Trossen's stop about 9 mm above
  the table, above a 6 mm wire lying on it. Because it is part of the env, learned policies get the same help.
- Clip routing grippers: parallel grippers close to `robot.wire_gripper_closed_pos` (2.5 mm per finger, a 5 mm gap
  on the 6 mm wire) rather than shut (`gripper_close_on_wire`), so the FEM cable is held, not crushed flat. The
  Robotiq's soft drive needs no limit. The wire pinch height is per robot (`robot.wire_grasp_z_offset`).

## Trossen

The kit is one articulation: a tabletop, a frame and two arms facing each other. It is turned -90 deg about Z
so the right arm faces +X like the other robots, with its base at x = 0.10; the left arm holds its rest pose at
the far end. Its jaws pair through a PhysX mimic joint, so one joint is driven. The arm reaches about half as
far as a UR5, so each Trossen task folder's `task_cfg/` scales the UR5 layout by 0.55 toward the arm (wire
0.22 m, same 12 articulation segments, channel modules, clip and socket positions, random ranges, safe zone).
Connector, clip-slot and channel cross-sections keep their sizes. What does not scale:

- The safe zone's x minimum is 0.3: the home pose holds the grasp point at x = 0.35.
- Clip routing keeps the UR5's 5 cm from the port to the first clip, so the 6 cm jaws don't land on the clip.
- `robot.press_z_offset` (4 mm) raises every press of the closed jaws, which are wide plates; at the UR5 depth they
  pin the short chain against the table and clip and it diverges.
- Insertion keeps the socket 9 cm past the port (x = 0.575, not 0.5485): with the 66 mm the scale left, a jaw
  closing beside a port next to a low, offset socket landed on the full-size socket.
- `robot.fine_tracking_step` (2 mm a step, against 6 mm elsewhere) runs the insertion policy's reach and
  placement: coarser, the arm overshoots, swinging the port across the socket mouth or folding the wrist onto
  its limit above the port. The lift keeps 6 mm, or the arm can't raise the wire's load to the target.
- Lay in channel: the ports keep 8 mm of cable in the housing (`wire.port_cable_embed_depth`) and the far port sits
  15 mm up. With the cable filling the housing the wide plate jaws missed the port in 3 of 5 rollouts, and from
  17 mm the short chain's first points stayed above the walls.
- Lay in channel, deformable, scripted: `task.scripted_swing` carries the port at the wire's length, 0.10 m up,
  instead of the default draw 4 cm past it at 0.20 m, which the arm can't pull this wire to.

## Known limits

- Curved channel, scripted: a released wire springs back toward the chord and rests on the wall tops. The slot leaves
  about 1 mm of side clearance around the 8 mm wire, and one gripper can't hold the arc and press at once.
- Articulation wires can still diverge under contact; the fault guard resets that env (see Task notes). With
  `wire.solver_position_iterations` below 64 a held chain buzzes between the fingers and the far hold and diverges
  far more often, worst on Franka.
