# WireCraft release notes

## 0.1.0 (first release candidate, 2026-11-09)

### What is in it

- One single-arm `DirectRLEnv` for UR5 (Robotiq 2F-85), Franka and the right arm of the Trossen AI Stationary kit,
  with three tasks (connector insertion, clip routing, laying a wire in a channel) and two cables (articulated
  capsule chain, FEM deformable). Gym IDs and the override order are in `README.md`.
- Eight connector families from one shared profile table: printable cylinder, cuboid, stacked cuboid and
  DisplayPort-style; real Ethernet, USB-A, DisplayPort and HDMI.
- Layered configuration (defaults, robot, task, wire, connector, mode, experiment files, command line) with type
  and key checks; every run saves the resolved env and agent configs.
- Scripted policies through the normal action interface for insertion with every connector, and for 1-clip routing
  and straight channels with the stacked cuboid.
- Entry points: `scripts/reinforcement_learning/rsl_rl/train.py` / `play.py`, `scripts/wirecraft/run.py`
  (random/scripted rollouts), `verify_matrix.py` + `build_report.py` (every registered ID), and
  `eval_checkpoint.py` (score a trained policy, with video).

### Compatible with the v14.5 insertion policies

The insertion task reproduces the environment the released UR5 articulation insertion policies were trained in,
so their checkpoints load and behave as in training:

- Reward: the v14.5 staged reward plus +50 per control step while the port is seated, grasped or not.
  Success: the whole leading face 10 mm into the bore at some step of the episode, grasped or not; it does not
  end training episodes. The released policies were trained with a bonus that also required the jaw closed.
- Observation (42 values on the UR5), 6-DoF delta action at scale 0.01, 8 s episodes, faults reset only past
  1e4 rad/s, and the same reset distribution (cable tilt/yaw/drop/shift, socket shift), shared by train, random
  and scripted mode.
- Controller: the IK step is added to the joint angles read at the start of the control step, the UR5 jaw target
  switches at once and the UR5 IK step is not clipped.
- Cable and port physics: the trained cable recipe (see README, "Physics shared with the trained policies"), also
  used by clip routing and channels; the port origin sits at the cable end (the FEM cable keeps 8 mm in the housing).

What was checked to be the same in both environments:

- Observation: from one canonical reset without randomization, all 42 values match except the port height
  (2 mm, one physics update of settling) and the two values derived from it; the first actions agree within a few
  percent (`evaluations/paired_model8900/jobs/probe_norand2/probe.json` vs `open_probe_geom.json`).
- Normalizer: the checkpoint's empirical observation normalizer is loaded and frozen;
  `eval_checkpoint.py` reports `observation_normalizer.matches_checkpoint` (true for model_8900).
- Contact: the two contact values are each finger pad's normal force on the insertion port only, /10 and
  clipped to [0, 1]. The training runs filtered the policy's pad sensors to `wire/end_port` alone
  (`fixed_port.install`; recorded in the frozen run's `contact_groups.json`); the 13-body port+cable sensors there
  were diagnostic only.
- Robot and table USD files are byte-identical; actuator gains, solver iterations, dt and decimation match.

Paired check with the same checkpoint (`model_8900`, deterministic mean action):

| Environment | Episodes | Success (seated at some step) | Seated at the 8 s timeout | Simulation faults |
| --- | --- | --- | --- | --- |
| Training environment (frozen) | 10 | 9/10 | 5/10 | 0 |
| This release, seed 42 | 100 | 94 % | 49 % | 0 |
| This release, seed 43 | 100 | 83 % | 55 % | 0 |
| Before alignment (old cable, controller and fault limits) | 50 | 0 % | 0 % | 46 |

### Known limits

- Trossen, FEM cable, Ethernet insertion (scripted) does not pass yet: the round soft boot turns in the jaws and the
  port can't be lined up within the wrist's range. Deferred to a later release.
- Scripted policies cover insertion with every connector, and 1-clip routing and straight channels with the stacked
  cuboid. 2/3-clip routing, curved channels and the other routing/channel connectors run in random mode only.
- Scripted clip routing does not always seat the wire: a stiff articulation cable can stay on top of the clip's lips
  (most often with the Franka).
- The insertion reward has only been tested with PPO on the UR5, articulated cable and printable connectors; the
  paired check with a released checkpoint covers the UR5 articulation StackedCuboid policy only.
- The Trossen layout is the UR5 one scaled by 0.55 for its smaller reach, tuned for the scripted policies; its
  insertion cable starts at 0.805 m instead of 0.85 m, which would drop the cable onto the arm.
