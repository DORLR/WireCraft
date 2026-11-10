# WireCraft environments

96 Gym IDs, all built from one single-arm `DirectRLEnv`
(`source/isaaclab_tasks/isaaclab_tasks/direct/wirecraft/single_arm/common/env.py`).
`python scripts/wirecraft/run.py --list ids.json` writes the full registry.

## Naming

```
WireCraft-<Robot>-<Wire>-<Connector>-<Task>-v0
```

| Field | Values |
| --- | --- |
| Robot | `UR5` (Robotiq 2F-85), `Franka` (Panda), `Trossen` (right arm of the AI Stationary kit) |
| Wire | `Articulation` (12-capsule chain), `Deformable` (FEM) |
| Connector, printable | `Cylinder`, `Cuboid`, `StackedCuboid`, `DisplayPortStyle` |
| Connector, real | `Ethernet`, `USBA`, `DisplayPort`, `HDMI` |

## Tasks

| Task | ID suffix | Connectors | Success | IDs |
| --- | --- | --- | --- | --- |
| Connector insertion | `Insertion` | all 8 | whole leading face 10 mm into the bore at some step of the episode | 48 |
| Clip routing | `{1,2,3}Clip-Routing` | `StackedCuboid` (1-3 clips), `USBA` (1 clip) | wire through every clip window | 24 |
| Lay in channel | `{Straight,Curved}Channel` | `StackedCuboid`, `USBA` | at least 80 % of the wire inside the slot | 24 |

Each count is 3 robots x 2 wires x the connector/variant combinations.

## Spaces

| | Insertion | Clip routing | Lay in channel |
| --- | --- | --- | --- |
| Action | 7: tool delta position and rotation (scale 0.01), gripper open (<0) / close | articulation 4: tool delta position (scale 0.5), gripper; deformable 7 as insertion | as clip routing |
| Observation (UR5, 7 joints) | 42 | 24 | 27 + one per channel module (straight 30, curved 32) |
| Episode | 8 s at 60 Hz | 5 s | 5 s |

Franka has 9 joints (two more observation values). Observations, rewards and resets per task live in
`single_arm/common/tasks/<task>.py`; the insertion reward is described in the task README.

## Coverage

| | Insertion | 1-clip routing | Straight channel | 2/3-clip routing, curved channel, USBA routing/channel |
| --- | --- | --- | --- | --- |
| Random policy (uniform actions) | all | all | all | all |
| Scripted policy | all 8 connectors | `StackedCuboid` | `StackedCuboid` | not yet |
| PPO reward tested | UR5, articulation, printable connectors | - | - | - |

Configuration layers (robot, task, wire, connector, mode, experiment YAML, command line) and the per-robot
layout notes are in [`source/isaaclab_tasks/isaaclab_tasks/direct/wirecraft/README.md`](../../source/isaaclab_tasks/isaaclab_tasks/direct/wirecraft/README.md).
