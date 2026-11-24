# CoRL 2026 Paper Reproduction Guide (`corl2026-repro`)

This branch (`corl2026-repro`) adds opt-in compatibility presets, data-collection flags, and a closed-loop
LeRobot policy evaluator on top of `main` so you can reproduce the CoRL 2026 paper
(*WireCraft: A Simulation Benchmark for Industrial DLO Manipulation*, [arXiv:2606.18097](https://arxiv.org/abs/2606.18097))
and directly reuse the published CoRL 2026 HuggingFace datasets without modifying `main`.

---

## 1. Published Datasets (HuggingFace — `DORLR`)

All datasets below use LeRobot v3.0 format at **30 FPS** with **7-D joint state** (`6` UR5 arm joints + `finger_joint`)
and **7-D absolute post-IK joint targets** (`6` arm joint targets + binary gripper target `0.0` open / `0.785` closed).

### Simulation IL/VLA Benchmark Datasets (Tables 1–2, Appendices E–F)

| Task | HuggingFace Dataset | Episodes | Frames | Cameras (`512×512`) | Paper Baselines |
| --- | --- | ---: | ---: | --- | --- |
| Connector Insertion (FEM) | [`DORLR/Deformable-StackedCuboid-Insertion-v0-Scripted-1000-apr29-1`](https://huggingface.co/datasets/DORLR/Deformable-StackedCuboid-Insertion-v0-Scripted-1000-apr29-1) | 1,000 | 340,282 | `cam_side2`, `cam_color` | ACT, Diffusion, DiT VLA, pi0.5 |
| 1-Clip Routing (FEM) | [`DORLR/Deformable-1Clip-Routing-v0-Scripted-1600`](https://huggingface.co/datasets/DORLR/Deformable-1Clip-Routing-v0-Scripted-1600) | 1,600 | 353,131 | `cam_side2`, `cam_color` | ACT, DiT VLA |
| Straight-Channel Seating (FEM) | [`DORLR/Deformable-StraightChannel-v0-Scripted-1000-aug08-1`](https://huggingface.co/datasets/DORLR/Deformable-StraightChannel-v0-Scripted-1000-aug08-1) | 1,000 | 210,021 | `cam_side2`, `cam_color` | ACT, DiT VLA |

### Real-World & Sim-to-Real Validation Datasets (Section 6, Appendix Table `tab:real_data_sources`)

All nine hardware/sim-to-real datasets share the 30 FPS 7-D UR5 schema (`cam_side`, `cam_wrist`; real-world captures also include depth streams):

| Task | Source | HuggingFace Dataset | Episodes | Frames |
| --- | --- | --- | ---: | ---: |
| Connector Insertion | Real-world (R) | [`DORLR/ur5_camera_ready_insertion_task`](https://huggingface.co/datasets/DORLR/ur5_camera_ready_insertion_task) | 50 | 30,018 |
| Connector Insertion | Sim Scripted (S) | [`DORLR/ur5_camera_ready_insertion_task_sim_scripted`](https://huggingface.co/datasets/DORLR/ur5_camera_ready_insertion_task_sim_scripted) | 200 | 121,890 |
| Connector Insertion | Sim RL Rollout (P) | [`DORLR/ur5_camera_ready_insertion_task_sim_rl`](https://huggingface.co/datasets/DORLR/ur5_camera_ready_insertion_task_sim_rl) | 200 | 27,394 |
| 1-Clip Routing | Real-world (R) | [`DORLR/ur5_camera_ready_routing_task`](https://huggingface.co/datasets/DORLR/ur5_camera_ready_routing_task) | 51 | 30,634 |
| 1-Clip Routing | Sim Scripted (S) | [`DORLR/ur5_camera_ready_routing_task_sim_scripted`](https://huggingface.co/datasets/DORLR/ur5_camera_ready_routing_task_sim_scripted) | 200 | 125,304 |
| 1-Clip Routing | Sim RL Rollout (P) | [`DORLR/ur5_camera_ready_routing_task_sim_rl`](https://huggingface.co/datasets/DORLR/ur5_camera_ready_routing_task_sim_rl) | 200 | 53,226 |
| Straight-Channel Seating | Real-world (R) | [`DORLR/ur5_camera_ready_seating_task`](https://huggingface.co/datasets/DORLR/ur5_camera_ready_seating_task) | 50 | 30,033 |
| Straight-Channel Seating | Sim Scripted (S) | [`DORLR/ur5_camera_ready_seating_task_sim_scripted`](https://huggingface.co/datasets/DORLR/ur5_camera_ready_seating_task_sim_scripted) | 200 | 114,954 |
| Straight-Channel Seating | Sim RL Rollout (P) | [`DORLR/ur5_camera_ready_seating_task_sim_rl`](https://huggingface.co/datasets/DORLR/ur5_camera_ready_seating_task_sim_rl) | 200 | 56,511 |

---

## 2. Why This Branch Exists (`corl2026-repro` vs `main`)

The public `main` branch reflects newer WireCraft updates (improved printable USD assets, stricter geometric checks,
and an O-ring wire holder for clip routing). Training or evaluating against the CoRL 2026 paper protocol differs from
`main` defaults in six concrete ways:

| Aspect | CoRL 2026 Paper (`corl2026-repro` presets) | `main` Defaults |
| --- | --- | --- |
| Task ID (Insertion) | `WireCraft-UR5-Deformable-StackedCuboid-Insertion-v0` (FEM deformable) | `README` examples show `Articulation` |
| State & Action Encoding | **7-D** joint pos (`arm + finger`) -> **7-D** absolute post-IK joint targets (`action_encoding=joint_target`) | **14-D** state (joints + EE pose) -> **7-D** Cartesian EE delta-pose (`ee_dx..rz, gripper`) |
| Frame Rate & Resolution | **30 FPS** (`record_every_n=2` from 60 Hz sim), **`512×512`** (`cam_side2`, `cam_color`) | **60 FPS**, **`224×224`** (`side2`, `color`) |
| Connector & Socket Colors | Pure blue plug `(0.0, 0.0, 1.0)`, pure red socket `(0.8, 0.0, 0.0)` (ACT trained with `image_transforms.enable=false`) | `.usda` pale sky-blue plug `(0.15, 0.4, 0.75)`, salmon socket `(0.75, 0.2, 0.12)` |
| Scene Geometry / Randomization | Insertion: `root_pos=[0.15,0,0.805]`, `xy_range=[0.1,0.1]`, `clearance=0.003`. Clip Routing: `root_pos=[0.15,0,0.805]`, `clip_y_default=0.12`, `clip_y_range=0.04`, `clip_yaw=±15°`, `clip_depth=15 mm`, `clip_wall_height=12.5 mm`, `wire_gripper_closed_pos=0.72`, `wire_holder` tuned (`port_clearance=0.0, sustain_steps=1`) to match legacy `wire_support_pad`. Channel: red channel `(1,0,0)`, fixed pose | Insertion: `xy_range=[0.05,0.15]`, `clearance=0.0015`. Clip Routing: `root_pos=[0.05,0,0.805]`, `clip_y_default=0.0`, `clip_y_range=0.06`, `clip_yaw=±45°`, `clip_depth=30 mm`, `wire_holder=true` |
| Success Metrics | Insertion: multi-tier tip-in-socket box & grasped (`T2` Reach `offset=(-0.005,0,0), half=(0.020,0.010,0.010)`; `T0` Insert `offset=(0,0,0), half=(0.010,0.005,0.005)`). Clip routing: 20 consecutive steps seated & gripper open. Channel: `>=75%` in channel | Insertion: `is_seated()` (all leading-face vertices `>=10 mm` past mouth inside convex bore polygon). Clip routing: loose `routed() & is_released`. Channel: `>80%` in channel |

> **Note on the scripted insertion expert:** The scripted policy in `single_arm/common/scripted/insertion.py` is a
> refactored deterministic geometry-feedback FSM. It omits the paper Appendix D.1 stochastic perturbations
> (`pickdrop_recovery` with probability `0.3`, random grasp `tilt_angle ∈ [-0.45, -0.15]`, `tail_x_offset ∈ [-0.02, -0.01]`,
> and per-phase speed scales / dwell waits). For exact paper data distribution, train directly on the published
> HuggingFace dataset `DORLR/Deformable-StackedCuboid-Insertion-v0-Scripted-1000-apr29-1`.

---

## 3. Training ACT (Exact CoRL 2026 Hyperparameters)

See [`scripts/wirecraft/configs/corl2026_act_insertion.yaml`](../../scripts/wirecraft/configs/corl2026_act_insertion.yaml)
for the exact hyperparameters used in the paper (W&B run `ys7rp397` -> 100k checkpoint:
`Reach (T2) = 60.94 ± 11.16%`, `Insert (T0) = 29.69 ± 9.24%` over 64 rollouts).

Run inside your `lerobot` environment (`LeRobot >= 0.4`):

```bash
lerobot-train \
  --dataset.repo_id=DORLR/Deformable-StackedCuboid-Insertion-v0-Scripted-1000-apr29-1 \
  --dataset.image_transforms.enable=false \
  --dataset.use_imagenet_stats=true \
  --policy.type=act \
  --policy.vision_backbone=resnet18 \
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
  --policy.chunk_size=100 \
  --policy.n_action_steps=50 \
  --policy.n_obs_steps=1 \
  --policy.dim_model=512 \
  --policy.dim_feedforward=3200 \
  --policy.n_heads=8 \
  --policy.n_encoder_layers=4 \
  --policy.n_decoder_layers=1 \
  --policy.use_vae=true \
  --policy.latent_dim=32 \
  --policy.kl_weight=10.0 \
  --policy.dropout=0.1 \
  --policy.optimizer_lr=1e-5 \
  --policy.optimizer_lr_backbone=1e-5 \
  --policy.optimizer_weight_decay=1e-4 \
  --policy.push_to_hub=false \
  --batch_size=64 \
  --steps=100000 \
  --save_freq=5000 \
  --seed=1000 \
  --output_dir=outputs/train/corl2026_act_insertion
```

> Key detail: `n_decoder_layers=1` (matching LeRobot ACT default, **not** 7), `chunk_size=100`, `n_action_steps=50`,
> `optimizer_lr=1e-5`, `optimizer_lr_backbone=1e-5`, `image_transforms.enable=false`, `use_imagenet_stats=true`.

---

## 4. Closed-Loop Evaluation (`eval_lerobot.py`)

### What "800 simulator steps" means

In the paper evaluation (`sim.dt = 1/120 s`, `decimation = 2`, so `step_dt = 1/60 s`):
- `--episode_length 800` sets a horizon of **800 environment control steps** (`13.33 s`, i.e., **1,600 physics substeps**).
- With `n_action_steps = 50`, ACT executes **16 inference calls × 50 actions = 800 executed actions** per episode.
- At each step `t`, the evaluator drives `control_mode="joint"`, `action_scale=1.0`, sends arm delta `pred[:6] - current_joint_pos[:6]`, and maps the gripper prediction at the midpoint `0.3925 rad` (`< 0.3925 -> -1.0` open, `>= 0.3925 -> +1.0` close).
- Evaluation runs **4 seeds (`1000, 2000, 3000, 4000`) × 16 parallel envs = 64 rollouts** per checkpoint.
- 1-Clip Routing and Straight-Channel Seating use `--episode_length 900` (`15.0 s`).

### Connector Insertion

```bash
python scripts/wirecraft/eval_lerobot.py --headless \
  --task WireCraft-UR5-Deformable-StackedCuboid-Insertion-v0 \
  --experiment scripts/wirecraft/experiments/corl2026_insertion.yaml \
  --checkpoint outputs/train/corl2026_act_insertion/checkpoints/100000/pretrained_model \
  --lerobot_python /path/to/conda/envs/lerobot/bin/python \
  --num_envs 16 --seeds 1000,2000,3000,4000 --episode_length 800 \
  --output out/eval_corl2026_insertion.json
```

### 1-Clip Routing

```bash
python scripts/wirecraft/eval_lerobot.py --headless \
  --task WireCraft-UR5-Deformable-StackedCuboid-1Clip-Routing-v0 \
  --experiment scripts/wirecraft/experiments/corl2026_clip_routing.yaml \
  --checkpoint /path/to/clip_routing_act_100k \
  --lerobot_python /path/to/conda/envs/lerobot/bin/python \
  --num_envs 16 --seeds 1000,2000,3000,4000 --episode_length 900 \
  --output out/eval_corl2026_clip_routing.json
```

### Straight-Channel Seating

```bash
python scripts/wirecraft/eval_lerobot.py --headless \
  --task WireCraft-UR5-Deformable-StackedCuboid-StraightChannel-v0 \
  --experiment scripts/wirecraft/experiments/corl2026_straight_channel.yaml \
  --checkpoint /path/to/straight_channel_act_100k \
  --lerobot_python /path/to/conda/envs/lerobot/bin/python \
  --num_envs 16 --seeds 1000,2000,3000,4000 --episode_length 900 \
  --output out/eval_corl2026_channel.json
```

---

## 5. Collecting New Demonstrations in the CoRL 2026 Schema

Pass `--paper_schema` to `collect_demos.py` (which enables `--action_encoding joint_target`, `--record_every_n 2` for
30 FPS, `--camera_resolution 512`, and `--camera_prefix cam_`), and `--no_env_state` to `to_lerobot.py`:

```bash
# 1. Inside the Isaac Lab / WireCraft environment:
python scripts/wirecraft/collect_demos.py --headless \
  --task WireCraft-UR5-Deformable-StackedCuboid-Insertion-v0 \
  --experiment scripts/wirecraft/experiments/corl2026_insertion.yaml \
  --policy scripted --paper_schema --cameras side2 color --only_success \
  --num_envs 16 --num_episodes 1000 --output demos/corl2026_insertion

# 2. Inside the lerobot environment:
python scripts/wirecraft/to_lerobot.py demos/corl2026_insertion \
  --repo_id <hf_user>/wirecraft_corl2026_insertion \
  --root datasets/corl2026_insertion --vcodec h264 --no_env_state
```
