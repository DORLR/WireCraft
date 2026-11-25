# CoRL 2026 Paper Reproduction Guide (`corl2026-repro`)

This branch (`corl2026-repro`) adds opt-in compatibility presets, data-collection flags, and a closed-loop
LeRobot policy evaluator on top of `main` so you can reproduce the CoRL 2026 paper
(*WireCraft: A Simulation Benchmark for Industrial DLO Manipulation*, [arXiv:2606.18097](https://arxiv.org/abs/2606.18097))
and directly evaluate or retrain against the published CoRL 2026 HuggingFace datasets and checkpoints without modifying `main`.

- **HuggingFace Collection:** [`DORLR/WireCraft`](https://huggingface.co/collections/DORLR/wirecraft-6ac6955d8f31d72eff08d873)

---

## 1. Published Datasets (12 Public Datasets on HuggingFace — `DORLR`)

All 12 datasets below are **public** and use LeRobot v3.0 format at **30 FPS** with **7-D joint state**
(`6` UR5 arm joints + `finger_joint`) and **7-D absolute post-IK joint targets** (`6` arm joint targets + gripper target).

### Simulation IL/VLA Benchmark Datasets (Tables 1–2, Appendices E–F)

Use these three datasets to train or evaluate **simulation** policies (`wirecraft-corl2026-sim-*`):

| Task | HuggingFace Dataset | Episodes | Frames | Cameras (`512×512`) | Paper Baselines |
| --- | --- | ---: | ---: | --- | --- |
| Connector Insertion (FEM) | [`DORLR/Deformable-StackedCuboid-Insertion-v0-Scripted-1000-apr29-1`](https://huggingface.co/datasets/DORLR/Deformable-StackedCuboid-Insertion-v0-Scripted-1000-apr29-1) | 1,000 | 340,282 | `cam_side2`, `cam_color` | ACT, Diffusion, DiT VLA, pi0.5 |
| 1-Clip Routing (FEM) | [`DORLR/Deformable-1Clip-Routing-v0-Scripted-1600`](https://huggingface.co/datasets/DORLR/Deformable-1Clip-Routing-v0-Scripted-1600) | 1,600 | 353,131 | `cam_side2`, `cam_color` | ACT, DiT VLA |
| Straight-Channel Seating (FEM) | [`DORLR/Deformable-StraightChannel-v0-Scripted-1000-aug08-1`](https://huggingface.co/datasets/DORLR/Deformable-StraightChannel-v0-Scripted-1000-aug08-1) | 1,000 | 210,021 | `cam_side2`, `cam_color` | ACT, DiT VLA |

### Real-World & Sim-to-Real Validation Datasets (Section 6, Appendix Table `tab:real_data_sources`)

Use these nine camera-ready datasets for the **physical UR5** sim-to-real experiments (`wirecraft-corl2026-real-*`).
All nine share the 30 FPS 7-D UR5 schema (`cam_side`, `cam_wrist`; real captures also include depth streams). Every tenth episode within each source (~10%) was held out for validation, and hardware checkpoint selection used only the 5 held-out real episodes per task.

| Task | Source | HuggingFace Dataset | Episodes | Frames |
| --- | --- | --- | ---: | ---: |
| Connector Insertion | Real-world (`R`) | [`DORLR/ur5_camera_ready_insertion_task`](https://huggingface.co/datasets/DORLR/ur5_camera_ready_insertion_task) | 50 | 30,018 |
| Connector Insertion | Sim Scripted (`S`) | [`DORLR/ur5_camera_ready_insertion_task_sim_scripted`](https://huggingface.co/datasets/DORLR/ur5_camera_ready_insertion_task_sim_scripted) | 200 | 121,890 |
| Connector Insertion | Sim RL Rollout (`P`) | [`DORLR/ur5_camera_ready_insertion_task_sim_rl`](https://huggingface.co/datasets/DORLR/ur5_camera_ready_insertion_task_sim_rl) | 200 | 27,394 |
| 1-Clip Routing | Real-world (`R`) | [`DORLR/ur5_camera_ready_routing_task`](https://huggingface.co/datasets/DORLR/ur5_camera_ready_routing_task) | 51 | 30,634 |
| 1-Clip Routing | Sim Scripted (`S`) | [`DORLR/ur5_camera_ready_routing_task_sim_scripted`](https://huggingface.co/datasets/DORLR/ur5_camera_ready_routing_task_sim_scripted) | 200 | 125,304 |
| 1-Clip Routing | Sim RL Rollout (`P`) | [`DORLR/ur5_camera_ready_routing_task_sim_rl`](https://huggingface.co/datasets/DORLR/ur5_camera_ready_routing_task_sim_rl) | 200 | 53,226 |
| Straight-Channel Seating | Real-world (`R`) | [`DORLR/ur5_camera_ready_seating_task`](https://huggingface.co/datasets/DORLR/ur5_camera_ready_seating_task) | 50 | 30,033 |
| Straight-Channel Seating | Sim Scripted (`S`) | [`DORLR/ur5_camera_ready_seating_task_sim_scripted`](https://huggingface.co/datasets/DORLR/ur5_camera_ready_seating_task_sim_scripted) | 200 | 114,954 |
| Straight-Channel Seating | Sim RL Rollout (`P`) | [`DORLR/ur5_camera_ready_seating_task_sim_rl`](https://huggingface.co/datasets/DORLR/ur5_camera_ready_seating_task_sim_rl) | 200 | 56,511 |

---

## 2. Published Checkpoints (20 Public Checkpoints — Exact Task Mapping)

> **Which checkpoint should I use?**
> - For **Isaac Lab simulation evaluation** (Tables 1–2), use the **8 `wirecraft-corl2026-sim-*` checkpoints** paired with the matching `--task` and `--experiment scripts/wirecraft/experiments/corl2026_*.yaml` preset below.
> - For **physical UR5 hardware deployment** (Section 6 / Figure 5), use the **12 `wirecraft-corl2026-real-*` checkpoints**. They expect real RealSense D405 views (`cam_side`, `cam_wrist`) and are **not** intended for the simulation benchmark tables.

### Simulation Checkpoints (8 Checkpoints — Tables 1–2, Appendices E–F)

| Task & Preset | Policy | Step | Paper Result | HuggingFace Checkpoint | Notes / Dependencies |
| --- | --- | ---: | --- | --- | --- |
| **Connector Insertion** (`corl2026_insertion.yaml`) | **ACT** | 100k | 60.94 ± 11.16% Reach / 29.69 ± 9.24% Insert | [`DORLR/wirecraft-corl2026-sim-insertion-act`](https://huggingface.co/DORLR/wirecraft-corl2026-sim-insertion-act) | Standard `lerobot` ACT (`eval_lerobot.py` ready) |
| Connector Insertion (`corl2026_insertion.yaml`) | Diffusion Policy | 200k | 37.50 ± 9.88% Reach / 20.31 ± 6.81% Insert | [`DORLR/wirecraft-corl2026-sim-insertion-diffusion`](https://huggingface.co/DORLR/wirecraft-corl2026-sim-insertion-diffusion) | Requires `lerobot[diffusion]` (`diffusers`) |
| Connector Insertion (`corl2026_insertion.yaml`) | DiT VLA (`multi_task_dit`) | 200k | 10.94 ± 2.71% Reach / 3.13 ± 3.13% Insert | [`DORLR/wirecraft-corl2026-sim-insertion-dit-vla`](https://huggingface.co/DORLR/wirecraft-corl2026-sim-insertion-dit-vla) | Custom `multi_task_dit` policy + CLIP ViT-B/16 tokenizer |
| Connector Insertion (`corl2026_insertion.yaml`) | pi0.5 | 20k | 6.25 ± 4.42% Reach / 0.00 ± 0.00% Insert | [`DORLR/wirecraft-corl2026-sim-insertion-pi05`](https://huggingface.co/DORLR/wirecraft-corl2026-sim-insertion-pi05) | Requires `lerobot` pi0.5 + PaliGemma tokenizer + 224×224 remapped views |
| **1-Clip Routing** (`corl2026_clip_routing.yaml`) | **ACT** | 100k | 64.06 ± 11.16% | [`DORLR/wirecraft-corl2026-sim-routing-act`](https://huggingface.co/DORLR/wirecraft-corl2026-sim-routing-act) | Standard `lerobot` ACT (`eval_lerobot.py` ready) |
| 1-Clip Routing (`corl2026_clip_routing.yaml`) | DiT VLA (`multi_task_dit`) | 300k | 39.06 ± 16.16% | [`DORLR/wirecraft-corl2026-sim-routing-dit-vla`](https://huggingface.co/DORLR/wirecraft-corl2026-sim-routing-dit-vla) | Custom `multi_task_dit` policy + CLIP ViT-B/16 tokenizer |
| **Straight-Channel** (`corl2026_straight_channel.yaml`) | **ACT** | 100k | 18.80 ± 7.65% | [`DORLR/wirecraft-corl2026-sim-seating-act`](https://huggingface.co/DORLR/wirecraft-corl2026-sim-seating-act) | Standard `lerobot` ACT (`eval_lerobot.py` ready; normalizer reconstructed deterministically from dataset `meta/stats.json`) |
| Straight-Channel (`corl2026_straight_channel.yaml`) | DiT VLA (`multi_task_dit`) | 300k | 15.62 ± 12.88% | [`DORLR/wirecraft-corl2026-sim-seating-dit-vla`](https://huggingface.co/DORLR/wirecraft-corl2026-sim-seating-dit-vla) | Custom `multi_task_dit`; normalizer reconstructed deterministically from dataset `meta/stats.json` |

### Real-World UR5 Camera-Ready ACT Checkpoints (12 Checkpoints — Section 6, Figure 5 & Appendix G)

All 12 checkpoints correspond to the camera-ready UR5 + Robotiq 2F-85 + dual RealSense D405 evaluation (20 real trials per condition; ACT trained 80k steps with batch size 16, lr `1e-5`, seed 1000, selected by lowest masked L1 action-prediction loss on the 5 held-out real episodes). Insertion checkpoints include `image_mask.json` (`black_rows: [[0, 23], [456, 480]]`).

| Hardware Task | Data Mixture | Step | Paper Result (20 trials) | HuggingFace Checkpoint |
| --- | --- | ---: | --- | --- |
| **Connector Insertion** (Ethernet-style) | `R` (Real only) | 40k | 85% Reach (17/20), 15% Insert (3/20) | [`DORLR/wirecraft-corl2026-real-insertion-act-r`](https://huggingface.co/DORLR/wirecraft-corl2026-real-insertion-act-r) |
| Connector Insertion | `R+S` (+ Sim Scripted) | 80k | 80% Reach (16/20), 45% Insert (9/20) | [`DORLR/wirecraft-corl2026-real-insertion-act-rs`](https://huggingface.co/DORLR/wirecraft-corl2026-real-insertion-act-rs) |
| Connector Insertion | `R+P` (+ Sim RL) | 55k | 95% Reach (19/20), 65% Insert (13/20) | [`DORLR/wirecraft-corl2026-real-insertion-act-rp`](https://huggingface.co/DORLR/wirecraft-corl2026-real-insertion-act-rp) |
| Connector Insertion | `R+S+P` (All sources) | 80k | 85% Reach (17/20), **80% Insert (16/20)** | [`DORLR/wirecraft-corl2026-real-insertion-act-rsp`](https://huggingface.co/DORLR/wirecraft-corl2026-real-insertion-act-rsp) |
| **1-Clip Routing** | `R` (Real only) | 55k | 45% Success (9/20) | [`DORLR/wirecraft-corl2026-real-routing-act-r`](https://huggingface.co/DORLR/wirecraft-corl2026-real-routing-act-r) |
| 1-Clip Routing | `R+S` (+ Sim Scripted) | 80k | 50% Success (10/20) | [`DORLR/wirecraft-corl2026-real-routing-act-rs`](https://huggingface.co/DORLR/wirecraft-corl2026-real-routing-act-rs) |
| 1-Clip Routing | `R+P` (+ Sim RL) | 80k | **65% Success (13/20)** | [`DORLR/wirecraft-corl2026-real-routing-act-rp`](https://huggingface.co/DORLR/wirecraft-corl2026-real-routing-act-rp) |
| 1-Clip Routing | `R+S+P` (All sources) | 70k | 60% Success (12/20) | [`DORLR/wirecraft-corl2026-real-routing-act-rsp`](https://huggingface.co/DORLR/wirecraft-corl2026-real-routing-act-rsp) |
| **Straight-Channel Seating** | `R` (Real only) | 80k | 75% Success (15/20) | [`DORLR/wirecraft-corl2026-real-seating-act-r`](https://huggingface.co/DORLR/wirecraft-corl2026-real-seating-act-r) |
| Straight-Channel Seating | `R+S` (+ Sim Scripted) | 80k | 85% Success (17/20) | [`DORLR/wirecraft-corl2026-real-seating-act-rs`](https://huggingface.co/DORLR/wirecraft-corl2026-real-seating-act-rs) |
| Straight-Channel Seating | `R+P` (+ Sim RL) | 75k | **90% Success (18/20)** | [`DORLR/wirecraft-corl2026-real-seating-act-rp`](https://huggingface.co/DORLR/wirecraft-corl2026-real-seating-act-rp) |
| Straight-Channel Seating | `R+S+P` (All sources) | 80k | **90% Success (18/20)** | [`DORLR/wirecraft-corl2026-real-seating-act-rsp`](https://huggingface.co/DORLR/wirecraft-corl2026-real-seating-act-rsp) |

---

## 3. Why This Branch Exists (`corl2026-repro` vs `main`)

The public `main` branch reflects newer WireCraft updates (improved printable USD assets, stricter geometric checks,
and an O-ring wire holder for clip routing). Training or evaluating against the CoRL 2026 paper protocol differs from
`main` defaults in six concrete ways:

| Aspect | CoRL 2026 Paper (`corl2026-repro` presets) | `main` Defaults |
| --- | --- | --- |
| Task ID (Insertion) | `WireCraft-UR5-Deformable-StackedCuboid-Insertion-v0` (FEM deformable) | `README` examples show `Articulation` |
| State & Action Encoding | **7-D** joint pos (`arm + finger`) -> **7-D** absolute post-IK joint targets (`action_encoding=joint_target`) | **14-D** state (joints + EE pose) -> **7-D** Cartesian EE delta-pose (`ee_dx..rz, gripper`) |
| Frame Rate & Resolution | **30 FPS** (`record_every_n=2` from 60 Hz sim), **`512×512`** (`cam_side2`, `cam_color`) | **60 FPS**, **`224×224`** (`side2`, `color`) |
| Connector & Socket Colors | Pure blue plug `(0.0, 0.0, 1.0)`, pure red socket `(0.8, 0.0, 0.0)` (ACT trained with `image_transforms.enable=false`) | `.usda` pale sky-blue plug `(0.15, 0.4, 0.75)`, salmon socket `(0.75, 0.2, 0.12)` |
| Scene Geometry / Randomization | Insertion: `root_pos=[0.15,0,0.805]`, `xy_range=[0.1,0.1]`, `clearance=0.003`. Clip Routing: `root_pos=[0.15,0,0.805]`, `clip_y_default=0.12`, `clip_y_range=0.04`, `clip_yaw=±15°`, `clip_depth=15 mm`, `clip_wall_height=12.5 mm`, `wire_gripper_closed_pos=0.72`, `wire_holder` tuned (`port_clearance=0.0, sustain_steps=1`). Channel: red channel `(1,0,0)`, fixed pose | Insertion: `xy_range=[0.05,0.15]`, `clearance=0.0015`. Clip Routing: `root_pos=[0.05,0,0.805]`, `clip_y_default=0.0`, `clip_y_range=0.06`, `clip_yaw=±45°`, `clip_depth=30 mm`, `wire_holder=true` |
| Success Metrics | Insertion: multi-tier tip-in-socket box & grasped (`T2` Reach `offset=(-0.005,0,0), half=(0.020,0.010,0.010)`; `T0` Insert `offset=(0,0,0), half=(0.010,0.005,0.005)`). Clip routing: 20 consecutive steps seated & gripper open. Channel: `>=75%` of geometric max in channel | Insertion: `is_seated()` (all leading-face vertices `>=10 mm` past mouth inside convex bore polygon). Clip routing: `routed() & is_released`. Channel: `>80%` in channel |

> **Note on the scripted insertion expert:** The scripted policy in `single_arm/common/scripted/insertion.py` is a
> refactored deterministic geometry-feedback FSM. It omits the paper Appendix D.1 stochastic perturbations
> (`pickdrop_recovery` with probability `0.3`, random grasp `tilt_angle ∈ [-0.45, -0.15]`, `tail_x_offset ∈ [-0.02, -0.01]`,
> and per-phase speed scales / dwell waits). For the exact paper data distribution, train directly on the published
> HuggingFace dataset `DORLR/Deformable-StackedCuboid-Insertion-v0-Scripted-1000-apr29-1`.

---

## 4. Training ACT (Exact CoRL 2026 Hyperparameters)

See [`scripts/wirecraft/configs/corl2026_act_insertion.yaml`](../../scripts/wirecraft/configs/corl2026_act_insertion.yaml)
for the exact hyperparameters used in the paper (W&B run `ys7rp397` -> published as
[`DORLR/wirecraft-corl2026-sim-insertion-act`](https://huggingface.co/DORLR/wirecraft-corl2026-sim-insertion-act)).

Run inside your `lerobot` environment (`LeRobot >= 0.4`, tested with `0.5.2`):

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

## 5. Closed-Loop Evaluation (`eval_lerobot.py`)

### What "800 simulator steps" means

In the paper evaluation (`sim.dt = 1/120 s`, `decimation = 2`, so `step_dt = 1/60 s`):
- `--episode_length 800` sets a horizon of **800 environment control steps** (`13.33 s`, i.e., **1,600 physics substeps**).
- With `n_action_steps = 50`, ACT executes **16 inference calls × 50 actions = 800 executed actions** per episode.
- At each step `t`, the evaluator drives `control_mode="joint"`, `action_scale=1.0`, sends arm delta `pred[:6] - current_joint_pos[:6]`, and maps the gripper prediction at the midpoint `0.3925 rad` (`< 0.3925 -> -1.0` open, `>= 0.3925 -> +1.0` close).
- Connector Insertion runs **4 seeds (`1000, 2000, 3000, 4000`) × 16 parallel envs = 64 rollouts** per checkpoint.
- 1-Clip Routing and Straight-Channel Seating run **3 seeds (`1000, 2000, 3000`) × 16 parallel envs = 48 rollouts** with `--episode_length 900` (`15.0 s`).

### Connector Insertion

```bash
python scripts/wirecraft/eval_lerobot.py --headless \
  --task WireCraft-UR5-Deformable-StackedCuboid-Insertion-v0 \
  --experiment scripts/wirecraft/experiments/corl2026_insertion.yaml \
  --checkpoint DORLR/wirecraft-corl2026-sim-insertion-act \
  --lerobot_python /path/to/conda/envs/lerobot/bin/python \
  --num_envs 16 --seeds 1000,2000,3000,4000 --episode_length 800 \
  --output out/eval_corl2026_insertion.json
```

### 1-Clip Routing

```bash
python scripts/wirecraft/eval_lerobot.py --headless \
  --task WireCraft-UR5-Deformable-StackedCuboid-1Clip-Routing-v0 \
  --experiment scripts/wirecraft/experiments/corl2026_clip_routing.yaml \
  --checkpoint DORLR/wirecraft-corl2026-sim-routing-act \
  --lerobot_python /path/to/conda/envs/lerobot/bin/python \
  --num_envs 16 --seeds 1000,2000,3000 --episode_length 900 \
  --output out/eval_corl2026_clip_routing.json
```

### Straight-Channel Seating

```bash
python scripts/wirecraft/eval_lerobot.py --headless \
  --task WireCraft-UR5-Deformable-StackedCuboid-StraightChannel-v0 \
  --experiment scripts/wirecraft/experiments/corl2026_straight_channel.yaml \
  --checkpoint DORLR/wirecraft-corl2026-sim-seating-act \
  --lerobot_python /path/to/conda/envs/lerobot/bin/python \
  --num_envs 16 --seeds 1000,2000,3000 --episode_length 900 \
  --output out/eval_corl2026_channel.json
```

### Verified Closed-Loop Reference Results on `corl2026-repro`

Measured on **Isaac Sim 5.1.0-rc.19** (`5.1.0-rc.19+release.26219.9c81211b.gl`) / RTX 5090 Laptop GPU / `lerobot 0.5.2` (`huggingface/lerobot@580d818a`) using the exact commands and published HF checkpoints above:

> **Isaac Sim version note:** Use **Isaac Sim 5.1** (not 5.0, where kinematic rigid-body poses written during `reset()` can lag behind PhysX in Hydra rendering). Note that Isaac Sim 5.1's Kit startup banner still prints `Kit/Isaac-Sim/5.0/user.config.json` and writes logs under `Kit/Isaac-Sim/5.0/`; verify your actual version via the `VERSION` file in the Isaac Sim installation directory.

| Task | Published Checkpoint | Rollouts | Measured on `corl2026-repro` | Paper Table |
| --- | --- | --- | --- | --- |
| Connector Insertion | `DORLR/wirecraft-corl2026-sim-insertion-act` | 64 (`4×16`) | `T2_reach`: **84.38 ± 9.38%** (54/64), `T0_insert`: **56.25 ± 10.83%** (36/64), `wc_is_seated`: **25.00 ± 9.88%** (16/64), `faulted`: 0.0% | Table 1: 60.94% Reach / 29.69% Insert (note: `rerender_on_reset=true` eliminates the stale step-0 frame present in the legacy eval loop) |
| 1-Clip Routing | `DORLR/wirecraft-corl2026-sim-routing-act` | 48 (`3×16`) | `paper_sustain20_open`: **8.33 ± 2.95%** (4/48); 37.5% (3/8) on 4-env runs | Table 2: 64.06 ± 11.16% (see physics note below) |
| Straight-Channel | `DORLR/wirecraft-corl2026-sim-seating-act` | 48 (`3×16`) | `paper_ratio_75`: **6.25 ± 5.10%** (3/48), `mean_best_fraction`: 25.97%, `faulted`: 0.0% | Table 2: 18.80 ± 7.65% |

> **Physics note on 1-Clip Routing & Straight-Channel zero-shot transfer from legacy paper checkpoints:**
> The public WireCraft release refactored the deformable wire solver, connector meshes, and clip-routing wire grip
> (replacing the legacy rigid `wire_support_pad` with the arming `wire_holder` ring, which can disarm if an IL policy
> closes briefly before reaching the wire). Consequently, legacy paper checkpoints trained on the original IsaacLab
> clip-routing/channel scenes achieve lower zero-shot success in refactored WireCraft than in the original Table 2
> simulator, whereas re-training ACT on demonstrations collected with `--paper_schema` in `corl2026-repro` avoids
> cross-simulator physics shift.

---

## 6. Collecting New Demonstrations in the CoRL 2026 Schema

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
