# Colab Notebooks — Battery Thermal RL

Run order:

| Notebook | Purpose |
|---|---|
| `00_colab_setup.ipynb` | Mount Drive, install deps, verify GPU, test imports |
| `01_train_ppo_colab.ipynb` | Train PPO (3D pack) — save to Drive |
| `02_train_sac_colab.ipynb` | Train SAC (3D pack) — save to Drive |
| `03_compare_controllers_colab.ipynb` | Compare all controllers, generate result plots |

## Quick Start

1. Open `00_colab_setup.ipynb` and run all cells top to bottom.
2. Choose **Option A** (clone from GitHub) or **Option B** (Drive project folder).
3. Run `01_train_ppo_colab.ipynb` for PPO training.
4. Run `02_train_sac_colab.ipynb` for SAC training.
5. Run `03_compare_controllers_colab.ipynb` to benchmark all controllers.

All models, logs, and results save to:

```
MyDrive/battery_rl/
├── models/
│   ├── ppo_3d_pack/    ← ppo_pack_final.zip, best_model.zip, checkpoints/
│   └── sac_3d_pack/    ← sac_pack_final.zip, best_model.zip, checkpoints/
├── logs/
│   ├── ppo_3d_pack/    ← TensorBoard
│   └── sac_3d_pack/    ← TensorBoard
└── results/
    └── comparison/     ← CSV + plots
```

## Before Full Training — Smoke Test

Run 10 000 steps to verify the pipeline works before committing to 3M:

```bash
!python training/train_pack_ppo_3d.py --timesteps 10000 --n-envs 2 \
    --save-dir "$BASE_DRIVE_DIR/models/ppo_3d_pack" \
    --log-dir  "$BASE_DRIVE_DIR/logs/ppo_3d_pack"

!python training/train_pack_sac_3d.py --timesteps 10000 \
    --save-dir "$BASE_DRIVE_DIR/models/sac_3d_pack" \
    --log-dir  "$BASE_DRIVE_DIR/logs/sac_3d_pack"
```

## Warning

SAC requires a **continuous Box action space**. The environment already uses
`spaces.Box(low=0.0, high=1.0)` for the cooling command, so SAC is compatible.
If you modify the environment to use a discrete action space, SAC will raise an
error before training starts — this is intentional.

## Architecture

The notebooks are a **remote control**, not an engine. All physics, training
logic, and evaluation code lives in the project Python files:

```
envs/      — thermal environments
training/  — PPO and SAC training scripts
evaluation/ — controller comparison
scripts/   — baseline controllers, animation
configs/   — pack and cell configuration
models/    — thermal model (physics)
```

Do not paste environment or training code into notebook cells.
