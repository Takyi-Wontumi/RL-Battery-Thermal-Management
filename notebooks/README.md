# Colab Notebooks — Battery Thermal RL

These notebooks are a **Colab translation layer** for the VS Code project.
They do not contain any environment, training, or evaluation logic.
All real code lives in the project Python files and runs unchanged.

## Run Order

| Notebook | Purpose |
|---|---|
| `00_colab_setup.ipynb` | Clone repo, install deps, GPU check, diagnostic, import test |
| `01_train_ppo_colab.ipynb` | Run `training/train_pack_ppo_3d.py` with Drive output paths |
| `02_train_sac_colab.ipynb` | Run `training/train_pack_sac_3d.py` with Drive output paths |
| `03_compare_controllers_colab.ipynb` | Run `evaluation/compare_controllers.py`, display results |

## What Each Notebook Does

1. Mounts Google Drive
2. Clones or pulls the GitHub repository
3. `cd`s into the project root
4. Installs `requirements.txt`
5. Checks GPU availability
6. Runs a diagnostic cell (pwd, file structure, import graph)
7. Calls the existing project scripts with `PYTHONPATH=.`
8. Saves models / logs / results to Google Drive

## Google Drive Output Structure

```
MyDrive/battery_rl/
├── models/
│   ├── ppo_3d_pack/    ← ppo_pack_final.zip, best_model.zip, vec_normalize.pkl, checkpoints/
│   └── sac_3d_pack/    ← sac_pack_final.zip, best_model.zip, checkpoints/
├── logs/
│   ├── ppo_3d_pack/    ← TensorBoard events
│   └── sac_3d_pack/    ← TensorBoard events
└── results/
    └── comparison/     ← comparison_metrics.csv + 7 plots
```

## How Training Commands Look

```bash
!PYTHONPATH=. python training/train_pack_ppo_3d.py \
  --timesteps 3000000 \
  --n-envs 8 \
  --save-dir /content/drive/MyDrive/battery_rl/models/ppo_3d_pack \
  --log-dir  /content/drive/MyDrive/battery_rl/logs/ppo_3d_pack

!PYTHONPATH=. python training/train_pack_sac_3d.py \
  --timesteps 3000000 \
  --save-dir /content/drive/MyDrive/battery_rl/models/sac_3d_pack \
  --log-dir  /content/drive/MyDrive/battery_rl/logs/sac_3d_pack
```

## Smoke Test (before full training)

```bash
!PYTHONPATH=. python training/train_pack_ppo_3d.py --timesteps 10000 --n-envs 2 \
  --save-dir /content/drive/MyDrive/battery_rl/test/ppo \
  --log-dir  /content/drive/MyDrive/battery_rl/test/ppo_logs

!PYTHONPATH=. python training/train_pack_sac_3d.py --timesteps 10000 \
  --save-dir /content/drive/MyDrive/battery_rl/test/sac \
  --log-dir  /content/drive/MyDrive/battery_rl/test/sac_logs
```

## Rules

- Do **not** paste environment code, reward functions, or training loops into notebook cells.
- Do **not** rename any project files or change any existing imports.
- The VS Code project must continue working locally without any modification.
- The notebooks are the only Colab-specific layer.
