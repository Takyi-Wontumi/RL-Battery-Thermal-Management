#!/usr/bin/env bash
# =============================================================================
# run_training.sh
#
# One command to run the full RL training suite and baseline comparison.
#
# Usage:
#   bash run_training.sh                  # local CPU defaults (100K steps each)
#   bash run_training.sh --colab          # Colab GPU defaults (1M PPO, 750K SAC)
#   bash run_training.sh --ppo-steps 200000 --sac-steps 150000
#   bash run_training.sh --baselines-only # skip RL training, run baselines + demo
#   bash run_training.sh --skip-baselines # skip baseline comparison
#
# Stages:
#   1. Physics demo       — run_3d_simulation.py (HPPC + multi-zone, ~30 s)
#   2. Baseline benchmark — compare_pack_baselines_3d.py (6 classical controllers)
#   3. PPO training       — train_pack_ppo_3d.py
#   4. SAC training       — train_pack_sac_3d.py
#   5. Full evaluation    — compare_controllers.py (baselines + PPO + SAC)
#
# All outputs go to:
#   outputs/comparison/        — evaluation CSVs and plots
#   models/ppo_pack_3d/        — PPO model weights and checkpoints
#   models/sac_pack_3d/        — SAC model weights and checkpoints
#   logs/ppo_pack_3d/          — PPO TensorBoard logs
#   logs/sac_pack_3d/          — SAC TensorBoard logs
#   simulation_results.png     — HPPC demo plot (project root)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PPO_STEPS=5000000
SAC_STEPS=3000000
N_ENVS=4
DEVICE="auto"
BASELINES_ONLY=false
SKIP_BASELINES=false
SKIP_DEMO=false
EPISODES=20

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --colab)
            PPO_STEPS=10000000
            SAC_STEPS=750000
            N_ENVS=8
            shift ;;
        --ppo-steps)
            PPO_STEPS="$2"; shift 2 ;;
        --sac-steps)
            SAC_STEPS="$2"; shift 2 ;;
        --n-envs)
            N_ENVS="$2"; shift 2 ;;
        --device)
            DEVICE="$2"; shift 2 ;;
        --episodes)
            EPISODES="$2"; shift 2 ;;
        --baselines-only)
            BASELINES_ONLY=true; shift ;;
        --skip-baselines)
            SKIP_BASELINES=true; shift ;;
        --skip-demo)
            SKIP_DEMO=true; shift ;;
        *)
            echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use the project venv if it exists, otherwise fall back to PATH python
if [[ -f ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

export PYTHONPATH="$SCRIPT_DIR"

echo ""
echo "============================================================"
echo "  RL Battery Thermal Management — Full Training Suite"
echo "============================================================"
echo "  Python:          $PYTHON"
echo "  PPO timesteps:   $PPO_STEPS"
echo "  SAC timesteps:   $SAC_STEPS"
echo "  Parallel envs:   $N_ENVS  (PPO)"
echo "  Device:          $DEVICE"
echo "  Baselines only:  $BASELINES_ONLY"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------
FAILED_STAGES=()
START_TOTAL=$(date +%s)

run_stage() {
    local label="$1"
    local cmd="${@:2}"
    echo ""
    echo "------------------------------------------------------------"
    echo "  STAGE: $label"
    echo "  CMD:   $cmd"
    echo "------------------------------------------------------------"
    local t0=$(date +%s)
    if $cmd; then
        local t1=$(date +%s)
        echo "  ✓ $label done  ($(( t1 - t0 ))s)"
    else
        local t1=$(date +%s)
        echo "  ✗ $label FAILED  ($(( t1 - t0 ))s) — continuing..."
        FAILED_STAGES+=("$label")
    fi
}

# ---------------------------------------------------------------------------
# Stage 1 — Physics demo (HPPC + multi-zone, produces simulation_results.png)
# ---------------------------------------------------------------------------
if [[ "$SKIP_DEMO" == false ]]; then
    run_stage "Physics demo (HPPC + multi-zone)" \
        $PYTHON scripts/run_3d_simulation.py
fi

# ---------------------------------------------------------------------------
# Stage 2 — Baseline benchmark (6 classical controllers, 4 heat profiles)
# ---------------------------------------------------------------------------
if [[ "$SKIP_BASELINES" == false ]]; then
    run_stage "Baseline benchmark" \
        $PYTHON -m scripts.compare_pack_baselines_3d
fi

if [[ "$BASELINES_ONLY" == true ]]; then
    echo ""
    echo "============================================================"
    echo "  Baselines-only run complete."
    echo "============================================================"
    exit 0
fi

# ---------------------------------------------------------------------------
# Stage 3 — PPO training
# ---------------------------------------------------------------------------
run_stage "PPO training ($PPO_STEPS steps, curriculum)" \
    $PYTHON training/train_pack_ppo_3d.py \
        --timesteps "$PPO_STEPS" \
        --n-envs "$N_ENVS" \
        --device "$DEVICE" \
        --curriculum \
        --save-dir models/ppo_pack_3d_multizone_sensor \
        --log-dir  logs/ppo_pack_3d_multizone_sensor

# ---------------------------------------------------------------------------
# Stage 4 — SAC training
# ---------------------------------------------------------------------------
run_stage "SAC training ($SAC_STEPS steps, curriculum)" \
    $PYTHON training/train_pack_sac_3d.py \
        --timesteps "$SAC_STEPS" \
        --device "$DEVICE" \
        --curriculum \
        --save-dir models/sac_pack_3d_multizone_sensor \
        --log-dir  logs/sac_pack_3d_multizone_sensor

# ---------------------------------------------------------------------------
# Stage 5 — Full evaluation (baselines + whichever RL models exist)
# ---------------------------------------------------------------------------
PPO_MODEL="models/ppo_pack_3d_multizone_sensor/best_model.zip"
SAC_MODEL="models/sac_pack_3d_multizone_sensor/best_model.zip"

EVAL_ARGS=(
    --results-dir outputs/comparison
    --episodes "$EPISODES"
)

[[ -f "$PPO_MODEL" ]] && EVAL_ARGS+=(--ppo-model "$PPO_MODEL")
[[ -f "$SAC_MODEL" ]] && EVAL_ARGS+=(--sac-model "$SAC_MODEL")

run_stage "Full evaluation" \
    $PYTHON evaluation/compare_controllers.py "${EVAL_ARGS[@]}"

# ---------------------------------------------------------------------------
# Stage 6 — 3D PyVista static screenshots (PI + SAC + PPO side-by-side)
# ---------------------------------------------------------------------------
run_stage "3D visualization (PI / SAC / PPO)" \
    $PYTHON scripts/visualize_3d_controllers.py \
        --output-dir outputs/3d_viz \
        --profile NonuniformStep

# ---------------------------------------------------------------------------
# Stage 7 — GIF animations  (all baselines + RL models if available)
#
# All 6 classical controllers always run.
# SAC / PPO are appended only if their model files exist.
# Output: outputs/3d_pyvista_<controller>_<profile>.gif
# ---------------------------------------------------------------------------
GIF_ARGS=(
    --no-cooling
    --constant-05
    --constant-1
    --bang-bang
    --proportional
    --PI
    --profile NonuniformStep
    --stride 10
    --fps 12
)

[[ -f "$SAC_MODEL" ]] && GIF_ARGS+=(--SAC)
[[ -f "$PPO_MODEL" ]] && GIF_ARGS+=(--PPO)

run_stage "GIF animations (baselines + RL)" \
    $PYTHON -m scripts.animate_pack_3d_pyvista "${GIF_ARGS[@]}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
END_TOTAL=$(date +%s)
ELAPSED=$(( END_TOTAL - START_TOTAL ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo "============================================================"
echo "  Run complete  (${MINS}m ${SECS}s total)"
echo "============================================================"
echo ""
echo "  Key outputs:"
echo "    simulation_results.png            — HPPC + multi-zone demo"
echo "    outputs/                          — baseline benchmark plots/CSV"
echo "    outputs/comparison/               — full evaluation plots/CSV"
echo "    outputs/3d_viz/                   — 3D PyVista screenshots + comparison figure"
echo "    outputs/3d_pyvista_*_*.gif        — GIF animations (all controllers)"
echo "    models/ppo_pack_3d_multizone_sensor/  — PPO weights"
echo "    models/sac_pack_3d_multizone_sensor/  — SAC weights"
echo "    logs/ppo_pack_3d_multizone_sensor/    — PPO TensorBoard logs"
echo "    logs/sac_pack_3d_multizone_sensor/    — SAC TensorBoard logs"
echo ""
echo "  TensorBoard:"
echo "    $PYTHON -m tensorboard.main --logdir logs/"
echo ""

if [[ ${#FAILED_STAGES[@]} -gt 0 ]]; then
    echo "  Failed stages:"
    for s in "${FAILED_STAGES[@]}"; do
        echo "    ✗ $s"
    done
    echo ""
    exit 1
fi

echo "  All stages passed."
echo ""
