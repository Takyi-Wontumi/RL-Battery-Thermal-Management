"""
notebooks/05_gpu_training_colab.py

Complete GPU-only production training pipeline for RL Battery Thermal Management.
Run this ONE command in a Google Colab GPU cell:

    !python notebooks/05_gpu_training_colab.py

Or upload the file to /content/ and run:
    !python /content/05_gpu_training_colab.py

What this script does (in order):
  Step 1  GPU assertion       — aborts immediately if no CUDA GPU is detected
  Step 2  CUDA validation     — reads driver + toolkit version via nvidia-smi
  Step 3  PyTorch install     — installs the CUDA-backed wheel matching the driver
  Step 4  Dependency install  — installs all project packages + xvfb
  Step 5  GPU benchmark       — FP32 matmul warmup, reports achievable TFLOPS
  Step 6  Clone / pull repo   — git clone or git pull
  Step 7  Drive mount         — mounts Google Drive for model persistence (optional)
  Step 8  Env verification    — confirms 29-D sensor env + CUDA are both live

  Stage 1  Baseline benchmark      — 6 classical controllers × 4 heat profiles
  Stage 2  PPO training            — 5 M steps, curriculum, --device cuda
  Stage 3  SAC training            — 3 M steps, curriculum, --device cuda
  Stage 4  Full evaluation         — all controllers + safety-shielded RL variants
  Stage 5  3D visualization        — PyVista screenshots + GIF animations
  Stage 6  Failure diagnosis       — per-episode failure plots + root-cause report
"""

from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
import time

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  —  edit the variables in this block before running
# ═══════════════════════════════════════════════════════════════════════════════

REPO_URL = "https://github.com/Takyi-Wontumi/RL-Battery-Thermal-Management.git"
REPO_DIR = "/content/RL-Battery-Thermal-Management"

# Google Drive persistence ────────────────────────────────────────────────────
# USE_DRIVE = True   → mount Drive, save models there (survive session resets)
# USE_DRIVE = False  → save locally under REPO_DIR (lost when session ends)
USE_DRIVE  = True
DRIVE_ROOT = "/content/drive/MyDrive/RL-Battery-Thermal"

# Training scale ──────────────────────────────────────────────────────────────
# "SMOKE" → 10 000 steps each  (~10 min end-to-end)  — pipeline verification
# "FULL"  → 5 M PPO + 3 M SAC (~7–10 h on T4 GPU)   — production quality
TRAINING_SCALE = "FULL"

# Pipeline stage toggles ──────────────────────────────────────────────────────
RUN_BASELINES = True
RUN_PPO       = True
RUN_SAC       = True
RUN_EVAL      = True
RUN_VIZ       = True
RUN_DIAGNOSIS = True

# ═══════════════════════════════════════════════════════════════════════════════
# DERIVED CONFIG  —  do not edit below this line
# ═══════════════════════════════════════════════════════════════════════════════

_SCALE_CFGS = {
    "SMOKE": dict(
        ppo=10_000,      sac=10_000,      n_envs=2, episodes=4,
        ppo_s2=3_000,    ppo_s3=6_000,
        sac_s2=3_000,    sac_s3=6_000,
    ),
    "FULL": dict(
        ppo=5_000_000,   sac=3_000_000,   n_envs=4, episodes=20,
        ppo_s2=1_000_000, ppo_s3=2_500_000,
        sac_s2=500_000,   sac_s3=1_500_000,
    ),
}

if TRAINING_SCALE not in _SCALE_CFGS:
    print(f"Unknown TRAINING_SCALE '{TRAINING_SCALE}'. Choose 'SMOKE' or 'FULL'.")
    sys.exit(1)

_C        = _SCALE_CFGS[TRAINING_SCALE]
PPO_STEPS = _C["ppo"];     SAC_STEPS = _C["sac"]
N_ENVS    = _C["n_envs"];  EPISODES  = _C["episodes"]
PPO_S2    = _C["ppo_s2"];  PPO_S3    = _C["ppo_s3"]
SAC_S2    = _C["sac_s2"];  SAC_S3    = _C["sac_s3"]

FAILED_STAGES: list[str] = []
_T_START = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _sh(cmd: str, **kw) -> subprocess.CompletedProcess:
    """Run a shell command via the system shell (no exception on failure)."""
    return subprocess.run(cmd, shell=True, **kw)


def _sh_ok(cmd: str, **kw) -> None:
    """Run a shell command; sys.exit(1) on non-zero return code."""
    r = subprocess.run(cmd, shell=True, **kw)
    if r.returncode != 0:
        sys.exit(1)


def _capture(cmd: str) -> tuple[int, str]:
    """Run a shell command, return (returncode, combined stdout+stderr)."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def _banner(title: str) -> None:
    print(f"\n{'═' * 64}")
    print(f"  {title}")
    print(f"{'═' * 64}\n")


def _step(n: int, title: str) -> None:
    print(f"\n{'─' * 64}")
    print(f"  STEP {n}: {title}")
    print(f"{'─' * 64}")


def _run_stage(label: str, cmd: list[str]) -> None:
    """Run a pipeline stage as a subprocess; record failures without aborting."""
    print(f"\n  ┌─ STAGE: {label}")
    print(f"  │  CMD:   {' '.join(str(a) for a in cmd)}")
    t0  = time.time()
    env = {**os.environ, "PYTHONPATH": REPO_DIR}
    r   = subprocess.run(cmd, cwd=REPO_DIR, env=env)
    dt  = int(time.time() - t0)
    if r.returncode == 0:
        print(f"  └─ ✓  {label}  ({dt}s)\n")
    else:
        print(f"  └─ ✗  {label} FAILED  ({dt}s) — continuing …\n")
        FAILED_STAGES.append(label)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — GPU ASSERTION
# ═══════════════════════════════════════════════════════════════════════════════
_banner("RL-Battery-Thermal: Colab GPU Production Training")
_step(1, "GPU assertion")

_rc, _smi_csv = _capture(
    "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"
)
if _rc != 0:
    print("\n  ✗  NO GPU DETECTED — ABORTING\n")
    print("  This script requires a GPU runtime.")
    print("  In Colab:  Runtime → Change runtime type → GPU (T4 / A100)")
    sys.exit(1)

_gpu_lines = [ln.strip() for ln in _smi_csv.splitlines() if ln.strip()]
print(f"  ✓  {len(_gpu_lines)} GPU(s) found:")
for _ln in _gpu_lines:
    print(f"       {_ln}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — CUDA TOOLKIT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
_step(2, "CUDA toolkit validation")

_, _nvcc_out = _capture("nvcc --version")
for _ln in _nvcc_out.splitlines():
    if "release" in _ln.lower():
        print(f"  nvcc:  {_ln.strip()}")
        break
else:
    print("  nvcc not on PATH (normal in Colab — using nvidia-smi for version).")

_, _smi_full = _capture("nvidia-smi")
_cuda_ver = "12.1"                        # safe fallback for recent Colab runtimes
_m = re.search(r"CUDA Version:\s*(\d+\.\d+)", _smi_full)
if _m:
    _cuda_ver = _m.group(1)
print(f"  CUDA version (driver): {_cuda_ver}")

_maj, _min = (int(x) for x in _cuda_ver.split(".")[:2])
if   (_maj, _min) >= (12, 4): _cu_tag = "cu124"
elif (_maj, _min) >= (12, 1): _cu_tag = "cu121"
elif (_maj, _min) >= (11, 8): _cu_tag = "cu118"
else:                          _cu_tag = "cu117"
print(f"  PyTorch wheel tag:     {_cu_tag}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — INSTALL CUDA-BACKED PYTORCH
# ═══════════════════════════════════════════════════════════════════════════════
_step(3, "Install PyTorch with CUDA support")

_torch_index = f"https://download.pytorch.org/whl/{_cu_tag}"
print(f"  Index URL: {_torch_index}")
_sh_ok(f"pip install -q torch torchvision torchaudio --index-url {_torch_index}")
print("  ✓  PyTorch installed.")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — INSTALL PROJECT DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════════
_step(4, "Install project dependencies")

# System: xvfb for headless PyVista rendering
_sh("apt-get install -qq -y xvfb 2>/dev/null")
print("  xvfb ready (headless 3D rendering).")

# Python packages (mirrors requirements.txt, excluding torch which is already done)
_pkgs = " ".join([
    "numpy>=1.26",
    "pandas>=2.2",
    "matplotlib>=3.8",
    "gymnasium>=0.29",
    "'stable-baselines3[extra]>=2.3.0'",
    "tensorboard>=2.16",
    "pillow>=10.0",
    "tqdm>=4.66",
    "'pyvista>=0.43'",
    "imageio>=2.34",
])
_sh_ok(f"pip install -q {_pkgs}")
print("  ✓  All project dependencies installed.")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — GPU WARMUP BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════
_step(5, "GPU warmup benchmark")

import torch  # noqa: E402  — available after pip install above

if not torch.cuda.is_available():
    print("  ✗  torch.cuda.is_available() = False after install.")
    print("     The CUDA wheel may not match the driver. Try updating the driver.")
    sys.exit(1)

_dev  = torch.device("cuda")
_n    = 4096
_a, _b = (torch.randn(_n, _n, device=_dev, dtype=torch.float32) for _ in range(2))

# Warmup (not timed)
torch.cuda.synchronize()
for _ in range(3):
    _c = torch.mm(_a, _b)
torch.cuda.synchronize()

# Timed benchmark
_REPS = 10
_tb0  = time.perf_counter()
for _ in range(_REPS):
    _c = torch.mm(_a, _b)
torch.cuda.synchronize()
_tb1 = time.perf_counter()

_tflops   = 2 * _n ** 3 * _REPS / (_tb1 - _tb0) / 1e12
_vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
_gpu_name = torch.cuda.get_device_name(0)

print(f"  GPU:         {_gpu_name}")
print(f"  VRAM:        {_vram_gb:.1f} GB")
print(f"  Throughput:  {_tflops:.2f} TFLOPS  (FP32 {_n}×{_n} matmul × {_REPS})")

for _k, _ref in {"T4": 8.1, "A100": 77.6, "V100": 14.0, "L4": 30.3, "P100": 9.3}.items():
    if _k in _gpu_name:
        print(f"  Reference {_k} peak FP32: {_ref} TFLOPS  → benchmark at {_tflops / _ref * 100:.0f}%")
        break

if TRAINING_SCALE == "FULL":
    print()
    if "T4" in _gpu_name:
        print("  Estimated wall time on T4:")
        print("    PPO 5 M steps : ~4–6 h")
        print("    SAC 3 M steps : ~2–3 h")
    elif "A100" in _gpu_name:
        print("  Estimated wall time on A100:")
        print("    PPO 5 M steps : ~1–2 h")
        print("    SAC 3 M steps : ~45–90 min")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — CLONE / PULL REPOSITORY
# ═══════════════════════════════════════════════════════════════════════════════
_step(6, "Clone / pull repository")

if os.path.isdir(REPO_DIR):
    _sh_ok(f"git -C {REPO_DIR} pull")
    print(f"  ✓  Repository updated:  {REPO_DIR}")
else:
    _sh_ok(f"git clone {REPO_URL} {REPO_DIR}")
    print(f"  ✓  Repository cloned:   {REPO_DIR}")

os.environ["PYTHONPATH"] = REPO_DIR
os.chdir(REPO_DIR)
print(f"  Working directory: {os.getcwd()}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — GOOGLE DRIVE MOUNT  (optional, requires interactive auth in Colab)
# ═══════════════════════════════════════════════════════════════════════════════
_step(7, "Google Drive persistence")

PPO_SAVE = PPO_LOG = SAC_SAVE = SAC_LOG = None   # resolved below

if USE_DRIVE:
    try:
        from google.colab import drive as _colab_drive
        _colab_drive.mount("/content/drive")
        _dm = f"{DRIVE_ROOT}/models"
        _dl = f"{DRIVE_ROOT}/logs"
        os.makedirs(_dm, exist_ok=True)
        os.makedirs(_dl, exist_ok=True)
        print(f"  ✓  Drive mounted.  Models → {_dm}")

        PPO_SAVE = f"{_dm}/ppo_pack_3d_multizone_sensor"
        SAC_SAVE = f"{_dm}/sac_pack_3d_multizone_sensor"
        PPO_LOG  = f"{_dl}/ppo_pack_3d_multizone_sensor"
        SAC_LOG  = f"{_dl}/sac_pack_3d_multizone_sensor"

        # Local symlinks so all project scripts resolve models at relative paths
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs",   exist_ok=True)
        for _local, _target in [
            ("models/ppo_pack_3d_multizone_sensor", PPO_SAVE),
            ("models/sac_pack_3d_multizone_sensor", SAC_SAVE),
            ("logs/ppo_pack_3d_multizone_sensor",   PPO_LOG),
            ("logs/sac_pack_3d_multizone_sensor",   SAC_LOG),
        ]:
            os.makedirs(_target, exist_ok=True)
            if os.path.islink(_local):
                os.remove(_local)
            if not os.path.exists(_local):
                os.symlink(_target, _local)
        print("  Symlinks created: models/* → Drive")

    except Exception as _e:
        print(f"  ⚠  Drive mount failed ({_e}) — falling back to local storage.")
        USE_DRIVE = False

if not USE_DRIVE:
    PPO_SAVE = "models/ppo_pack_3d_multizone_sensor"
    SAC_SAVE = "models/sac_pack_3d_multizone_sensor"
    PPO_LOG  = "logs/ppo_pack_3d_multizone_sensor"
    SAC_LOG  = "logs/sac_pack_3d_multizone_sensor"
    for _d in (PPO_SAVE, SAC_SAVE, PPO_LOG, SAC_LOG):
        os.makedirs(_d, exist_ok=True)
    print("  Models will be saved locally (lost when the Colab session ends).")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 — ENVIRONMENT VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
_step(8, "Environment verification")

_verify = subprocess.run(
    [
        sys.executable, "-c",
        (
            "from envs.battery_pack_thermal_env_3d import BatteryPackThermalEnv3D;"
            "from configs.pack_config import CellConfig, PackConfig;"
            "from training.train_pack_ppo_3d import make_sensor_config, make_actuator_config, make_pack_config;"
            "import torch;"
            "cfg = make_pack_config();"
            "env = BatteryPackThermalEnv3D("
            "    cell_config=CellConfig(), pack_config=cfg,"
            "    enable_sensor_simulation=True,"
            "    sensor_config=make_sensor_config(cfg),"
            "    actuator_config=make_actuator_config(cfg));"
            "obs, _ = env.reset(); env.close();"
            "assert obs.shape == (29,), f'Expected (29,), got {obs.shape}';"
            "assert torch.cuda.is_available(), 'CUDA not available';"
            "print(f'  Env OK  obs={obs.shape}  CUDA={torch.cuda.is_available()}"
            "  GPU={torch.cuda.get_device_name(0)}');"
        ),
    ],
    cwd=REPO_DIR,
    env={**os.environ, "PYTHONPATH": REPO_DIR},
)
if _verify.returncode != 0:
    print("  ✗  Environment verification failed. Check the error above and re-run.")
    sys.exit(1)
print("  ✓  All checks passed — environment is ready.\n")

print(f"  Training plan  ({TRAINING_SCALE}):")
print(f"    PPO  {PPO_STEPS:>12,} steps · {N_ENVS} envs · curriculum {PPO_S2:,} / {PPO_S3:,}")
print(f"    SAC  {SAC_STEPS:>12,} steps · curriculum {SAC_S2:,} / {SAC_S3:,}")
print(f"    Eval {EPISODES} episodes · safety shield enabled")
print(f"    PPO → {PPO_SAVE}")
print(f"    SAC → {SAC_SAVE}")

# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE STAGES
# ═══════════════════════════════════════════════════════════════════════════════

_PY = sys.executable

# ─── Stage 1: Baseline benchmark ──────────────────────────────────────────────
if RUN_BASELINES:
    _run_stage(
        "Baseline benchmark  (6 classical controllers × 4 heat profiles)",
        [_PY, "-m", "scripts.compare_pack_baselines_3d"],
    )

# ─── Stage 2: PPO training ────────────────────────────────────────────────────
# Curriculum:
#   Stage 1 (0 → PPO_S2):       perfect sensors, no noise, no delays
#   Stage 2 (PPO_S2 → PPO_S3):  light noise 0.10 °C, 2 s cooling delay
#   Stage 3 (PPO_S3 → end):     full realism — 0.20 °C noise, 2 s sensor delay,
#                                5 s cooling delay, sparse thermistors
if RUN_PPO:
    _run_stage(
        f"PPO training  ({PPO_STEPS:,} steps, curriculum, device=cuda)",
        [
            _PY, "training/train_pack_ppo_3d.py",
            "--timesteps",  str(PPO_STEPS),
            "--n-envs",     str(N_ENVS),
            "--device",     "cuda",
            "--curriculum",
            "--stage2-at",  str(PPO_S2),
            "--stage3-at",  str(PPO_S3),
            "--save-dir",   PPO_SAVE,
            "--log-dir",    PPO_LOG,
        ],
    )

# ─── Stage 3: SAC training ────────────────────────────────────────────────────
if RUN_SAC:
    _run_stage(
        f"SAC training  ({SAC_STEPS:,} steps, curriculum, device=cuda)",
        [
            _PY, "training/train_pack_sac_3d.py",
            "--timesteps",       str(SAC_STEPS),
            "--device",          "cuda",
            "--curriculum",
            "--stage2-at",       str(SAC_S2),
            "--stage3-at",       str(SAC_S3),
            "--buffer-size",     "1000000",
            "--learning-starts", "10000",
            "--save-dir",        SAC_SAVE,
            "--log-dir",         SAC_LOG,
        ],
    )

# ─── Stage 4: Full evaluation ─────────────────────────────────────────────────
# --shield adds PPO+Shield and SAC+Shield variants (Zone-PI override when
# T_max_meas ≥ 44 °C, full cooling when ≥ 44.5 °C).
if RUN_EVAL:
    _ppo_best = f"{PPO_SAVE}/best_model.zip"
    _sac_best = f"{SAC_SAVE}/best_model.zip"
    _eval_cmd = [
        _PY, "evaluation/compare_controllers.py",
        "--results-dir", "outputs/comparison",
        "--episodes",    str(EPISODES),
        "--shield",
    ]
    if os.path.exists(_ppo_best):
        _eval_cmd += ["--ppo-model", _ppo_best]
    if os.path.exists(_sac_best):
        _eval_cmd += ["--sac-model", _sac_best]

    _run_stage(
        "Full evaluation  (all controllers + safety-shielded RL variants)",
        _eval_cmd,
    )

# ─── Stage 5: 3D visualization ────────────────────────────────────────────────
if RUN_VIZ:
    # Start Xvfb for headless PyVista rendering (already installed in Step 4)
    _sh("Xvfb :99 -screen 0 1024x768x24 &")
    os.environ["DISPLAY"] = ":99"
    time.sleep(1)   # give Xvfb a moment to start

    _run_stage(
        "3D static screenshots  (PI / SAC / PPO side-by-side)",
        [
            _PY, "scripts/visualize_3d_controllers.py",
            "--output-dir", "outputs/3d_viz",
            "--profile",    "NonuniformStep",
        ],
    )

    _gif_cmd = [
        _PY, "-m", "scripts.animate_pack_3d_pyvista",
        "--no-cooling", "--constant-05", "--constant-1",
        "--bang-bang",  "--proportional", "--PI",
        "--profile", "NonuniformStep",
        "--stride",  "10",
        "--fps",     "12",
    ]
    if os.path.exists(f"{SAC_SAVE}/best_model.zip"):
        _gif_cmd.append("--SAC")
    if os.path.exists(f"{PPO_SAVE}/best_model.zip"):
        _gif_cmd.append("--PPO")

    _run_stage("GIF animations  (all controllers)", _gif_cmd)

# ─── Stage 6: Failure diagnosis ───────────────────────────────────────────────
if RUN_DIAGNOSIS:
    _ppo_ok = os.path.exists(f"{PPO_SAVE}/best_model.zip")
    _sac_ok = os.path.exists(f"{SAC_SAVE}/best_model.zip")
    if _ppo_ok or _sac_ok:
        _diag_cmd = [
            _PY, "scripts/diagnose_rl_failures.py",
            "--output-dir", "outputs/diagnosis",
            "--episodes",   "5",
        ]
        if _ppo_ok:
            _diag_cmd += ["--ppo-model", f"{PPO_SAVE}/best_model.zip"]
        if _sac_ok:
            _diag_cmd += ["--sac-model", f"{SAC_SAVE}/best_model.zip"]
        _run_stage(
            "Failure diagnosis  (per-episode plots + root-cause report)",
            _diag_cmd,
        )
    else:
        print("  [SKIP] Diagnosis — no RL model files found.")

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
_total = int(time.time() - _T_START)
_h, _rem = divmod(_total, 3600)
_m, _s   = divmod(_rem,   60)

_banner(f"Run complete  ({_h}h {_m}m {_s}s total)")

print("  Key outputs:")
print("    outputs/phase2_3d_baseline_*.{png,csv}  — baseline benchmark")
print("    outputs/comparison/                      — all-controller evaluation")
print("    outputs/3d_viz/                          — 3D PyVista screenshots")
print("    outputs/3d_pyvista_*.gif                 — GIF animations")
print("    outputs/diagnosis/                       — failure plots + report")
print(f"    {PPO_SAVE}/")
print(f"      best_model.zip · vec_normalize.pkl · checkpoints/")
print(f"    {SAC_SAVE}/")
print(f"      best_model.zip · sac_pack_final.zip · checkpoints/")

print()
print("  TensorBoard (run in a separate cell):")
print(f"    %load_ext tensorboard")
print(f"    %tensorboard --logdir {os.path.dirname(PPO_LOG)}")

if USE_DRIVE and os.path.isdir("/content/drive"):
    print()
    print("  Models persisted to Google Drive:")
    print(f"    {PPO_SAVE}/best_model.zip")
    print(f"    {SAC_SAVE}/best_model.zip")

print()
if FAILED_STAGES:
    print(f"  Failed stages ({len(FAILED_STAGES)}):")
    for _s in FAILED_STAGES:
        print(f"    ✗  {_s}")
    print()
    sys.exit(1)
else:
    print("  All stages passed.")
print()
