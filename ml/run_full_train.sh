#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# CyclingTransformer — complete end-to-end CUDA training pipeline
# Tuned for RTX 4090 (24 GB) and RTX 5090 (32 GB)
#
# What this script does, in order:
#   1. Pull latest code from GitHub
#   2. Create / update Python venv
#   3. Install CUDA PyTorch + training deps
#   4. Verify GPU is detected
#   5. Generate synthetic training data (50k athletes)
#   6. Train the model with AMP + torch.compile
#   7. Verify the saved checkpoint
#   8. (Optional) push model back to git
#
# Usage:
#   bash ml/run_full_train.sh                  # full run (default)
#   bash ml/run_full_train.sh --small-test     # smoke test, ~5 min
#   bash ml/run_full_train.sh --reuse          # skip data regen
#   bash ml/run_full_train.sh --push           # git-push model after training
#   bash ml/run_full_train.sh --no-pull        # skip git pull (offline)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Arg defaults ──────────────────────────────────────────────────────────────
SMALL_TEST=false
REUSE=false
DO_PUSH=false
NO_PULL=false
ATHLETES=50000
EPOCHS=100

while [[ $# -gt 0 ]]; do
  case "$1" in
    --small-test) SMALL_TEST=true; shift ;;
    --reuse)      REUSE=true;      shift ;;
    --push)       DO_PUSH=true;    shift ;;
    --no-pull)    NO_PULL=true;    shift ;;
    --athletes)   ATHLETES="$2";   shift 2 ;;
    --epochs)     EPOCHS="$2";     shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── Colors ────────────────────────────────────────────────────────────────────
CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
step() { echo -e "\n${CYAN}[$(date '+%H:%M:%S')] ── $* ──${NC}"; }
ok()   { echo -e "${GREEN}✓ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠ $*${NC}"; }
die()  { echo -e "${RED}✗ $*${NC}"; exit 1; }

# ─────────────────────────────────────────────────────────────────────────────
# RTX 4090 / 5090 tuned constants
# ─────────────────────────────────────────────────────────────────────────────
# Model ~1M params, seq_len=90, input_dim=57, fp16 AMP
#   4090 (24 GB VRAM): batch 512 uses ~3–4 GB → 6× headroom
#   5090 (32 GB VRAM): batch 512 uses ~3–4 GB → 8× headroom
BATCH_SIZE=512

# DataLoader workers — IO is fast with parquet; 8 workers keeps GPU fed
DATALOADER_WORKERS=8

# torch.compile — first epoch takes ~90 s to compile, then ~15–20% faster
USE_COMPILE=true

# AMP (fp16) — mandatory on consumer GPUs for training speed
# Set NO_AMP=true only if you hit numerical instability (very rare)
NO_AMP=false

# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     CyclingTransformer — Full CUDA Training Pipeline         ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo "  Project root : $PROJECT_ROOT"
echo "  Started at   : $(date)"

# ── Step 1: Git pull ──────────────────────────────────────────────────────────
step "Step 1/7 — Git pull"
if $NO_PULL; then
    warn "Skipping git pull (--no-pull)"
else
    git pull origin main
    ok "Repo up to date: $(git log --oneline -1)"
fi

# ── Step 2: Virtual environment ───────────────────────────────────────────────
step "Step 2/7 — Python venv"
VENV_DIR="$PROJECT_ROOT/.venv-train"
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
    ok "Created venv at $VENV_DIR"
else
    ok "Reusing existing venv at $VENV_DIR"
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip --quiet

# ── Step 3: Install CUDA deps ─────────────────────────────────────────────────
step "Step 3/7 — Install / verify CUDA PyTorch + training deps"
pip install -r ml/requirements-train.txt --quiet
ok "Dependencies installed"

# ── Step 4: GPU sanity check ──────────────────────────────────────────────────
step "Step 4/7 — GPU verification"
GPU_INFO=$(python - <<'PYEOF'
import torch, sys
if not torch.cuda.is_available():
    print("NO_CUDA")
    sys.exit(0)
props = torch.cuda.get_device_properties(0)
vram_gb = props.total_memory / 1e9
cc = f"{props.major}.{props.minor}"
print(f"{props.name} | VRAM: {vram_gb:.1f} GB | Compute: {cc} | CUDA: {torch.version.cuda}")
PYEOF
)

if [[ "$GPU_INFO" == "NO_CUDA" ]]; then
    die "No CUDA GPU found. Check: nvidia-smi. Training on CPU would take days — aborting."
fi

echo "  GPU: $GPU_INFO"

# Auto-tune batch size for large VRAM (5090 has 32 GB)
VRAM_GB=$(python -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory / 1e9))")
if (( VRAM_GB >= 28 )); then
    BATCH_SIZE=1024
    DATALOADER_WORKERS=12
    warn "Detected ≥28 GB VRAM (RTX 5090?) → bumping batch to $BATCH_SIZE, workers to $DATALOADER_WORKERS"
fi
ok "GPU check passed"

# ── Environment for training ──────────────────────────────────────────────────
export PYTHONPATH="$PROJECT_ROOT/backend"
export PYTHONIOENCODING="utf-8"
export DATALOADER_WORKERS="$DATALOADER_WORKERS"

N_CPU=$(nproc --all 2>/dev/null || echo 8)
export OMP_NUM_THREADS=$N_CPU
export MKL_NUM_THREADS=$N_CPU

mkdir -p backend/models ml/data

# ── Step 5: Generate synthetic data ───────────────────────────────────────────
step "Step 5/7 — Synthetic data generation"

if $SMALL_TEST; then
    DATA_FILE="./ml/data/synthetic_small.parquet"
    MODEL_FILE="./backend/models/cycling_coach_test.pt"
    ATHLETES=500
    EPOCHS=20
    USE_COMPILE=false   # compile overhead not worth it for a smoke test
    GEN_ARGS=(--athletes 500 --weeks 26 --output "$DATA_FILE")
    MODE="SMOKE TEST"
else
    DATA_FILE="./ml/data/synthetic.parquet"
    MODEL_FILE="./backend/models/cycling_coach.pt"
    GEN_ARGS=(--athletes "$ATHLETES" --output "$DATA_FILE")
    MODE="FULL TRAINING"
fi

echo "  Mode     : $MODE"
echo "  Athletes : $ATHLETES"
echo "  Data     : $DATA_FILE"
echo "  Model    : $MODEL_FILE"
echo "  Batch    : $BATCH_SIZE"
echo "  Epochs   : $EPOCHS"
echo "  Compile  : $USE_COMPILE"
echo ""

if $REUSE && [[ -f "$DATA_FILE" ]]; then
    SIZE_MB=$(du -m "$DATA_FILE" | cut -f1)
    warn "Reusing existing $DATA_FILE (${SIZE_MB} MB) — pass without --reuse to regenerate"
else
    T0=$SECONDS
    python -m ml.training.generate_synthetic "${GEN_ARGS[@]}"
    ok "Data generated in $(( SECONDS - T0 ))s"
fi

# ── Step 6: Train ─────────────────────────────────────────────────────────────
step "Step 6/7 — Training (this is the long part)"

TRAIN_ARGS=(
    -m ml.training.train
    --data       "$DATA_FILE"
    --output     "$MODEL_FILE"
    --epochs     "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --patience   20            # early stop if val doesn't improve for 20 epochs
)
$USE_COMPILE && TRAIN_ARGS+=(--compile)
$NO_AMP      && TRAIN_ARGS+=(--no-amp)

T0=$SECONDS
python "${TRAIN_ARGS[@]}"
TRAIN_SEC=$(( SECONDS - T0 ))
ok "Training complete in $(( TRAIN_SEC / 60 ))m $(( TRAIN_SEC % 60 ))s"

# ── Step 7: Verify checkpoint ─────────────────────────────────────────────────
step "Step 7/7 — Verify checkpoint"
python - "$MODEL_FILE" <<'PYEOF'
import sys, torch
path = sys.argv[1]
ckpt = torch.load(path, map_location="cpu", weights_only=False)
cfg  = ckpt.get("config", {})
met  = ckpt.get("metrics", {})
print(f"  input_dim  : {cfg.get('input_dim')}   (expect 57 for full health-feature model)")
print(f"  d_model    : {cfg.get('d_model')}")
print(f"  num_layers : {cfg.get('num_layers')}")
print(f"  saved epoch: {met.get('epoch')}")
print(f"  val_loss   : {met.get('val_loss', 'n/a'):.4f}" if met.get('val_loss') else f"  val_loss   : n/a")
print(f"  wt_acc     : {met.get('wt_acc', 0):.1f}%")
print(f"  FTP MAE    : {met.get('ftp_mae', 0):.1f} W")
n_params = sum(v.numel() for v in ckpt["state_dict"].values())
print(f"  parameters : {n_params:,}")
PYEOF
ok "Checkpoint looks good: $MODEL_FILE"

# ── Optional: push model to git ───────────────────────────────────────────────
if $DO_PUSH; then
    step "Pushing model to GitHub"
    git add -f "$MODEL_FILE"
    git commit -m "chore: retrained model (57-feature schema, health signals, $(date +'%Y-%m-%d'))"
    git push origin main
    ok "Model pushed to GitHub"
else
    echo ""
    echo -e "${YELLOW}To copy the model back to your dev machine:${NC}"
    echo "  scp $MODEL_FILE user@dev-machine:~/ai-coach/backend/models/"
    echo ""
    echo -e "${YELLOW}Or push to git from this machine:${NC}"
    echo "  bash ml/run_full_train.sh --reuse --push"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                      ALL DONE                               ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo "  Model   : $MODEL_FILE"
echo "  Finished: $(date)"
