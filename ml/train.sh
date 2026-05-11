#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# CyclingTransformer pre-training pipeline — Linux / CUDA
#
# Run from the project root:
#   bash ml/train.sh                           # full run: 50k athletes, 100 epochs
#   bash ml/train.sh --small-test              # smoke test: 500 athletes, 20 epochs
#   bash ml/train.sh --athletes 15000 --fast   # quick CPU/GPU test with small model
#   bash ml/train.sh --reuse                   # skip data regen if parquet exists
#   bash ml/train.sh --athletes 50000 --batch-size 256 --compile  # full CUDA run
#
# Environment prerequisites (see ml/requirements-train.txt):
#   pip install -r ml/requirements-train.txt
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
ATHLETES=50000
EPOCHS=0          # 0 = pick per-mode default
BATCH_SIZE=256    # larger batch = better GPU utilization; reduce if OOM
FAST=false
SMALL_TEST=false
REUSE=false
COMPILE=false
NO_AMP=false

# ── Parse args ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --athletes)    ATHLETES="$2";    shift 2 ;;
    --epochs)      EPOCHS="$2";      shift 2 ;;
    --batch-size)  BATCH_SIZE="$2";  shift 2 ;;
    --fast)        FAST=true;        shift   ;;
    --small-test)  SMALL_TEST=true;  shift   ;;
    --reuse)       REUSE=true;       shift   ;;
    --compile)     COMPILE=true;     shift   ;;
    --no-amp)      NO_AMP=true;      shift   ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── Environment ───────────────────────────────────────────────────────────────
# backend package must be importable by both generator and trainer
export PYTHONPATH="$(pwd)/backend"
export PYTHONIOENCODING="utf-8"

# Let PyTorch use all CPU cores for data-loading workers
N_CPU=$(nproc --all 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
export OMP_NUM_THREADS=$N_CPU
export MKL_NUM_THREADS=$N_CPU

mkdir -p backend/models ml/data

# ── Mode-specific defaults ────────────────────────────────────────────────────
if $SMALL_TEST; then
    DATA_FILE="./ml/data/synthetic_small.parquet"
    MODEL_FILE="./backend/models/cycling_coach_test.pt"
    [[ "$EPOCHS" -eq 0 ]] && EPOCHS=20
    ATHLETES=500
    WEEKS_ARG="--weeks 26"
    FAST=true
    MODE_LABEL="SMOKE TEST  (500 athletes, 26 weeks)"
else
    DATA_FILE="./ml/data/synthetic.parquet"
    MODEL_FILE="./backend/models/cycling_coach.pt"
    [[ "$EPOCHS" -eq 0 ]] && EPOCHS=100
    WEEKS_ARG=""
    MODE_LABEL="FULL PRE-TRAINING ($ATHLETES athletes)"
fi

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "=== $MODE_LABEL ==="
echo "  Data file  : $DATA_FILE"
echo "  Model file : $MODEL_FILE"
echo "  Epochs     : $EPOCHS"
echo "  Batch size : $BATCH_SIZE"
echo "  Model size : $(  $FAST && echo 'FAST (d=64, 4 layers, ~221k params)' || echo 'FULL (d=128, 6 layers, ~1M params)'  )"
echo "  CPU cores  : $N_CPU"
echo ""

# ── Step 1: Generate synthetic data (or reuse) ────────────────────────────────
if $REUSE && [[ -f "$DATA_FILE" ]]; then
    SIZE_MB=$(du -m "$DATA_FILE" | cut -f1)
    echo "[1/2] Reusing existing $DATA_FILE (${SIZE_MB} MB)"
else
    echo "[1/2] Generating synthetic data..."
    python -m ml.training.generate_synthetic \
        --athletes "$ATHLETES" \
        --output   "$DATA_FILE" \
        $WEEKS_ARG
fi

echo ""
echo "[2/2] Training model..."

# ── Build train.py arg list ───────────────────────────────────────────────────
TRAIN_ARGS=(
    -m ml.training.train
    --data       "$DATA_FILE"
    --output     "$MODEL_FILE"
    --epochs     "$EPOCHS"
    --batch-size "$BATCH_SIZE"
)
$FAST    && TRAIN_ARGS+=(--fast)
$COMPILE && TRAIN_ARGS+=(--compile)
$NO_AMP  && TRAIN_ARGS+=(--no-amp)

python "${TRAIN_ARGS[@]}"

echo ""
echo "=== DONE ==="
echo "Model saved to: $MODEL_FILE"
echo "Copy it back to the server: scp $MODEL_FILE user@your-server:~/ai-coach/backend/models/"
