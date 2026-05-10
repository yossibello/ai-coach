#!/usr/bin/env bash
# ============================================================
#  CyclingTransformer — Linux pre-training pipeline
#  Optimised for multi-core servers (e.g. 64-CPU cloud node)
#
#  Usage:
#    ./ml/pretrain.sh [options]
#
#  Options:
#    --athletes N       Number of synthetic athletes (default 200000)
#    --weeks N          Weeks of history per athlete (default 104)
#    --epochs N         Training epochs (default 100)
#    --batch-size N     Mini-batch size (default 256)
#    --threads N        CPU threads for PyTorch (default: nproc)
#    --fast             Use small model d=64 4-layer (for smoke testing)
#    --checkpoint PATH  Resume / fine-tune from checkpoint
#    --reuse            Skip data generation if parquet already exists
#    --smoke            Alias for: --athletes 500 --weeks 26 --epochs 20 --fast
#
#  Run from repo root:
#    cd /path/to/ai-coach
#    chmod +x ml/pretrain.sh
#    ./ml/pretrain.sh
#    ./ml/pretrain.sh --smoke          # quick validation test
#    ./ml/pretrain.sh --athletes 50000 # medium run
# ============================================================
set -euo pipefail

# ── Defaults ────────────────────────────────────────────────
ATHLETES=200000
WEEKS=104
EPOCHS=100
BATCH_SIZE=256
THREADS=$(nproc)
FAST=""
CHECKPOINT=""
REUSE=false
DATA_FILE="./ml/data/synthetic.parquet"
MODEL_FILE="./backend/models/cycling_coach.pt"

# ── Argument parsing ─────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --athletes)   ATHLETES=$2; shift 2 ;;
    --weeks)      WEEKS=$2; shift 2 ;;
    --epochs)     EPOCHS=$2; shift 2 ;;
    --batch-size) BATCH_SIZE=$2; shift 2 ;;
    --threads)    THREADS=$2; shift 2 ;;
    --fast)       FAST="--fast"; shift ;;
    --checkpoint) CHECKPOINT="--checkpoint $2"; shift 2 ;;
    --reuse)      REUSE=true; shift ;;
    --smoke)
      ATHLETES=500; WEEKS=26; EPOCHS=20; FAST="--fast"
      DATA_FILE="./ml/data/synthetic_small.parquet"
      MODEL_FILE="./backend/models/cycling_coach_test.pt"
      shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Environment ──────────────────────────────────────────────
export PYTHONPATH="${PWD}/backend"
export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export PYTHONIOENCODING=utf-8

# Activate virtualenv if present
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [[ -f "venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

PYTHON=$(command -v python3 || command -v python)

# ── Banner ───────────────────────────────────────────────────
echo ""
echo "=== CyclingTransformer Pre-training ==="
if [[ -n "$FAST" ]]; then
  echo "  MODE       : FAST (d=64, 4 layers — smoke test only)"
else
  echo "  MODE       : FULL (d=128, 6 layers — production)"
fi
echo "  Athletes   : $ATHLETES"
echo "  Weeks      : $WEEKS"
echo "  Epochs     : $EPOCHS"
echo "  Batch size : $BATCH_SIZE"
echo "  Threads    : $THREADS"
echo "  Data file  : $DATA_FILE"
echo "  Model file : $MODEL_FILE"
echo ""

mkdir -p "$(dirname "$DATA_FILE")" "$(dirname "$MODEL_FILE")"

# ── Step 1: Generate synthetic data ──────────────────────────
if [[ "$REUSE" == true && -f "$DATA_FILE" ]]; then
  echo "[1/2] Reusing existing data: $DATA_FILE"
else
  echo "[1/2] Generating synthetic data..."
  $PYTHON -m ml.training.generate_synthetic \
    --athletes "$ATHLETES" \
    --weeks    "$WEEKS" \
    --output   "$DATA_FILE"
fi

# ── Step 2: Train model ──────────────────────────────────────
echo "[2/2] Training model..."
# shellcheck disable=SC2086
$PYTHON -m ml.training.train \
  --data       "$DATA_FILE" \
  --output     "$MODEL_FILE" \
  --epochs     "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  $FAST \
  $CHECKPOINT

echo ""
echo "Done. Model saved to: $MODEL_FILE"
