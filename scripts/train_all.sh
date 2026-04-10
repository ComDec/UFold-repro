#!/usr/bin/env bash
# Reviewer-facing training script.
# Trains UFold on all 6 benchmarks from scratch.
# Each training run is 100 epochs; expected wall time per run:
#   Rivals:       ~1.5 h    (smallest)
#   UniRNA-SS:    ~5 h
#   bpRNA-1m:     ~6 h
#   ArchiveII:    ~12 h     (largest training set)
#   iPKnot:       ~5 h
# Total: ~30 h single-GPU sequential, or ~12 h with 3 GPU parallelism.
#
# Note: bpRNA-1m-new uses the same model as bpRNA-1m (no separate training).
#
# Usage:
#   bash scripts/train_all.sh <data_dir> <out_dir> [gpu_id]
#
# Example:
#   bash scripts/train_all.sh ./data ./models_retrain 0

set -u
cd "$(dirname "$0")/.." || exit 1

DATA_DIR="${1:-./data}"
OUT_DIR="${2:-./models_retrain}"
GPU="${3:-0}"

PY="${UFOLD_PYTHON:-}"
if [ -z "$PY" ] && [ -x "$HOME/miniforge3/envs/ufold-repro/bin/python" ]; then
    PY="$HOME/miniforge3/envs/ufold-repro/bin/python"
elif [ -z "$PY" ]; then
    PY="python"
fi

echo "[train_all] python   : $PY"
echo "[train_all] data_dir : $DATA_DIR"
echo "[train_all] out_dir  : $OUT_DIR"
echo "[train_all] gpu      : $GPU"

if ! "$PY" -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" 2>/dev/null; then
    echo "[train_all] ERROR: torch.cuda.is_available() is False"
    exit 2
fi

mkdir -p logs "$OUT_DIR"

run_one() {
    local name="$1" data_subdir="$2" train_file="$3" val_file="$4" test_files="$5"
    local save="${OUT_DIR}/${name}"
    mkdir -p "$save"
    local cmd=( PYTHONUNBUFFERED=1 "$PY" ufold_train_rivals.py
                --gpu "$GPU" --data_dir "${DATA_DIR}/${data_subdir}"
                --save_dir "$save" --train_file "$train_file"
                --test_files $test_files )
    if [ -n "$val_file" ]; then
        cmd+=( --val_file "$val_file" )
    fi
    echo "[train_all] training $name ..."
    "${cmd[@]}" > "logs/train_${name}.log" 2>&1
    echo "[train_all] $name done -> $save"
}

# ---------- 1. Rivals ----------
run_one rivals    rivals         TrainSetA-addss.pkl  "" \
    "TestSetA-addss.pkl TestSetB-addss.pkl"

# ---------- 2. UniRNA-SS ----------
run_one unirna_ss all_data_1024_0.75 train.pkl valid.pkl "test.pkl"

# ---------- 3. bpRNA-1m (also provides model for bpRNA-1m-new) ----------
run_one bprna1m   mxfold2        TR0-canonicals.pkl   VL0-canonicals.pkl \
    "TS0-canonicals.pkl"

# ---------- 4. ArchiveII ----------
run_one archiveII mxfold2        RNAStrAlign600-train.pkl "" \
    "archiveII.pkl"

# ---------- 5. iPKnot ----------
run_one ipknot    ipkont         bpRNA-TR0.pkl        "" \
    "bpRNA-PK-TS0-1K.pkl"

# ---------- 6. bpRNA-1m-new: inference-only using bpRNA-1m epoch 9 ----------
echo "[train_all] bpRNA-1m-new: using early-stopped epoch 9 checkpoint from bpRNA-1m ..."
"$PY" eval_from_checkpoint.py \
    --gpu "$GPU" \
    --checkpoint "${OUT_DIR}/bprna1m/ufold_train_rivals_9.pt" \
    --test_file "${DATA_DIR}/mxfold2/bpRNAnew.pkl" \
    > logs/eval_bprna1m_new.log 2>&1
echo "[train_all] bpRNA-1m-new done -> logs/eval_bprna1m_new.log"

echo "[train_all] ALL DONE. Checkpoints in $OUT_DIR, logs in logs/"
