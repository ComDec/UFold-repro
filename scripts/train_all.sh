#!/usr/bin/env bash
# Train UFold on all 5 benchmarks from scratch.
# Total: ~30 h single-GPU sequential.
#
# bpRNA-1m-new uses the same checkpoint as bpRNA-1m (epoch 99).
#
# Usage:
#   bash scripts/train_all.sh <data_dir> <out_dir> [gpu_id]

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

# 1. Rivals
run_one rivals    rivals         TrainSetA-addss.pkl  "" \
    "TestSetA-addss.pkl TestSetB-addss.pkl"

# 2. UniRNA-SS
run_one unirna_ss all_data_1024_0.75 train.pkl valid.pkl "test.pkl"

# 3. bpRNA-1m (checkpoint also used for bpRNA-1m-new)
run_one bprna1m   mxfold2        TR0-canonicals.pkl   VL0-canonicals.pkl \
    "bpRNAnew.pkl"

# 4. ArchiveII
run_one archiveII mxfold2        RNAStrAlign600-train.pkl "" \
    "archiveII.pkl"

# 5. iPKnot
run_one ipknot    ipkont         bpRNA-TR0.pkl        "" \
    "bpRNA-PK-TS0-1K.pkl"

echo "[train_all] ALL DONE. Checkpoints in $OUT_DIR, logs in logs/"
