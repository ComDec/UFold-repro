#!/usr/bin/env bash
# Reviewer-facing evaluation script.
# Evaluates all 6 UFold benchmarks (Rivals, UniRNA-SS, bpRNA-1m, bpRNA-1m-new, ArchiveII, iPKnot)
# against the checkpoints in $2, using datasets in $1.
#
# Usage:
#   bash scripts/eval_all.sh <data_dir> <ckpt_dir> [gpu_ids]
#
# Example:
#   bash scripts/eval_all.sh ./data ./checkpoints
#   bash scripts/eval_all.sh ./data ./checkpoints 0,1,2
#
# The script launches 3 chains on 3 GPUs in parallel. Total wall time ~30 min.
# All results printed to logs/verify_*.log.

set -u
cd "$(dirname "$0")/.." || exit 1

DATA_DIR="${1:-./data}"
CKPT_DIR="${2:-./checkpoints}"
GPU_IDS="${3:-0,1,2}"

IFS=',' read -ra GPUS <<< "$GPU_IDS"
if [ "${#GPUS[@]}" -lt 3 ]; then
    echo "[eval_all] need at least 3 GPU ids, got: $GPU_IDS"
    exit 1
fi
GPU_A="${GPUS[0]}"   # longest (ArchiveII)
GPU_B="${GPUS[1]}"   # sequential chain
GPU_C="${GPUS[2]}"   # bpRNA-1m-new

# Prefer ufold-repro conda env if available, otherwise rely on current PATH.
PY="${UFOLD_PYTHON:-}"
if [ -z "$PY" ] && [ -x "$HOME/miniforge3/envs/ufold-repro/bin/python" ]; then
    PY="$HOME/miniforge3/envs/ufold-repro/bin/python"
elif [ -z "$PY" ]; then
    PY="python"
fi

echo "[eval_all] python    : $PY"
echo "[eval_all] data_dir  : $DATA_DIR"
echo "[eval_all] ckpt_dir  : $CKPT_DIR"
echo "[eval_all] gpus      : $GPU_A, $GPU_B, $GPU_C"

# Sanity check: CUDA available
if ! "$PY" -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" 2>/dev/null; then
    echo "[eval_all] ERROR: torch.cuda.is_available() is False"
    exit 2
fi

# Sanity check: checkpoints present
for f in rivals_ep99 unirna_ss_ep99 bprna1m_ep99 bprna1m_ep9 archiveII_ep99 ipknot_ep99; do
    if [ ! -f "${CKPT_DIR}/${f}.pt" ]; then
        echo "[eval_all] ERROR: missing checkpoint ${CKPT_DIR}/${f}.pt"
        exit 3
    fi
done

mkdir -p logs

# ---------- Chain A (longest, solo): ArchiveII ----------
(
    PYTHONUNBUFFERED=1 "$PY" eval_from_checkpoint.py \
        --gpu "$GPU_A" \
        --checkpoint "${CKPT_DIR}/archiveII_ep99.pt" \
        --test_file "${DATA_DIR}/mxfold2/archiveII.pkl" \
        > logs/verify_archiveII.log 2>&1
) &
PID_A=$!

# ---------- Chain B (sequential): iPKnot -> bpRNA-1m -> Rivals A -> Rivals B -> UniRNA-SS ----------
(
    PYTHONUNBUFFERED=1 "$PY" eval_from_checkpoint.py \
        --gpu "$GPU_B" \
        --checkpoint "${CKPT_DIR}/ipknot_ep99.pt" \
        --test_file "${DATA_DIR}/ipkont/bpRNA-PK-TS0-1K.pkl" \
        > logs/verify_ipknot.log 2>&1 && \
    PYTHONUNBUFFERED=1 "$PY" eval_from_checkpoint.py \
        --gpu "$GPU_B" \
        --checkpoint "${CKPT_DIR}/bprna1m_ep99.pt" \
        --test_file "${DATA_DIR}/mxfold2/TS0-canonicals.pkl" \
        > logs/verify_bprna1m.log 2>&1 && \
    PYTHONUNBUFFERED=1 "$PY" eval_from_checkpoint.py \
        --gpu "$GPU_B" \
        --checkpoint "${CKPT_DIR}/rivals_ep99.pt" \
        --test_file "${DATA_DIR}/rivals/TestSetA-addss.pkl" \
        > logs/verify_rivals_testA.log 2>&1 && \
    PYTHONUNBUFFERED=1 "$PY" eval_from_checkpoint.py \
        --gpu "$GPU_B" \
        --checkpoint "${CKPT_DIR}/rivals_ep99.pt" \
        --test_file "${DATA_DIR}/rivals/TestSetB-addss.pkl" \
        > logs/verify_rivals_testB.log 2>&1 && \
    PYTHONUNBUFFERED=1 "$PY" eval_from_checkpoint.py \
        --gpu "$GPU_B" \
        --checkpoint "${CKPT_DIR}/unirna_ss_ep99.pt" \
        --test_file "${DATA_DIR}/all_data_1024_0.75/test.pkl" \
        > logs/verify_unirna_ss.log 2>&1
) &
PID_B=$!

# ---------- Chain C (bpRNA-1m-new, uses ep9 for early stopping) ----------
(
    PYTHONUNBUFFERED=1 "$PY" eval_from_checkpoint.py \
        --gpu "$GPU_C" \
        --checkpoint "${CKPT_DIR}/bprna1m_ep9.pt" \
        --test_file "${DATA_DIR}/mxfold2/bpRNAnew.pkl" \
        > logs/verify_bprna1m_new.log 2>&1
) &
PID_C=$!

echo "[eval_all] launched: PIDs $PID_A $PID_B $PID_C"
echo "[eval_all] waiting for all chains to finish..."
wait $PID_A $PID_B $PID_C

echo
echo "[eval_all] ==================== SUMMARY ===================="
printf "%-20s %-40s %s\n" "Benchmark" "Log file" "F1 (expected)"
printf "%-20s %-40s %s\n" "---------" "--------" "-------------"
for entry in \
    "Rivals/TestSetA   logs/verify_rivals_testA.log   0.6343" \
    "Rivals/TestSetB   logs/verify_rivals_testB.log   0.4145" \
    "UniRNA-SS         logs/verify_unirna_ss.log      0.4394" \
    "bpRNA-1m          logs/verify_bprna1m.log        0.4653" \
    "bpRNA-1m-new      logs/verify_bprna1m_new.log    0.5387" \
    "ArchiveII         logs/verify_archiveII.log      0.6584" \
    "iPKnot            logs/verify_ipknot.log         0.4118"
do
    read -r name log expected <<< "$entry"
    f1_line=$(grep "^  f1 " "$log" 2>/dev/null | tail -1 | awk -F: '{gsub(/ /, "", $2); print $2}')
    printf "%-20s %-40s %-8s (expected %s)\n" "$name" "$log" "${f1_line:-FAIL}" "$expected"
done
