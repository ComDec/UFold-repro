#!/usr/bin/env bash
# Verification launcher for UFold checkpoints.
# Runs all 7 benchmark evaluations on 3 GPUs, logs to logs/verify_*.log.
# Designed to be called via `! bash scripts/run_verification.sh` from Claude Code
# so it inherits the user's interactive shell (with GPU access).

set -u
cd "$(dirname "$0")/.." || exit 1

# Use the ufold-repro conda env (per environment.yml — python 3.11 + torch 2.0.1 + cu118).
# This is the exact env that produced every Run 1-7 result documented in REPRODUCTION.md.
CONDA_BASE="/home/xiwang/miniforge3"
ENV_NAME="ufold-repro"
PY="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"

if [ ! -x "$PY" ]; then
    echo "[verify] ERROR: python not found at $PY"
    echo "[verify] Create the env first:  conda env create -f environment.yml"
    exit 2
fi

# Activate for child processes (sets LD_LIBRARY_PATH, PATH, CONDA_PREFIX, etc.).
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Sanity check: CUDA must be visible. If not, abort before launching chains.
if ! "$PY" -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" 2>/dev/null; then
    echo "[verify] ERROR: torch.cuda.is_available() is False in env '$ENV_NAME'"
    echo "[verify] Run directly to see the traceback:"
    echo "           $PY -c 'import torch; print(torch.cuda.is_available())'"
    exit 2
fi
echo "[verify] CUDA OK in env '$ENV_NAME' (python $($PY --version 2>&1 | cut -d' ' -f2), torch $($PY -c 'import torch; print(torch.__version__)'))"

mkdir -p logs
: > logs/verify.pids

# NOTE: UniRNA-SS was already verified in an earlier run (see logs/verify_unirna_ss.log).
# This script re-runs the remaining 6 benchmarks on GPUs 0, 1, 2.

# ---------- Chain 1: GPU 0 ----------
# ArchiveII (ep99) — expected F1 = 0.6584  (longest job, ~30 min)
(
    PYTHONUNBUFFERED=1 "$PY" run_exp.py exp35 eval_from_checkpoint.py \
        --gpu 0 \
        --checkpoint models_archiveII/ufold_train_rivals_99.pt \
        --test_file /home/xiwang/project/develop/data/mxfold2/archiveII.pkl \
        > logs/verify_archiveII.log 2>&1
    echo "[verify] chain1 done at $(date)" >> logs/verify.pids
) &
CHAIN1_PID=$!
echo "chain1_pid=$CHAIN1_PID" >> logs/verify.pids

# ---------- Chain 2: GPU 1 (sequential: iPKnot -> bpRNA-1m -> Rivals A -> Rivals B) ----------
(
    PYTHONUNBUFFERED=1 "$PY" run_exp.py exp36 eval_from_checkpoint.py \
        --gpu 1 \
        --checkpoint models_ipknot/ufold_train_rivals_99.pt \
        --test_file /home/xiwang/project/develop/data/ipkont/bpRNA-PK-TS0-1K.pkl \
        > logs/verify_ipknot.log 2>&1 && \
    PYTHONUNBUFFERED=1 "$PY" run_exp.py exp39 eval_from_checkpoint.py \
        --gpu 1 \
        --checkpoint models_bprna1m/ufold_train_rivals_99.pt \
        --test_file /home/xiwang/project/develop/data/mxfold2/TS0-canonicals.pkl \
        > logs/verify_bprna1m.log 2>&1 && \
    PYTHONUNBUFFERED=1 "$PY" run_exp.py exp40 eval_from_checkpoint.py \
        --gpu 1 \
        --checkpoint models_rivals/ufold_train_rivals_99.pt \
        --test_file /home/xiwang/project/develop/data/rivals/TestSetA-addss.pkl \
        > logs/verify_rivals_testA.log 2>&1 && \
    PYTHONUNBUFFERED=1 "$PY" run_exp.py exp41 eval_from_checkpoint.py \
        --gpu 1 \
        --checkpoint models_rivals/ufold_train_rivals_99.pt \
        --test_file /home/xiwang/project/develop/data/rivals/TestSetB-addss.pkl \
        > logs/verify_rivals_testB.log 2>&1
    echo "[verify] chain2 done at $(date)" >> logs/verify.pids
) &
CHAIN2_PID=$!
echo "chain2_pid=$CHAIN2_PID" >> logs/verify.pids

# ---------- Chain 3: GPU 2 (bpRNA-1m-new only; UniRNA-SS already verified) ----------
(
    PYTHONUNBUFFERED=1 "$PY" run_exp.py exp38 eval_from_checkpoint.py \
        --gpu 2 \
        --checkpoint models_bprna1m/ufold_train_rivals_9.pt \
        --test_file /home/xiwang/project/develop/data/mxfold2/bpRNAnew.pkl \
        > logs/verify_bprna1m_new.log 2>&1
    echo "[verify] chain3 done at $(date)" >> logs/verify.pids
) &
CHAIN3_PID=$!
echo "chain3_pid=$CHAIN3_PID" >> logs/verify.pids

disown -a 2>/dev/null || true

echo "[verify] launched:"
echo "  chain1 (GPU 0, ArchiveII)                        pid=$CHAIN1_PID  log=logs/verify_archiveII.log"
echo "  chain2 (GPU 1, iPKnot->bpRNA-1m->Rivals A->B)    pid=$CHAIN2_PID  logs=logs/verify_{ipknot,bprna1m,rivals_testA,rivals_testB}.log"
echo "  chain3 (GPU 2, bpRNA-1m-new)                     pid=$CHAIN3_PID  log=logs/verify_bprna1m_new.log"
echo "[verify] Expected wall time: ~30 min. Jobs are backgrounded and will survive this shell exiting."
