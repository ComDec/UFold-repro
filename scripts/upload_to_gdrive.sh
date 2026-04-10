#!/usr/bin/env bash
# Upload datasets + code repo + checkpoints to Google Drive via rclone.
#
# Targets:
#   Datasets      -> gdrive_xw:UniRNA/ss_dataset/
#   Code repo     -> gdrive_xw:UniRNA/baselines/UFold/repo/
#   Checkpoints   -> gdrive_xw:UniRNA/baselines/UFold/checkpoints/
#
# Rivals: only the addss (contrafold) and eternafold variants (6 files).
# mxfold2: only the 6 pkl files actually used by any reproduction run.

set -eu
cd "$(dirname "$0")/.." || exit 1

DATA_SRC="/home/xiwang/project/develop/data"
REMOTE="gdrive_xw:UniRNA"

log() { echo "[upload] $(date '+%H:%M:%S') $*"; }

log "=== datasets: all_data_1024_0.75 (UniRNA-SS, 3.4 GB) ==="
rclone copy "$DATA_SRC/all_data_1024_0.75/" "$REMOTE/ss_dataset/all_data_1024_0.75/" \
    --include "{train,valid,test}.pkl" \
    --transfers 4 --checkers 8 --progress --stats 15s

log "=== datasets: ipkont (iPKnot, 3.0 GB) ==="
rclone copy "$DATA_SRC/ipkont/" "$REMOTE/ss_dataset/ipkont/" \
    --include "{bpRNA-TR0,bpRNA-PK-TS0-1K}.pkl" \
    --transfers 4 --checkers 8 --progress --stats 15s

log "=== datasets: rivals (6 files, ~6 GB) ==="
rclone copy "$DATA_SRC/rivals/" "$REMOTE/ss_dataset/rivals/" \
    --include "{TrainSetA,TestSetA,TestSetB}-{addss,eternafold}.pkl" \
    --transfers 4 --checkers 8 --progress --stats 15s

log "=== datasets: mxfold2 (6 files, ~11 GB) ==="
rclone copy "$DATA_SRC/mxfold2/" "$REMOTE/ss_dataset/mxfold2/" \
    --include "{TR0-canonicals,VL0-canonicals,TS0-canonicals,RNAStrAlign600-train,archiveII,bpRNAnew}.pkl" \
    --transfers 4 --checkers 8 --progress --stats 15s

log "=== checkpoints (204 MB) ==="
rclone copy "./checkpoints/" "$REMOTE/baselines/UFold/checkpoints/" \
    --transfers 6 --checkers 8 --progress --stats 15s

log "=== code repo (excluding models_*/, logs/, docs/superpowers/, checkpoints/, .git/) ==="
rclone copy "." "$REMOTE/baselines/UFold/repo/" \
    --exclude ".git/**" \
    --exclude "models_*/**" \
    --exclude "logs/**" \
    --exclude "checkpoints/**" \
    --exclude "docs/superpowers/**" \
    --exclude "**/__pycache__/**" \
    --exclude "*.pyc" \
    --exclude ".ruff_cache/**" \
    --exclude ".worktrees/**" \
    --exclude "*.log" \
    --transfers 6 --checkers 8 --progress --stats 15s

log "=== DONE ==="
log "Verifying remote tree..."
rclone lsf "$REMOTE/ss_dataset/" --dirs-only
rclone lsf "$REMOTE/baselines/UFold/" --dirs-only

log "Remote file counts:"
for subdir in ss_dataset/all_data_1024_0.75 ss_dataset/ipkont ss_dataset/rivals ss_dataset/mxfold2 baselines/UFold/checkpoints baselines/UFold/repo; do
    n=$(rclone ls "$REMOTE/$subdir/" 2>/dev/null | wc -l)
    log "  $subdir : $n files"
done
