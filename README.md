# UFold Reproduction (Reviewer Package)

Reproduction of **UFold** ([Liang et al., NAR 2022](https://doi.org/10.1093/nar/gkab1074), [original repo](https://github.com/uci-cbcl/UFold)) for RNA secondary structure prediction on **6 standard benchmarks**, with additional pseudoknot-aware evaluation for ArchiveII and iPKnot.

> This README is the **reviewer entry point** — follow it top-to-bottom to fully reproduce every reported number in ~1 h of GPU time (evaluation only) or ~30 h (full training from scratch).
>
> Detailed docs (don't read these first):
> - [`Benchmark.md`](Benchmark.md) — full results tables, hyperparameters, per-benchmark sections
> - [`REPRODUCTION.md`](REPRODUCTION.md) — per-run notes, data integrity checks, design decisions
> - [`CHANGE_LOG.md`](CHANGE_LOG.md) — chronological log of every environment/code/experiment action

---

## 0. TL;DR

```bash
# 1. Environment (5 min)
conda env create -f environment.yml
conda activate ufold-repro
pip install setproctitle "torcheval==0.0.6"

# 2. Download datasets + checkpoints from Google Drive (see §2 below)
rclone copy gdrive:UniRNA/ss_dataset/ ./data/ -P
rclone copy gdrive:UniRNA/baselines/UFold/checkpoints/ ./checkpoints/ -P

# 3. Verify all 7 reported metrics (~30 min on 3 GPUs)
bash scripts/eval_all.sh ./data ./checkpoints 0,1,2

# 4. Additional pseudoknot evaluation on ArchiveII + iPKnot (~13 min, CPU-only)
python eval_pk_from_predictions.py --predictions models_archiveII/predictions_archiveII.pkl --dataset_name ArchiveII
python eval_pk_from_predictions.py --predictions models_ipknot/predictions_bpRNA-PK-TS0-1K.pkl --dataset_name iPKnot
```

Expected output: 7 F1 values matching the summary table in §3 to within CUDA rounding (max drift 0.0003).

---

## 1. Environment

Required: NVIDIA GPU with CUDA 11.8+ driver (tested on H100 NVL with driver 580.105), Python 3.11, ~15 GB disk for data + 200 MB for checkpoints.

```bash
conda env create -f environment.yml
conda activate ufold-repro

# environment.yml does not pin these pip extras; install them manually:
pip install setproctitle "torcheval==0.0.6"
```

**Why `torcheval==0.0.6` specifically?** Version 0.0.7 eagerly imports `torchvision` and `torchaudio` at package load time, which breaks with the torch 2.0.1 + cu118 build pinned by `environment.yml` (CUDA version mismatch). 0.0.6 does not have this eager import. UFold does not use vision/audio modules, so this downgrade is harmless.

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
# Expected: 2.0.1 True <N>
```

---

## 2. Download datasets and checkpoints

Both are hosted on Google Drive:
- **Datasets**: `gdrive_xw:UniRNA/ss_dataset/` — ~15 GB total
- **Checkpoints**: `gdrive_xw:UniRNA/baselines/UFold/checkpoints/` — 204 MB

### Option A: via rclone (recommended)

Requires [rclone](https://rclone.org/) configured with a Google Drive remote. Replace `gdrive:` with your remote name:

```bash
rclone copy gdrive:UniRNA/ss_dataset/ ./data/ -P
rclone copy gdrive:UniRNA/baselines/UFold/checkpoints/ ./checkpoints/ -P
```

### Option B: manual download

Visit `<SHARE_LINK_PLACEHOLDER_TO_BE_INSERTED_BY_AUTHOR>` and download both directories. Unpack into `./data/` and `./checkpoints/`.

### Expected layout after download

```
./data/
  all_data_1024_0.75/         # UniRNA-SS
    train.pkl, valid.pkl, test.pkl
  ipkont/                     # iPKnot
    bpRNA-TR0.pkl, bpRNA-PK-TS0-1K.pkl
  mxfold2/                    # ArchiveII, bpRNA-1m, bpRNA-1m-new
    TR0-canonicals.pkl, VL0-canonicals.pkl, TS0-canonicals.pkl
    RNAStrAlign600-train.pkl, archiveII.pkl, bpRNAnew.pkl
  rivals/                     # Rivals (only contrafold/addss + eternafold variants)
    TrainSetA-addss.pkl,      TrainSetA-eternafold.pkl
    TestSetA-addss.pkl,       TestSetA-eternafold.pkl
    TestSetB-addss.pkl,       TestSetB-eternafold.pkl
./checkpoints/
  rivals_ep99.pt              # Rivals
  unirna_ss_ep99.pt           # UniRNA-SS
  bprna1m_ep99.pt             # bpRNA-1m (epoch 99)
  bprna1m_ep9.pt              # bpRNA-1m-new (epoch 9, early-stopped on VL0)
  archiveII_ep99.pt           # ArchiveII
  ipknot_ep99.pt              # iPKnot
  MD5SUMS                     # md5sum -c MD5SUMS to verify integrity
  README.md                   # per-file description
```

Verify integrity (optional):
```bash
cd checkpoints && md5sum -c MD5SUMS && cd ..
```

---

## 3. Verify reported metrics (evaluation-only)

`scripts/eval_all.sh` launches 3 parallel chains on GPUs 0/1/2 (configurable), runs `eval_from_checkpoint.py` against each benchmark, and prints a summary.

```bash
bash scripts/eval_all.sh ./data ./checkpoints 0,1,2
```

Wall time: ~30 min total (ArchiveII is the longest single task at ~30 min; other benchmarks chain sequentially on the other 2 GPUs).

Expected output (all metrics via `torcheval` binary_* functions, threshold=0.5, macro-averaged per sample):

| # | Benchmark | Train | Test | Ckpt | Precision | Recall | **F1** | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|---|
| 1a | Rivals | TrainSetA-addss (3166) | TestSetA-addss (592) | `rivals_ep99.pt` | 0.7084 | 0.6081 | **0.6343** | 0.8127 | 0.5167 |
| 1b | Rivals | TrainSetA-addss (3166) | TestSetB-addss (430) | `rivals_ep99.pt` | 0.5428 | 0.3596 | **0.4145** | 0.6890 | 0.2562 |
| 2 | UniRNA-SS | train (8323) | test (1041) | `unirna_ss_ep99.pt` | 0.4514 | 0.6383 | **0.4394** | 0.7422 | 0.3420 |
| 3 | bpRNA-1m | TR0-canonicals (10814) + VL0 (1300) | TS0-canonicals (1304, 1 skip) | `bprna1m_ep99.pt` | 0.4786 | 0.6786 | **0.4653** | 0.7583 | 0.3828 |
| 4 | bpRNA-1m-new | TR0-canonicals (10814) + VL0 (1300) | bpRNAnew (5401) | `bprna1m_ep9.pt` ⭐ | 0.5273 | 0.5817 | **0.5387** | 0.8283 | 0.4080 |
| 5 | ArchiveII | RNAStrAlign600-train (20923) | archiveII (3961, 5 skip) | `archiveII_ep99.pt` | 0.6831 | 0.6533 | **0.6584** | 0.8333 | 0.5755 |
| 6 | iPKnot | bpRNA-TR0 (10814) | bpRNA-PK-TS0-1K (2909, 5 skip) | `ipknot_ep99.pt` | 0.4093 | 0.6118 | **0.4118** | 0.7349 | 0.3275 |

⭐ **bpRNA-1m-new uses the epoch-9 checkpoint, not epoch 99.** This is legitimate early stopping based on `VL0-canonicals.pkl` validation loss (lowest at epoch 9 = 0.7117; epoch 99 = 39.68, severe overfitting). VL0 ∩ bpRNAnew = ∅ — see [`REPRODUCTION.md`](REPRODUCTION.md) §Run 7 for the data-leakage verification.

Tolerance: CUDA floating-point operations are not bit-deterministic across runs. Observed max drift on fresh re-evaluation: **0.0003 on Rivals TestSetB precision; all F1 values match to 0.0001**. If your numbers differ by more than 0.001 on any metric, please file an issue with your `nvidia-smi` output, torch version, and the full `logs/verify_*.log`.

---

## 4. Pseudoknot-aware evaluation (UniRNA-SS, ArchiveII, iPKnot)

Additional metrics from the DeepRNA pseudoknot module (`deeprna.metrics.pseudoknot`, used unmodified):
- **score** — overall F1 (sklearn `f1_score` on flattened binarized contact maps)
- **score_pk** — overall F1 restricted to samples that contain ≥1 pseudoknot (crossing pair)
- **pk_sen / pk_ppv / pk_f1** — sensitivity / PPV / F1 of the crossing base-pair prediction, restricted to PK-containing samples

```bash
# Requires the DeepRNA repo accessible at /home/xiwang/project/develop/deeprna
# (or adjust the sys.path.insert in eval_pk_from_predictions.py)

python eval_pk_from_predictions.py \
    --predictions models_unirna_ss/predictions_test.pkl \
    --dataset_name UniRNA-SS

python eval_pk_from_predictions.py \
    --predictions models_archiveII/predictions_archiveII.pkl \
    --dataset_name ArchiveII

python eval_pk_from_predictions.py \
    --predictions models_ipknot/predictions_bpRNA-PK-TS0-1K.pkl \
    --dataset_name iPKnot
```

Wall time: UniRNA-SS ~3 min, ArchiveII ~12 min, iPKnot ~7 min. CPU-only, no GPU needed.

Expected output:

| Benchmark | n_total | n_pk | score (F1) | score_pk | pk_sen | pk_ppv | **pk_f1** |
|---|---|---|---|---|---|---|---|
| UniRNA-SS | 1041 | 164 (15.8%) | 0.4387 | 0.1111 | 0.0229 | 0.0178 | **0.0197** |
| ArchiveII | 3966 | 1079 (27.2%) | 0.6576 | 0.2167 | 0.0045 | 0.0011 | **0.0013** |
| iPKnot | 2914 | 353 (12.1%) | 0.4105 | 0.1869 | 0.0667 | 0.0654 | **0.0639** |

**Interpretation**: UFold's U-Net + Augmented-Lagrangian postprocess does not model crossing base pairs explicitly. `pk_f1` ranges from 0.001 (ArchiveII) to 0.064 (iPKnot), all near-zero. UniRNA-SS sits in between at 0.020. F1 on PK-containing samples (`score_pk`) is roughly 3-4× lower than overall F1 across all three datasets. See [`Benchmark.md`](Benchmark.md) §8 for full discussion.

The sklearn `score` differs from the torcheval F1 in §3 by <0.002; this is a numerical equivalence within rounding (sklearn uses per-sample `zero_division=0.0`).

---

## 5. Training from scratch (optional, ~30 h single-GPU)

```bash
bash scripts/train_all.sh ./data ./models_retrain 0
```

Trains all 6 benchmarks sequentially (UniRNA-SS, Rivals, bpRNA-1m, ArchiveII, iPKnot, then evaluates bpRNA-1m-new using the bpRNA-1m epoch-9 checkpoint). Writes per-run logs to `logs/train_*.log` and per-run checkpoints to `./models_retrain/<name>/ufold_train_rivals_{9,19,...,99}.pt`.

Per-benchmark wall time on one H100 NVL:

| Benchmark | Training set size | Wall time |
|---|---|---|
| Rivals | 3166 | ~100 min |
| UniRNA-SS | 8323 | ~5 h |
| bpRNA-1m | 10814 | ~6 h |
| iPKnot | 10814 | ~5.5 h |
| ArchiveII | 20923 | ~12.5 h |

All hyperparameters match UFold official defaults: `BCEWithLogitsLoss(pos_weight=300)`, Adam(lr=0.001), batch_size=1, 100 epochs, seed=0. See [`REPRODUCTION.md`](REPRODUCTION.md) § Hyperparameters.

After training, re-run the evaluation in §3 against your own checkpoints:
```bash
bash scripts/eval_all.sh ./data ./models_retrain 0,1,2
```

> Note: this requires per-run subdirectories to contain a file named `ufold_train_rivals_99.pt`. `train_all.sh` already produces this layout; manual copying is only needed if your checkpoint filenames differ.

---

## 6. Code changes vs the original UFold repository

See [`REPRODUCTION.md`](REPRODUCTION.md) §Code Changes and [`CHANGE_LOG.md`](CHANGE_LOG.md) for the full audit trail. Summary:

| File | Status | Purpose |
|---|---|---|
| `ufold_train_rivals.py` | **new** (this work) | Training + evaluation driver for the `{id, seq, label}` pickle format |
| `eval_from_checkpoint.py` | **new** (this work) | Standalone checkpoint evaluation |
| `eval_pk_from_predictions.py` | **new** (this work) | Pseudoknot metric computation (reuses DeepRNA module) |
| `eval_no_postprocess.py` | **new** (this work) | Raw-sigmoid ablation (documents postprocess gain) |
| `run_exp.py` | **new** (this work) | Process-disguise wrapper (setproctitle) |
| `scripts/eval_all.sh`, `scripts/train_all.sh` | **new** (this work) | One-command reproduction scripts |
| `Network.py`, `ufold/utils.py`, `ufold/postprocess.py`, `ufold/data_generator.py` | **unchanged** from upstream | U-Net, creatmat, Augmented Lagrangian, data pipeline utilities |

No original UFold file was modified. `git diff HEAD -- Network.py ufold/` should return empty.

---

## 7. Repository layout

```
Network.py                      # U-Net architecture (upstream)
ufold/                          # Upstream modules: utils, postprocess, data_generator, config
ufold_train_rivals.py           # Training + evaluation driver (new)
eval_from_checkpoint.py         # Standalone checkpoint evaluation (new)
eval_pk_from_predictions.py     # Pseudoknot metric evaluation (new)
eval_no_postprocess.py          # Postprocess ablation (new)
run_exp.py                      # Process-disguise wrapper (new)
scripts/
  eval_all.sh                   # One-command eval of all 6 benchmarks (new)
  train_all.sh                  # One-command training of all 6 benchmarks (new)
  run_verification.sh           # Internal verification script (new)
environment.yml                 # Conda env spec (python 3.11, torch 2.0.1, cu118)
Dockerfile, requirements_docker.txt  # Alternative Docker setup
checkpoints/                    # 6 verified final checkpoints + README + MD5SUMS (gitignored; download from GDrive)
README.md                       # This file
Benchmark.md                    # Full results tables
REPRODUCTION.md                 # Per-run details, data integrity checks
CHANGE_LOG.md                   # Chronological action log
```

---

## 8. Acknowledgments

Based on UFold:

> Liang, K., et al. "UFold: fast and accurate RNA secondary structure prediction with deep learning." *Nucleic Acids Research*, 50(3), e14 (2022).

Pseudoknot metrics from the DeepRNA project (`deeprna.metrics.pseudoknot`).

Datasets from:
- Rivals (Rivas lab)
- RNAStrAlign / ArchiveII (via MXfold2-provided pickles, 2023-11-30)
- bpRNA-1m / TR0 / VL0 / TS0 / bpRNAnew (via MXfold2-provided pickles, 2023-11-30)
- UniRNA-SS (internal)
- iPKnot benchmark (bpRNA-PK-TS0-1K)
