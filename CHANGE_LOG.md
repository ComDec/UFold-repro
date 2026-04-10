# Change Log

Records all environment setup, code modifications, and experiment runs for this reproduction.

---

## 2026-04-05: Initial Setup

### Environment

The original UFold README specifies:
```bash
conda env create -f UFold.yaml
conda activate UFold
```
With Python 3.11, PyTorch 2.0.1, CUDA 11.8.

**Actual environment used**: The host machine has Python 3.13.12, PyTorch 2.8.0+cu128, CUDA 12.8 pre-installed. This is a newer stack than the original UFold specifies. We verified all original UFold modules (Network.py, ufold/utils.py, ufold/postprocess.py, ufold/data_generator.py) work correctly under this environment.

**Compatibility note**: The `creatmat` function in `ufold/data_generator.py:880` crashes under NumPy >= 2.0 when given one-hot arrays containing N nucleotides (`[0,0,0,0].index(1)` raises `ValueError`). This is not a problem for the original UFold datasets (which contain only A/U/C/G), but affects the Rivals and MXfold2 datasets which contain N nucleotides. Our adapter uses `ufold/utils.py:creatmat` with string input, which handles N correctly.

**Environment files added** (for reviewers to reproduce):
- `environment.yml`: Conda environment with flexible version pins (Python >= 3.11, PyTorch >= 2.0)
- `Dockerfile` + `requirements_docker.txt`: Docker-based setup
- `test_env.py`: Environment verification script

### Data Adapter Approach

The user's datasets (Rivals, MXfold2) use a different pickle format from UFold's native format:
- **User format**: `[{id: str, seq: str, label: ndarray(N,N), matrix: ndarray(N,N)}, ...]`
- **UFold format**: `[RNA_SS_data(seq=ndarray(600,4), ss_label=ndarray(600,3), length=int, name=str, pairs=list), ...]`

**Decision**: Instead of converting user data to UFold format, we wrote an adapter class (`RivalsDataGenerator` + `RivalsDataset`) that produces the same 17-channel tensors as UFold's `Dataset_Cut_concat_new_merge_multi`. This was chosen because:
1. Converting to UFold format would require fabricating `ss_label` and `pairs` fields from the contact matrix, adding an unnecessary error-prone step.
2. The adapter directly uses `label` as the ground-truth contact map, avoiding any data transformation that could introduce bugs.
3. The 17-channel feature construction logic is identical to the original (verified by code review).

**Alternative approach** (not taken): A conversion script could transform `{seq, label}` dicts into `RNA_SS_data` namedtuples. This would allow using the original `ufold_train.py` unmodified but would require reverse-engineering `pairs` from the contact matrix and synthesizing `ss_label`.

---

## 2026-04-05: Rivals Experiment (Run 1)

### Code added
- `ufold_train_rivals.py` (initial version): `RivalsDataGenerator`, `RivalsDataset`, `train()`, `model_eval_all_test()`

### Experiment details
- Train: `TrainSetA-addss.pkl` (3166 samples)
- Test: `TestSetA-addss.pkl` (592), `TestSetB-addss.pkl` (430)
- Validation: None
- All hyperparameters: UFold defaults (see REPRODUCTION.md)
- GPU: NVIDIA H100 NVL #6
- Training time: 100.2 min
- Loss reporting: last-batch per epoch (original ufold_train.py style)

### Results
- TestSetA: precision=0.7084, f1=0.6343, auroc=0.8127, auprc=0.5167
- TestSetB: precision=0.5428, f1=0.4145, auroc=0.6890, auprc=0.2562

---

## 2026-04-05: Code Review (Round 1)

Subagent (Opus) reviewed `ufold_train_rivals.py` against original UFold. Findings:
- No critical bugs
- Fixed: misleading creatmat comment, line reference errors in docs
- Added: label symmetry assertion, recall metric, explicit error reporting
- Made GPU/paths configurable via argparse

---

## 2026-04-06: MXfold2 Experiment (Run 2)

### Code changes for this run
- Added `--train_file`, `--val_file`, `--test_files` CLI arguments to `ufold_train_rivals.py`
- Added `_compute_val_loss()` for periodic validation monitoring
- Added `binary_recall` to evaluation metrics
- Changed loss reporting from last-batch to epoch-average (`avg_loss`)

### Experiment details
- Train: `TR0-canonicals-addss.pkl` (10814 samples)
- Validation: `VL0-canonicals-addss.pkl` (1300 samples), monitored every 10 epochs
- Test: `TS0-canonicals-addss.pkl` (1305 samples)
- All hyperparameters: UFold defaults (unchanged)
- GPU: NVIDIA H100 NVL #6
- Training time: 342.2 min

### Results
- TS0: precision=0.4786, recall=0.6786, f1=0.4653, auroc=0.7583, auprc=0.3828
- 1 sample skipped (no positive labels)
- Significant overfitting observed in validation loss (expected with pos_weight=300, no early stopping)

---

## 2026-04-06: Code Review (Round 2)

Subagent (Opus) reviewed full codebase, REPRODUCTION.md, and Benchmark.md. Findings:
- No critical bugs, no data leakage, no test-set contamination
- Fixed: metric computation difference documentation, line references, sample count clarity
- Added: postprocessing slicing equivalence note, channel index correction

---

## 2026-04-06: Documentation Update

Added missing documentation per reproducibility guidelines:
- `CHANGE_LOG.md` (this file)
- Updated `Benchmark.md` with "How to Run New Benchmarks" section and prohibited actions
- Updated `REPRODUCTION.md` with evaluation approach notes

### Evaluation approach note

The user specified metrics consistent with `DeepRNA/deepprotein/tasks/utils.py:secondary_structure_metircs`. We reimplemented the evaluation using the same `torcheval` functions (`binary_precision`, `binary_recall`, `binary_f1_score`, `binary_auroc`, `binary_auprc`) rather than importing directly from DeepRNA, because:
1. DeepRNA requires additional dependencies (`chanfig`, `danling`, `NestedTensor`) not used by UFold.
2. The DeepRNA function has `binary_recall` commented out due to a torcheval bug.
3. Direct reimplementation with the same torcheval functions is simpler and independently verifiable.

The evaluation is per-sample flatten-then-compute, macro-averaged, with threshold=0.5 for precision/recall/F1 -- identical to DeepRNA. The only documented difference is skipping samples with no positive labels (AUROC/AUPRC undefined).

---

## 2026-04-08: UniRNA-SS & iPKnot Experiments (Runs 3-4)

### Experiment details

Both experiments ran simultaneously on GPU 4 (NVIDIA H100 NVL).

**Run 3 -- UniRNA-SS:**
- Data: `/home/xiwang/project/develop/data/all_data_1024_0.75/` (`train.pkl`, `valid.pkl`, `test.pkl`)
- Data format: `{id, seq, label}` dicts (same as Rivals but without `matrix` field)
- Train: 8,323 samples (seq len 23-1018), Valid: 1,041, Test: 1,041
- Training time: 301.8 min
- Results: precision=0.4514, recall=0.6383, f1=0.4394, auroc=0.7422, auprc=0.3420

**Run 4 -- iPKnot:**
- Data: `/home/xiwang/project/develop/data/ipkont/` (`bpRNA-TR0.pkl`, `bpRNA-PK-TS0-1K.pkl`)
- Data format: same `{id, seq, label}` dicts
- Train: 10,814 samples (seq len 33-498), Test: 2,914 (seq len 12-1000)
- Training time: 325.9 min
- Results: precision=0.4093, recall=0.6118, f1=0.4118, auroc=0.7349, auprc=0.3275 (5 samples skipped)

### Code changes for these runs

None. `ufold_train_rivals.py` was used unmodified. The `RivalsDataGenerator` only accesses `id`, `seq`, `label` keys -- the absence of `matrix` in these datasets causes no issues.

### Legitimacy review

Subagent (Opus) verified:
- No data overlap between train/valid/test splits (both experiments)
- Prediction file IDs and labels match test sets exactly
- Metrics independently recomputed from saved predictions match reported values
- No code modifications between runs
- Hyperparameters match UFold defaults

### Intermediate files generated

| Directory | Contents |
|---|---|
| `models_unirna_ss/` | 10 checkpoints, `predictions_test.pkl`, `training.log` |
| `models_ipknot/` | 10 checkpoints, `predictions_bpRNA-PK-TS0-1K.pkl`, `training.log` |

---

## 2026-04-08 to 2026-04-09: ArchiveII & bpRNA-1m/bpRNA-1m-new (Runs 5-7)

### Process disguise wrapper

Added `run_exp.py` (per `~/.claude/skills/process-disguise/SKILL.md`) to hide experiment identity from `ps aux`. All training/eval jobs launched through this wrapper show as `python train_gen.py exp=expN` in the process list. Real experiment names are preserved only in log filenames under `logs/`.

```python
# run_exp.py
import sys, os, setproctitle
expname = sys.argv[1]
target_script = sys.argv[2]
script_args = sys.argv[3:]
setproctitle.setproctitle(f"python train_gen.py exp={expname}")
sys.argv = [target_script] + script_args
exec(compile(open(target_script).read(), target_script, 'exec'))
```

This wrapper does not affect training behavior at all -- it only replaces the process title string via `setproctitle`.

### Run 5: ArchiveII (2026-04-09, GPU 1)

**First attempt (2026-04-08, GPU 3)**: training stopped after epoch 6 (no checkpoint saved, no error in log). Cause unknown (possibly OOM on ArchiveII's max 2968nt sequences or GPU 3 conflict). Removed old log and restarted.

**Second attempt (2026-04-09, GPU 1)**: completed 100 epochs in 753.2 min.

- Data: `RNAStrAlign600-train.pkl` (20,923) → `archiveII.pkl` (3,966)
- Data directory: `/home/xiwang/project/develop/data/mxfold2/`
- No validation set
- Hyperparameters: UFold defaults (unchanged)
- Results: P=0.6831, R=0.6533, **F1=0.6584**, AUROC=0.8333, AUPRC=0.5755 (5 skipped)
- Best F1 of all UFold benchmark runs so far; close to paper's reported ~0.70

### Run 6: bpRNA-1m re-train (2026-04-09, GPU 5)

Rationale: the user's dataset instruction file (`/home/xiwang/project/develop/deeprna/dataset_instruction.md`) specifies `TR0-canonicals.pkl` (without `-addss` suffix) for the bpRNA-1m benchmark. Run 2 (MXfold2) used the `-addss` variant. Re-ran to match the instruction file exactly.

- Data: `TR0-canonicals.pkl` (10,814), `VL0-canonicals.pkl` (1,300), `TS0-canonicals.pkl` (1,305)
- Training time: 352.3 min
- Results: **identical** to Run 2 (P=0.4786, R=0.6786, F1=0.4653, AUROC=0.7583, AUPRC=0.3828)
- Confirms that `-addss` and non-`-addss` pkl variants share the same ground-truth `label` field (they differ only in the unused `matrix` field)
- Validation loss: epoch 9 is **lowest** (0.7117), epoch 99 severely overfit (39.68)

### Run 7: bpRNA-1m-new inference (2026-04-09, GPU 1)

- Train/Val: same as Run 6 (no re-training)
- Test: `bpRNAnew.pkl` (5,401 samples)
- Checkpoint: `models_bprna1m/ufold_train_rivals_9.pt` (epoch 9, lowest VL0 val loss)
- Script: `eval_from_checkpoint.py` (pre-existing standalone inference script)
- Results: P=0.5273, R=0.5817, **F1=0.5387**, AUROC=0.8283, AUPRC=0.4080 (all 5401 evaluated)

**Checkpoint selection rationale**: The user asked for the "best" bpRNA-1m model. We interpret this as "the checkpoint with lowest held-out validation loss" since:
1. VL0 (validation set) is disjoint from bpRNAnew (test set) -- verified 0 ID/seq overlap
2. Early stopping on a held-out val set is standard practice, NOT cherry-picking on the test set
3. The training shows severe overfitting (val loss at epoch 99 is 56x higher than epoch 9)

This selection deviates from the default "use epoch 99 checkpoint" policy documented in Benchmark.md. The deviation is explicit, documented, and uses validation data only (not test data).

**Data leakage verification** (before running inference):
```python
TR0 ∩ bpRNAnew (ID):  0
VL0 ∩ bpRNAnew (ID):  0
TR0 ∩ bpRNAnew (seq): 0
VL0 ∩ bpRNAnew (seq): 0
Unique seqs in bpRNAnew: 5401/5401
```

### Intermediate files generated

| Directory | Contents |
|---|---|
| `models_archiveII/` | 10 checkpoints, `predictions_archiveII.pkl` (2.1 GB) |
| `models_bprna1m/` | 10 checkpoints, `predictions_TS0-canonicals.pkl`, `predictions_bpRNAnew_ep9.pkl` (~650 MB) |
| `logs/` | `ufold_archiveII_retrain.log`, `ufold_bprna1m_retrain.log`, `ufold_bprna1m_new_eval.log` |

All listed in `.gitignore`.

### Files added / modified

| File | Status | Purpose |
|---|---|---|
| `run_exp.py` | new | Process-disguise wrapper |
| `eval_from_checkpoint.py` | new (created 2026-04-08, never committed as of 2026-04-09) | Standalone inference script |
| `ufold_train_rivals.py` | unchanged since 2026-04-06 commit | |
| `.gitignore` | updated 2026-04-09 to add `logs/`, `models_archiveII/`, `models_bprna1m/` | |
| All original UFold files | unchanged (`Network.py`, `ufold/{postprocess,utils,data_generator}.py`) | |

### 2026-04-09: Code review round 3 (subagent, Opus)

Reviewed Runs 5, 6, 7 against the original UFold codebase, checked for data leakage, and verified all reported metrics. Findings:

**No critical methodological issues.**
- `git diff HEAD` on all original UFold files and on `ufold_train_rivals.py` returned 0 lines (nothing tampered during the runs)
- `run_exp.py` is a pure `setproctitle` wrapper (13 lines, `exec()` only) and cannot affect training behavior
- `eval_from_checkpoint.py` is byte-equivalent to `ufold_train_rivals.py:model_eval_all_test()` for the data pipeline and `evaluate()` logic
- All three runs' metrics were independently recomputed from saved predictions and matched the logs exactly
- Run 6 = Run 2 label identity verified at the raw ndarray level (confirms `-addss` vs non-`-addss` only differ in unused `matrix` field)
- Run 7 (bpRNA-1m-new) data leakage check: TR0 ∩ bpRNAnew = 0, VL0 ∩ bpRNAnew = 0 (both ID and sequence)
- Run 7 checkpoint selection by VL0 validation loss (epoch 9) is legitimate early stopping, not test-set cherry-picking

**Run 5 (ArchiveII) sequence overlap — now disclosed in docs.**

The review found that `RNAStrAlign600-train.pkl` and `archiveII.pkl` share **1,869 / 3,966 test sequences by exact string match (47.1%)**, of which **1,607 (40.5%)** have identical (seq, label) pairs. The initial Benchmark.md §5 Legitimacy review had stated "No train/test ID overlap" which was technically true (no ID matches) but under-reported the situation. This has been fixed in:
- `Benchmark.md` §5 Legitimacy review (now includes the full breakdown table and per-subset F1)
- `REPRODUCTION.md` Run 5 Data integrity (now lists the overlap numbers)

The overlap is **not introduced by this reproduction** — it is inherent to the community-standard RNAStrAlign → ArchiveII protocol used by the original UFold paper, MXfold2, SPOT-RNA, and other published methods using the same MXfold2-provided pickles (2023-11-30). The overall F1 (0.6584) remains directly comparable to the UFold paper's reported ~0.70 because both use the same overlapping splits.

Per-subset F1 (programmatically computed from saved predictions):
- Clean samples (n=2096, seq not in train): F1 = 0.6569
- Seq-only leak (n=262, same seq, different label): F1 = 0.5808
- Full leak (n=1603, same seq + same label): F1 = 0.6730
- Overall (n=3961): F1 = 0.6584

The +0.016 gap between clean and full-leak subsets indicates the model does not strongly memorize training sequences.

**Other documentation fixes made after review:**
- Corrected archiveII max sequence length: 1800 (not 2968 as originally written); removed the incorrect ">2000 nt OOM" rationale
- Added `logs/`, `models_archiveII/`, `models_bprna1m/` to `.gitignore`
- Clarified `eval_from_checkpoint.py` provenance (created 2026-04-08, never committed)

### 2026-04-09: Postprocess ablation on bpRNA-1m-new

Added one-off analysis script `eval_no_postprocess.py` to measure the Augmented Lagrangian postprocess gain. Runs the same inference pipeline as `eval_from_checkpoint.py` but replaces the `postprocess()` call with raw `sigmoid()` on the U-Net logits.

- Script: `eval_no_postprocess.py` (new, untracked, standalone one-off)
- Log: `logs/ufold_bprna1m_new_nopost.log`
- Checkpoint: `models_bprna1m/ufold_train_rivals_9.pt` (same as Run 7)
- Test: `bpRNAnew.pkl` (same as Run 7)

**Results:**

| Metric | Raw sigmoid | With postprocess | Δ |
|---|---|---|---|
| Precision | 0.1047 | 0.5273 | **+0.4226** |
| Recall | 0.8738 | 0.5817 | −0.2921 |
| F1 | 0.1842 | 0.5387 | **+0.3545** |
| AUROC | 0.9811 | 0.8283 | −0.1528 |
| AUPRC | 0.5552 | 0.4080 | −0.1472 |

**Key finding**: the F1 increase from postprocess is +0.3545 (nearly 3× raw baseline). Postprocess is the dominant contributor to UFold's final F1 score. AUROC/AUPRC are higher on raw sigmoid because postprocess binarizes the output and degrades threshold-free metrics — this is a metric artifact, not a model-performance drop. Full discussion in `Benchmark.md` §7 "Ablation: Augmented Lagrangian postprocess gain".

No changes to training code or existing evaluation code. `eval_no_postprocess.py` is a standalone file and does not import or modify `ufold_train_rivals.py` or `eval_from_checkpoint.py`.

---

## 2026-04-10: Pseudoknot-aware metrics on ArchiveII & iPKnot

User asked to additionally evaluate Runs 4 (iPKnot) and 5 (ArchiveII) with the DeepRNA pseudoknot metric module at `/home/xiwang/project/develop/deeprna/deeprna/metrics/pseudoknot.py`.

### Approach

Computed the PK metrics from the **existing saved predictions** (`models_archiveII/predictions_archiveII.pkl`, `models_ipknot/predictions_bpRNA-PK-TS0-1K.pkl`). No GPU re-inference — the Augmented-Lagrangian postprocessed `pred` arrays are already stored per-sample. This avoids any non-determinism risk in the postprocess and guarantees we're scoring the same checkpoint that produced the reported F1s.

`evaluate_structure_metrics` from `pseudoknot.py` was used **unmodified** (per CLAUDE.md's "尽可能不做修改的使用用户的评测函数" principle). The saved prediction key `pred` was passed as `pred_prob`. The function uses `pred > threshold` (default 0.5) internally, and UFold's postprocessed outputs (range [0, ~1.6]) binarize correctly at 0.5.

### Code added

- `eval_pk_from_predictions.py` — new standalone script. Loads a saved predictions pkl, slices each pred/label to `seq_len`, and calls `evaluate_structure_metrics` from the user's deeprna module. Also recomputes the standard torcheval metrics (precision/recall/F1/AUROC/AUPRC) on the same predictions as a sanity check against REPRODUCTION.md. Does not import, modify, or reach into `ufold_train_rivals.py` or `eval_from_checkpoint.py`.

No training code, no eval code, no data files modified.

### Runs

Both launched via `run_exp.py` (process disguise, exp33 / exp34) with logs in `logs/`. CPU-only, no GPU.

| Exp | Script | Predictions source | Log |
|---|---|---|---|
| exp33 | eval_pk_from_predictions.py | models_archiveII/predictions_archiveII.pkl | logs/ufold_archiveII_pkeval.log |
| exp34 | eval_pk_from_predictions.py | models_ipknot/predictions_bpRNA-PK-TS0-1K.pkl | logs/ufold_ipknot_pkeval.log |

### Sanity check (standard torcheval metrics recomputed from the saved pkls)

| Dataset | P | R | F1 | AUROC | AUPRC | Matches REPRODUCTION.md? |
|---|---|---|---|---|---|---|
| ArchiveII | 0.6831 | 0.6533 | 0.6584 | 0.8333 | 0.5755 | ✓ Run 5 exactly |
| iPKnot | 0.4093 | 0.6118 | 0.4118 | 0.7349 | 0.3275 | ✓ Run 4 exactly |

Bit-for-bit match. The saved predictions are the same artifacts that produced the originally documented numbers.

### Pseudoknot metrics

| Dataset | n_total | n_pk | score (sklearn F1) | score_pk | pk_sen | pk_ppv | **pk_f1** |
|---|---|---|---|---|---|---|---|
| ArchiveII | 3966 | 1079 (27.2%) | 0.6576 | 0.2167 | 0.0045 | 0.0011 | **0.0013** |
| iPKnot (bpRNA-PK-TS0-1K) | 2914 | 353 (12.1%) | 0.4105 | 0.1869 | 0.0667 | 0.0654 | **0.0639** |

The sklearn `score` differs from torcheval F1 by <0.002 (sklearn uses `zero_division=0.0`, per-sample; numerically equivalent to within rounding).

### Interpretation

- **UFold cannot predict pseudoknots.** pk_f1 is 0.0013 on ArchiveII and 0.0639 on iPKnot — both essentially no PK-prediction capability. This is expected: UFold's architecture (2D U-Net + Augmented-Lagrangian postprocess enforcing ≤1 partner per position) does not explicitly model crossing base pairs, and neither `RNAStrAlign600-train.pkl` nor `bpRNA-TR0.pkl` has enough PK supervision.
- **score_pk (F1 on PK-containing samples) is ~3× lower than overall F1** for both datasets: 0.22 vs 0.66 (ArchiveII) and 0.19 vs 0.41 (iPKnot). PK-containing samples are systematically harder because the standard base pairs around a PK site are often mispredicted too.
- **iPKnot PK metrics are slightly better than ArchiveII.** (pk_f1 0.064 vs 0.001). The iPKnot test set is bpRNA-PK-TS0 — a PK-curated subset of bpRNA. Its train split (`bpRNA-TR0.pkl`) contains some PK examples, giving UFold minimal exposure. ArchiveII's training set `RNAStrAlign600-train.pkl` comes from RNAStrAlign, which has even fewer PKs. Neither is competitive with PK-specialized methods (iPKnot, ProbKnot).

### Timing

- iPKnot eval: ~7.5 min total (60s torcheval + 383s pseudoknot)
- ArchiveII eval: ~13 min total (68s torcheval + 722s pseudoknot)
- Both ran concurrently; neither touched a GPU.

### Files

| File | Status |
|---|---|
| `eval_pk_from_predictions.py` | new, untracked |
| `logs/ufold_archiveII_pkeval.log` | new |
| `logs/ufold_ipknot_pkeval.log` | new |
| `CHANGE_LOG.md`, `REPRODUCTION.md`, `Benchmark.md` | updated with this evaluation |
| All training/eval code | unchanged |
