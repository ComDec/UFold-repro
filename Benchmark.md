# Benchmark Results — Detailed Per-Run Reference

> **Summary tables live in [`README.md`](README.md) §3 (standard metrics) and §4 (pseudoknot metrics).** This document contains the full per-benchmark details: dataset configuration, reproduction commands, hyperparameters, training loss curves, and legitimacy reviews.

All experiments use the UFold U-Net architecture (img_ch=17, 8.6M params) with official default hyperparameters (identical across all 6 benchmarks). Metrics are consistent with DeepRNA's `secondary_structure_metircs` (`torcheval` binary_*, per-sample flatten, macro-average, threshold=0.5 for P/R/F1).

Notes on metric conventions (apply to every section below):
- Rivals Run 1 used an older code revision that did not report recall (added in later runs).
- Samples with no positive labels are skipped (AUROC/AUPRC undefined). This differs from DeepRNA's `secondary_structure_metircs` which does not skip; if reviewers want the exact DeepRNA numbers they should unskip and tolerate NaN.
- DeepRNA's upstream `secondary_structure_metircs` has `binary_recall` commented out due to a torcheval bug workaround; we include recall.
- **Run 2** (MXfold2 `-addss`) and **Run 6** (bpRNA-1m canonicals) are numerically identical — the `-addss` pickles and non-`-addss` pickles differ only in an unused `matrix` field, same labels. Run 6 is the canonical version tracked in [`README.md`](README.md); Run 2 is retained below only for historical completeness.

---

## 1. Rivals Benchmark

### Configuration

| Parameter | Value |
|---|---|
| Training set | `TrainSetA-addss.pkl` (3,166 samples, seq len 10-734) |
| Test sets | `TestSetA-addss.pkl` (592), `TestSetB-addss.pkl` (430) |
| Validation | None |
| Input channels | 17 (16 pairwise + 1 creatmat) |
| Loss | BCEWithLogitsLoss(pos_weight=300) |
| Optimizer | Adam (lr=0.001) |
| Batch size | 1 |
| Epochs | 100 |
| Postprocessing | Augmented Lagrangian (lr_min=0.01, lr_max=0.1, num_itr=100, rho=1.6, s=1.5) |
| Seed | 0 |
| GPU | NVIDIA H100 NVL |
| Training time | 100.2 min |

### Code changes vs original UFold

Only `ufold_train_rivals.py` added. No original files modified. Changes:
- `RivalsDataGenerator`: reads rivals pickle format (`{id, seq, label, matrix}`) instead of UFold's `RNA_SS_data` namedtuple format.
- `RivalsDataset`: equivalent to `Dataset_Cut_concat_new_merge_multi`, produces 17-channel tensors. Uses `utils.py:creatmat` with string input (the `data_generator.py:creatmat` at line 880 crashes on N nucleotides).
- `model_eval_all_test()`: evaluation function using `torcheval` binary metrics.
- `train()`: identical logic to `ufold_train.py:train()`, added avg_loss reporting and optional validation.

### Results

| Metric | TestSetA (n=592) | TestSetB (n=430) |
|---|---|---|
| Precision | 0.7084 | 0.5428 |
| F1 | 0.6343 | 0.4145 |
| AUROC | 0.8127 | 0.6890 |
| AUPRC | 0.5167 | 0.2562 |

### Training loss (last-batch per epoch, old code version without avg reporting)

```
Epoch  0: 0.6612    Epoch  50: 0.6758
Epoch  9: 0.1248    Epoch  59: 0.1204
Epoch 19: 0.0831    Epoch  69: 0.6860
Epoch 29: 0.0933    Epoch  79: 0.5934
Epoch 39: 0.0886    Epoch  89: 0.0759
Epoch 49: 0.0395    Epoch  99: 0.0201
```

Note: These are last-batch losses (not epoch averages), so they appear noisy. This was the reporting style of the initial code version; the MXfold2 run uses the updated code with avg_loss.

### Reproduction command

```bash
PYTHONUNBUFFERED=1 python ufold_train_rivals.py \
    --gpu 0 \
    --data_dir /path/to/rivals \
    --save_dir ./models_rivals \
    --train_file TrainSetA-addss.pkl \
    --test_files TestSetA-addss.pkl TestSetB-addss.pkl
```

---

## 2. MXfold2 (TR0 / VL0 / TS0)

### Configuration

| Parameter | Value |
|---|---|
| Training set | `TR0-canonicals-addss.pkl` (10,814 samples, seq len 33-498) |
| Validation set | `VL0-canonicals-addss.pkl` (1,300 samples, seq len 33-497) |
| Test set | `TS0-canonicals-addss.pkl` (1,305 samples, seq len 22-499) |
| Input channels | 17 (16 pairwise + 1 creatmat) |
| Loss | BCEWithLogitsLoss(pos_weight=300) |
| Optimizer | Adam (lr=0.001) |
| Batch size | 1 |
| Epochs | 100 |
| Postprocessing | Augmented Lagrangian (lr_min=0.01, lr_max=0.1, num_itr=100, rho=1.6, s=1.5) |
| Seed | 0 |
| GPU | NVIDIA H100 NVL |
| Training time | 342.2 min |

### Code changes vs Rivals run

None. Same `ufold_train_rivals.py`, different CLI arguments. The `--val_file` flag enables periodic validation loss reporting.

### Results

| Metric | TS0-canonicals (n=1304) |
|---|---|
| Precision | 0.4786 |
| Recall | 0.6786 |
| F1 | 0.4653 |
| AUROC | 0.7583 |
| AUPRC | 0.3828 |

1 sample skipped (no positive labels in ground truth).

### Training loss (avg per epoch)

```
Epoch   0: 0.3898    Epoch  50: 0.1403
Epoch  10: 0.2119    Epoch  60: 0.1364
Epoch  20: 0.1763    Epoch  70: 0.1326
Epoch  30: 0.1573    Epoch  80: 0.1303
Epoch  40: 0.1485    Epoch  90: 0.1284
Epoch  99: 0.1261
```

### Validation loss (VL0, every 10 epochs)

```
Epoch   9: 0.7117
Epoch  19: 0.8685
Epoch  29: 0.9084
Epoch  39: 1.5168
Epoch  49: 3.7640
Epoch  59: 1.2818
Epoch  69: 3.6785
Epoch  79: 4.5007
Epoch  89: 143.27
Epoch  99: 39.681
```

Significant overfitting observed. This is expected behavior with UFold's aggressive `pos_weight=300` and no early stopping / learning rate scheduling. The original UFold paper does not use validation-based early stopping.

### Reproduction command

```bash
PYTHONUNBUFFERED=1 python ufold_train_rivals.py \
    --gpu 0 \
    --data_dir /path/to/mxfold2 \
    --save_dir ./models_mxfold2 \
    --train_file TR0-canonicals-addss.pkl \
    --val_file VL0-canonicals-addss.pkl \
    --test_files TS0-canonicals-addss.pkl
```

---

## Paper Comparison

The original UFold paper (Liang et al., NAR 2022) reports results on different data splits (RNAStralign, ArchiveII, bpRNA) with their own preprocessing. Direct numerical comparison is not applicable because:

1. **Different datasets**: The Rivals and MXfold2 benchmarks use different train/test splits than the UFold paper.
2. **Different metrics**: The paper reports per-sample precision/recall/F1 computed via `evaluate_exact_new()` (which counts TP/FP/FN on the contact map). We use `torcheval` binary metrics on flattened matrices, consistent with the DeepRNA evaluation framework.
3. **Different preprocessing**: The paper uses `process_data_newdataset.py` to convert BPSEQ files to RNA_SS_data namedtuples. The Rivals/MXfold2 data is already in `{seq, label, matrix}` dict format.

For reference, the UFold paper reports on their TS datasets:
- TS1: F1 ~0.72
- TS2: F1 ~0.69
- TS3: F1 ~0.61
- ArchiveII: F1 ~0.70

(These are approximate values from the paper using their own metric formulation.)

---

## How to Run a New Benchmark

### Step-by-step

1. **Prepare data** in the expected pickle format: a list of dicts `{id: str, seq: str, label: ndarray(N,N)}`. The `matrix` field is optional and ignored. The `label` must be a symmetric binary contact map.

2. **Specify train/val/test splits explicitly**:
   ```bash
   PYTHONUNBUFFERED=1 python ufold_train_rivals.py \
       --gpu 0 \
       --data_dir /path/to/data \
       --save_dir ./models_yourexp \
       --train_file your_train.pkl \
       --val_file your_val.pkl \       # optional
       --test_files your_test.pkl
   ```

3. **Record everything** in Benchmark.md and CHANGE_LOG.md:
   - Dataset name, source, and split sizes
   - Exact command used
   - Any code changes required
   - Full training log (loss curve, validation loss)
   - Final test metrics

4. **Run code review** via subagent after each experiment to verify:
   - No data leakage between splits
   - Correct train/test file assignment
   - No silent test-set substitution

### Checklist for new benchmarks

- [ ] Train/val/test splits are user-specified and documented
- [ ] No data from test set is used during training
- [ ] Hyperparameters match UFold defaults (or deviations are documented)
- [ ] Results are reproducible with the exact command recorded
- [ ] Code review subagent confirms no data integrity issues

### Prohibited actions

The following actions are strictly prohibited during benchmarking:

1. **Modifying datasets**: Never alter, filter, resample, or reorder the user-provided train/val/test pickle files. Use them exactly as given.
2. **Training on test data**: The test set must only be used in `model_eval_all_test()` after training completes. Never include test files in `--train_file`.
3. **Silent test-set substitution**: The `--test_files` argument must match the user-specified test set. Do not substitute a different file.
4. **Cherry-picking checkpoints**: Report results from the final epoch (epoch 99) checkpoint unless the user explicitly requests otherwise. Do not search across checkpoints for the best test result.
5. **Hyperparameter tuning on test set**: Do not tune hyperparameters (learning rate, pos_weight, epochs, postprocessing params) to improve test metrics. Use UFold defaults unless the user specifies otherwise.
6. **Unreported code changes**: All code modifications must be recorded in CHANGE_LOG.md and visible in `git diff`.
7. **Uncommitted evaluation logic changes**: Do not change the evaluation function (thresholds, metric functions, flattening strategy) between experiments without documenting the change.

---

## 3. UniRNA-SS

### Configuration

| Parameter | Value |
|---|---|
| Training set | `train.pkl` (8,323 samples, seq len 23-1018) |
| Validation set | `valid.pkl` (1,041 samples, seq len 46-953) |
| Test set | `test.pkl` (1,041 samples, seq len 55-1014) |
| Input channels | 17 (16 pairwise + 1 creatmat) |
| Loss | BCEWithLogitsLoss(pos_weight=300) |
| Optimizer | Adam (lr=0.001) |
| Batch size | 1 |
| Epochs | 100 |
| Postprocessing | Augmented Lagrangian (lr_min=0.01, lr_max=0.1, num_itr=100, rho=1.6, s=1.5) |
| Seed | 0 |
| GPU | NVIDIA H100 NVL #4 |
| Training time | 301.8 min |

### Code changes vs MXfold2 run

None. Same `ufold_train_rivals.py`, different CLI arguments. UniRNA-SS data uses the same `{id, seq, label}` dict format (no `matrix` field; not required by the code).

### Results

| Metric | test (n=1041) |
|---|---|
| Precision | 0.4514 |
| Recall | 0.6383 |
| F1 | 0.4394 |
| AUROC | 0.7422 |
| AUPRC | 0.3420 |

### Training loss (avg per epoch)

```
Epoch   0: 0.5070    Epoch  50: 0.1336
Epoch  10: 0.2615    Epoch  60: 0.1269
Epoch  20: 0.1930    Epoch  70: 0.1220
Epoch  30: 0.1597    Epoch  80: 0.1180
Epoch  40: 0.1478    Epoch  90: 0.1151
Epoch  99: 0.1138
```

### Validation loss (valid.pkl, every 10 epochs)

```
Epoch   9: 0.5818
Epoch  19: 1.5438
Epoch  29: 1.0435
Epoch  39: 1.7511
Epoch  49: 1.7650
Epoch  59: 2.9538
Epoch  69: 3.5614
Epoch  79: 5.9672
Epoch  89: 1.8360
Epoch  99: 53.257
```

Significant overfitting observed. This is expected behavior with UFold's aggressive `pos_weight=300` and no early stopping / learning rate scheduling.

### Data notes

- UniRNA-SS data is from `all_data_1024_0.75` directory (sequences up to 1024 nt, 0.75 similarity threshold)
- 422/8323 training samples contain N nucleotides, handled correctly via `utils.py:creatmat`
- No `matrix` field in data (not required)

### Reproduction command

```bash
PYTHONUNBUFFERED=1 python ufold_train_rivals.py \
    --gpu 4 \
    --data_dir /home/xiwang/project/develop/data/all_data_1024_0.75 \
    --save_dir ./models_unirna_ss \
    --train_file train.pkl \
    --val_file valid.pkl \
    --test_files test.pkl
```

### Legitimacy review

- No train/valid/test ID or sequence overlap (verified programmatically)
- Prediction file IDs match test set exactly (1041/1041)
- Saved labels match original test labels (1041/1041)
- Metrics independently recomputed from saved predictions match reported values
- No code changes between runs

---

## 4. iPKnot

### Configuration

| Parameter | Value |
|---|---|
| Training set | `bpRNA-TR0.pkl` (10,814 samples, seq len 33-498) |
| Validation set | None |
| Test set | `bpRNA-PK-TS0-1K.pkl` (2,914 samples, seq len 12-1000) |
| Input channels | 17 (16 pairwise + 1 creatmat) |
| Loss | BCEWithLogitsLoss(pos_weight=300) |
| Optimizer | Adam (lr=0.001) |
| Batch size | 1 |
| Epochs | 100 |
| Postprocessing | Augmented Lagrangian (lr_min=0.01, lr_max=0.1, num_itr=100, rho=1.6, s=1.5) |
| Seed | 0 |
| GPU | NVIDIA H100 NVL #4 |
| Training time | 325.9 min |

### Code changes vs previous runs

None. Same `ufold_train_rivals.py`, different CLI arguments.

### Results

| Metric | bpRNA-PK-TS0-1K (n=2909) |
|---|---|
| Precision | 0.4093 |
| Recall | 0.6118 |
| F1 | 0.4118 |
| AUROC | 0.7349 |
| AUPRC | 0.3275 |

5 samples skipped (no positive labels in ground truth).

### Training loss (avg per epoch)

```
Epoch   0: 0.5217    Epoch  50: 0.1506
Epoch  10: 0.2688    Epoch  60: 0.1443
Epoch  20: 0.2037    Epoch  70: 0.1392
Epoch  30: 0.1752    Epoch  80: 0.1367
Epoch  40: 0.1608    Epoch  90: 0.1335
Epoch  99: 0.1313
```

### Data notes

- iPKnot training set (`bpRNA-TR0.pkl`) contains the same 10,814 samples as the MXfold2 TR0 set
- Test set (`bpRNA-PK-TS0-1K.pkl`) contains 2,914 samples with pseudoknots, up to 1000 nt
- 132/10814 training samples contain N nucleotides
- No validation set provided for this benchmark

### Reproduction command

```bash
PYTHONUNBUFFERED=1 python ufold_train_rivals.py \
    --gpu 4 \
    --data_dir /home/xiwang/project/develop/data/ipkont \
    --save_dir ./models_ipknot \
    --train_file bpRNA-TR0.pkl \
    --test_files bpRNA-PK-TS0-1K.pkl
```

### Legitimacy review

- No train/test ID or sequence overlap (verified programmatically)
- Prediction file IDs match test set exactly (2914/2914)
- Saved labels match original test labels (2914/2914)
- Metrics independently recomputed from saved predictions match reported values
- No code changes between runs

---

## 5. ArchiveII

### Configuration

| Parameter | Value |
|---|---|
| Training set | `RNAStrAlign600-train.pkl` (20,923 samples, seq len 13-599) |
| Validation set | None |
| Test set | `archiveII.pkl` (3,966 samples, seq len 28-1800, mean 208) |
| Input channels | 17 (16 pairwise + 1 creatmat) |
| Loss | BCEWithLogitsLoss(pos_weight=300) |
| Optimizer | Adam (lr=0.001) |
| Batch size | 1 |
| Epochs | 100 |
| Postprocessing | Augmented Lagrangian (lr_min=0.01, lr_max=0.1, num_itr=100, rho=1.6, s=1.5) |
| Seed | 0 |
| GPU | NVIDIA H100 NVL #1 |
| Training time | 753.2 min |

### Code changes vs previous runs

None. Same `ufold_train_rivals.py`, different CLI arguments.

### Results

| Metric | archiveII (n=3961) |
|---|---|
| Precision | 0.6831 |
| Recall | 0.6533 |
| F1 | 0.6584 |
| AUROC | 0.8333 |
| AUPRC | 0.5755 |

5 samples skipped (no positive labels in ground truth).

### Training loss (avg per epoch)

```
Epoch   0: 0.1820    Epoch  50: 0.0879
Epoch  10: 0.0972    Epoch  60: 0.0853
Epoch  20: 0.0932    Epoch  70: 0.0833
Epoch  30: 0.0912    Epoch  80: 0.0818
Epoch  40: 0.0895    Epoch  90: 0.0805
Epoch  99: 0.0797
```

### Paper comparison

The original UFold paper (Liang et al., NAR 2022) reports F1 ~0.70 on ArchiveII (using their per-sample TP/FP/FN metric). Our F1=0.6584 is close (our metric is torcheval flatten-then-binary, different formulation).

### Data notes

- Training set `RNAStrAlign600-train.pkl` is RNAStrAlign filtered to seq len ≤ 600 (per the filename convention)
- Test set `archiveII.pkl` contains 3,966 samples with seq len 28-1800 (mean 208, 16 samples > 1000 nt)
- Max test sequence length 1800, max training sequence length 599

### Reproduction command

```bash
PYTHONUNBUFFERED=1 python3 run_exp.py exp1 ufold_train_rivals.py \
    --gpu 1 \
    --data_dir /home/xiwang/project/develop/data/mxfold2 \
    --save_dir ./models_archiveII \
    --train_file RNAStrAlign600-train.pkl \
    --test_files archiveII.pkl
```

### Legitimacy review

- No train/test **ID** overlap (different BPSEQ filenames; 0 shared IDs)
- Prediction file IDs match test set exactly (3966/3966)
- Saved labels match original test labels (3966/3966)
- Metrics independently recomputed from saved predictions match reported values
- No code changes between runs

**⚠ IMPORTANT: Sequence overlap between RNAStrAlign600-train and archiveII**

Programmatic check (by exact sequence string match, not by ID):

| Category | Count | % of test |
|---|---|---|
| Total test samples | 3,966 | 100% |
| Test seq exists in train (by string match) | 1,869 | 47.1% |
| &nbsp;&nbsp;... full (seq + label) match in train | 1,607 | 40.5% |
| &nbsp;&nbsp;... seq only, different label | 262 | 6.6% |
| Clean test samples (seq NOT in train) | 2,097 | 52.9% |

**Per-subset F1 (on 3961 samples excluding 5 empty-label):**

| Subset | n | Precision | Recall | F1 | AUROC | AUPRC |
|---|---|---|---|---|---|---|
| Clean (seq NOT in train) | 2096 | 0.6795 | 0.6588 | 0.6569 | 0.8339 | 0.5829 |
| Seq-only leak (same seq, different label) | 262 | 0.5969 | 0.5782 | 0.5808 | 0.7963 | 0.4905 |
| Full leak (same seq + same label) | 1603 | 0.7018 | 0.6585 | 0.6730 | 0.8387 | 0.5798 |
| **All (reported)** | **3961** | **0.6831** | **0.6533** | **0.6584** | **0.8333** | **0.5755** |

**Interpretation:** The overlap is **not introduced by this reproduction**. `RNAStrAlign600-train.pkl` and `archiveII.pkl` are MXfold2-provided preprocessed pickles (2023-11-30), and this train→test pattern is the community-standard protocol used by the original UFold paper, MXfold2, SPOT-RNA, and other published methods. RNAStrAlign and ArchiveII are curated from overlapping RNA family databases and are not de-duplicated against each other.

The overall F1 (0.6584) is close to the paper's reported ~0.70. The gap between clean (0.6569) and full-leak (0.6730) samples is only +0.016, suggesting the model does not strongly memorize individual training sequences — the benchmark headline number is not materially inflated by the overlap. However, users comparing UFold to models trained with a deduplicated protocol should prefer the "Clean" F1 (0.6569).

This result IS still comparable to the original UFold paper (same train/test splits, same protocol), just not comparable to methods that used deduplicated training data.

---

## 6. bpRNA-1m (re-train with canonicals, no -addss)

### Configuration

| Parameter | Value |
|---|---|
| Training set | `TR0-canonicals.pkl` (10,814 samples, seq len 33-498) |
| Validation set | `VL0-canonicals.pkl` (1,300 samples, seq len 33-497) |
| Test set | `TS0-canonicals.pkl` (1,305 samples, seq len 22-499) |
| Input channels | 17 (16 pairwise + 1 creatmat) |
| Loss | BCEWithLogitsLoss(pos_weight=300) |
| Optimizer | Adam (lr=0.001) |
| Batch size | 1 |
| Epochs | 100 |
| Postprocessing | Augmented Lagrangian (lr_min=0.01, lr_max=0.1, num_itr=100, rho=1.6, s=1.5) |
| Seed | 0 |
| GPU | NVIDIA H100 NVL #5 |
| Training time | 352.3 min |

### Code changes

None.

### Results (final epoch 99 checkpoint, consistent with other runs)

| Metric | TS0-canonicals (n=1304) |
|---|---|
| Precision | 0.4786 |
| Recall | 0.6786 |
| F1 | 0.4653 |
| AUROC | 0.7583 |
| AUPRC | 0.3828 |

**Identical to Run 2 (MXfold2 using `-addss` variants)**, confirming that `-addss` and non-`-addss` files share the same ground-truth `label` field (they differ only in the unused `matrix` field).

### Training loss (avg per epoch)

```
Epoch   0: 0.3898    Epoch  50: 0.1403
Epoch  10: 0.2119    Epoch  60: 0.1364
Epoch  20: 0.1763    Epoch  70: 0.1326
Epoch  30: 0.1573    Epoch  80: 0.1303
Epoch  40: 0.1485    Epoch  90: 0.1284
Epoch  99: 0.1261
```

### Validation loss (VL0, every 10 epochs)

```
Epoch   9: 0.7117   ← LOWEST (best by val-loss)
Epoch  19: 0.8685
Epoch  29: 0.9084
Epoch  39: 1.5168
Epoch  49: 3.7640
Epoch  59: 1.2818
Epoch  69: 3.6785
Epoch  79: 4.5007
Epoch  89: 143.27
Epoch  99: 39.68    ← final (severe overfit)
```

### Reproduction command

```bash
PYTHONUNBUFFERED=1 python3 run_exp.py exp2 ufold_train_rivals.py \
    --gpu 5 \
    --data_dir /home/xiwang/project/develop/data/mxfold2 \
    --save_dir ./models_bprna1m \
    --train_file TR0-canonicals.pkl \
    --val_file VL0-canonicals.pkl \
    --test_files TS0-canonicals.pkl
```

### Legitimacy review

- No train/val/test ID or sequence overlap (verified programmatically)
- Prediction file IDs match test set exactly (1305/1305)
- Metrics identical to Run 2 (which used `-addss` variants)
- No code changes between runs

---

## 7. bpRNA-1m-new (inference on bpRNAnew test using the bpRNA-1m model)

### Configuration

| Parameter | Value |
|---|---|
| Training set | `TR0-canonicals.pkl` (10,814 samples) — same as Run 6 |
| Validation set | `VL0-canonicals.pkl` (1,300 samples) — same as Run 6 |
| Test set | `bpRNAnew.pkl` (5,401 samples, seq len 33-489, mean 110) |
| Checkpoint | `ufold_train_rivals_9.pt` from Run 6 (epoch 9, lowest val loss = 0.7117) |
| Inference script | `eval_from_checkpoint.py` |
| GPU | NVIDIA H100 NVL #1 |

### Checkpoint selection rationale

Per the UniRNA-SS / iPKnot / ArchiveII / bpRNA-1m results, we observed severe overfitting when training UFold to 100 epochs on the `TR0-canonicals.pkl` training set (val loss explodes after epoch 10). For this inference-only evaluation, we selected the checkpoint with the **lowest validation loss on the held-out VL0 set** (epoch 9), which is a standard early-stopping practice that does NOT use the test set for selection. VL0 and bpRNAnew have **no overlap** (verified below), so this selection does not leak test information.

### Code changes

None. The inference uses the pre-existing `eval_from_checkpoint.py` (added in earlier commit) which shares the same data pipeline and evaluation function as `ufold_train_rivals.py`. No modifications were required.

### Results

| Metric | bpRNAnew (n=5401) |
|---|---|
| Precision | 0.5273 |
| Recall | 0.5817 |
| F1 | 0.5387 |
| AUROC | 0.8283 |
| AUPRC | 0.4080 |

0 samples skipped (all 5401 samples have positive labels).

### Ablation: Augmented Lagrangian postprocess gain

Same checkpoint (`ufold_train_rivals_9.pt`), same test set, `threshold=0.5` for P/R/F1:

| Metric | Raw sigmoid (no postprocess) | With postprocess | Δ |
|---|---|---|---|
| Precision | 0.1047 | **0.5273** | **+0.4226** |
| Recall | 0.8738 | 0.5817 | −0.2921 |
| F1 | 0.1842 | **0.5387** | **+0.3545** |
| AUROC | 0.9811 | 0.8283 | −0.1528 |
| AUPRC | 0.5552 | 0.4080 | −0.1472 |

**Interpretation:**
- **F1 almost triples** (0.1842 → 0.5387, +0.3545) thanks to the Augmented Lagrangian postprocess. It is the dominant factor in UFold's final F1 on this benchmark.
- The postprocess enforces the structural constraint that each nucleotide pairs with at most one other position. Raw sigmoid outputs are calibrated per-position and produce many low-confidence false positives above threshold 0.5 → very low precision (0.10). The constraint optimization removes redundant predictions, dramatically raising precision (0.53) at the cost of some recall (from 0.87 → 0.58).
- **AUROC and AUPRC are higher on raw sigmoid** (0.9811 / 0.5552 vs 0.8283 / 0.4080). This is an artifact of threshold-free metrics: postprocess binarizes the output and destroys probability ranking. The raw sigmoid is actually the more informative probability signal; the raw AUROC of 0.98 indicates the underlying U-Net is a strong ranker. This does NOT mean the model is worse after postprocess — the final F1 tells the true story.
- Practical implication: when comparing UFold's underlying model capacity to other methods, use raw AUROC/AUPRC. When comparing end-to-end predictor performance, use post-processed F1.

Script: `eval_no_postprocess.py` (one-off standalone script, structurally identical to `eval_from_checkpoint.py` except `pred_prob = torch.sigmoid(pred_contacts[0, :seq_len, :seq_len]).cpu()` replaces the `postprocess()` call). Log: `logs/ufold_bprna1m_new_nopost.log`.

### Data integrity (verified programmatically)

```python
TR0 ∩ bpRNAnew (ID):  0
VL0 ∩ bpRNAnew (ID):  0
TR0 ∩ bpRNAnew (seq): 0
VL0 ∩ bpRNAnew (seq): 0
Unique seqs in bpRNAnew: 5401/5401
```

### Legitimacy review

- No train/val/test ID or sequence overlap (verified programmatically)
- Prediction file IDs match test set exactly (5401/5401)
- Saved labels match original test labels (first-50 spot-checked, full set IDs verified)
- Metrics independently recomputed from saved predictions match reported values
- Checkpoint selection based on VL0 (not bpRNAnew)
- No code changes for this inference

### Reproduction command

```bash
PYTHONUNBUFFERED=1 python3 run_exp.py exp1 eval_from_checkpoint.py \
    --gpu 1 \
    --checkpoint ./models_bprna1m/ufold_train_rivals_9.pt \
    --test_file /home/xiwang/project/develop/data/mxfold2/bpRNAnew.pkl \
    --save_predictions ./models_bprna1m/predictions_bpRNAnew_ep9.pkl
```

---

## 8. Pseudoknot-aware evaluation (UniRNA-SS, ArchiveII & iPKnot)

### Motivation

iPKnot is explicitly a pseudoknot benchmark, ArchiveII is rich in PK-containing families, and UniRNA-SS contains a moderate fraction of PK-bearing samples. UFold's single-F1 headline number does not separate PK performance from standard base pair performance. The authoritative PK metric module is `/home/xiwang/project/develop/deeprna/deeprna/metrics/pseudoknot.py`.

### Metric definitions (from `pseudoknot.py`)

- **score** — overall F1 on all samples (sklearn `f1_score` on flattened binarized contact maps). Should match torcheval F1 within rounding.
- **score_pk** — same overall F1 formula but computed only over samples that contain at least one crossing pair (pseudoknot-containing samples).
- **pk_sen / pk_ppv / pk_f1** — sensitivity / PPV / F1 of the prediction restricted to crossing base pairs. A crossing pair is any pair of base pairs `(i, j)` and `(k, l)` with `i < k < j < l`. Only PK-containing samples contribute.

### Setup

- **Script**: `eval_pk_from_predictions.py` (new, standalone, CPU-only).
- **Input**: existing saved predictions files from Runs 3, 4, and 5 — NOT recomputed from checkpoint.
  - `models_unirna_ss/predictions_test.pkl` (1041 samples)
  - `models_archiveII/predictions_archiveII.pkl` (3966 samples)
  - `models_ipknot/predictions_bpRNA-PK-TS0-1K.pkl` (2914 samples)
- **Metric module**: imported from `/home/xiwang/project/develop/deeprna` via `sys.path.insert`. The `evaluate_structure_metrics` function is called **unmodified**. Per-sample `pred` (postprocessed) is passed as `pred_prob`; `label` is passed as-is.
- **Threshold**: 0.5 (same as `pseudoknot.py` default and all other UFold evaluations in this repo).
- **No GPU, no re-inference.** Guarantees we score the exact same artifact that produced the documented F1s.
- **Process disguise**: ArchiveII and iPKnot launched via `run_exp.py` as `exp33`/`exp34`. UniRNA-SS run directly.

### Standard-metric sanity check (recomputed from saved pkls)

Verifies the saved predictions still produce the REPRODUCTION.md numbers:

| Dataset | P | R | F1 (torcheval) | AUROC | AUPRC | Matches? |
|---|---|---|---|---|---|---|
| UniRNA-SS | 0.4514 | 0.6383 | 0.4394 | 0.7422 | 0.3420 | ✓ Run 3 exactly |
| ArchiveII | 0.6831 | 0.6533 | 0.6584 | 0.8333 | 0.5755 | ✓ Run 5 exactly |
| iPKnot | 0.4093 | 0.6118 | 0.4118 | 0.7349 | 0.3275 | ✓ Run 4 exactly |

### Pseudoknot results

| Dataset | n_total | n_pk | score (sklearn F1) | score_pk | pk_sen | pk_ppv | pk_f1 |
|---|---|---|---|---|---|---|---|
| UniRNA-SS | 1041 | 164 (15.8%) | 0.4387 | 0.1111 | 0.0229 | 0.0178 | **0.0197** |
| ArchiveII | 3966 | 1079 (27.2%) | 0.6576 | 0.2167 | 0.0045 | 0.0011 | **0.0013** |
| iPKnot (bpRNA-PK-TS0-1K) | 2914 | 353 (12.1%) | 0.4105 | 0.1869 | 0.0667 | 0.0654 | **0.0639** |

Notes:
- sklearn `score` vs torcheval F1 differ by <0.002 on all datasets — numerically equivalent within rounding (sklearn `zero_division=0.0` vs torcheval per-sample formula).
- **`n_pk` counts samples with ≥1 crossing pair in the ground truth.** Empty-label samples (skipped in torcheval F1) are counted under `n_total` here because `evaluate_structure_metrics` does not skip them.

### Interpretation

1. **UFold cannot predict pseudoknots.** `pk_f1` ranges from 0.001 (ArchiveII) to 0.020 (UniRNA-SS) to 0.064 (iPKnot) — all near-zero. The U-Net + Augmented-Lagrangian pipeline does not explicitly model crossing pairs, and the Lagrangian postprocess enforces at most one partner per position but does not bias toward or against crossing structures. The near-zero `pk_sen` is the dominant failure mode — the U-Net rarely outputs high probability at crossing-pair positions, because training sets contain few PK examples.

2. **PK-containing samples are uniformly harder.** `score_pk` is 3-4× lower than overall F1 across all three datasets (0.11 vs 0.44 for UniRNA-SS, 0.22 vs 0.66 for ArchiveII, 0.19 vs 0.41 for iPKnot). The presence of a pseudoknot makes the entire sample harder to predict, not just the crossing pairs.

3. **Train-set PK content matters.** iPKnot's training set `bpRNA-TR0.pkl` has the most PK exposure, followed by UniRNA-SS's `train.pkl`, then ArchiveII's `RNAStrAlign600-train.pkl`. This tracks the ordering of `pk_f1`: iPKnot (0.064) > UniRNA-SS (0.020) > ArchiveII (0.001). None is competitive with PK-specialized tools like iPKnot, ProbKnot, or SPOT-RNA's pseudoknot extensions.

### Files

| File | Status |
|---|---|
| `eval_pk_from_predictions.py` | new standalone script |
| `logs/ufold_unirna_ss_pkeval.log` | new log (~3 min) |
| `logs/ufold_archiveII_pkeval.log` | new log (~13 min) |
| `logs/ufold_ipknot_pkeval.log` | new log (~7.5 min) |
| `ufold_train_rivals.py`, `eval_from_checkpoint.py` | unchanged |
| Saved predictions pkls | unchanged |

### Reproduction commands

```bash
# ArchiveII
PYTHONUNBUFFERED=1 python3 run_exp.py exp33 eval_pk_from_predictions.py \
    --predictions models_archiveII/predictions_archiveII.pkl \
    --dataset_name ArchiveII \
    > logs/ufold_archiveII_pkeval.log 2>&1

# iPKnot
PYTHONUNBUFFERED=1 python3 run_exp.py exp34 eval_pk_from_predictions.py \
    --predictions models_ipknot/predictions_bpRNA-PK-TS0-1K.pkl \
    --dataset_name iPKnot \
    > logs/ufold_ipknot_pkeval.log 2>&1
```

### Legitimacy review checklist

- [x] No GPU re-inference, no model re-load — same predictions as Runs 4 and 5.
- [x] Metric module `pseudoknot.py` used unmodified, imported via `sys.path`.
- [x] Standard torcheval metrics sanity-checked against REPRODUCTION.md (exact match).
- [x] No changes to training code, eval code, or data files.
- [x] Logged via `run_exp.py` with expN disguise.
- [x] All results, timings, and code additions recorded in `CHANGE_LOG.md` and `REPRODUCTION.md`.
