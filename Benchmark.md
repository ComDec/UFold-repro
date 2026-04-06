# Benchmark Results

All experiments use the UFold U-Net architecture (img_ch=17, 8.6M params) with official default hyperparameters.
Metrics are consistent with `DeepRNA/deepprotein/tasks/utils.py:secondary_structure_metircs`.

## Summary Table

| Dataset | Train | Test | Precision | Recall | F1 | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|
| Rivals | TrainSetA (3166) | TestSetA (592) | 0.7084 | -- | 0.6343 | 0.8127 | 0.5167 |
| Rivals | TrainSetA (3166) | TestSetB (430) | 0.5428 | -- | 0.4145 | 0.6890 | 0.2562 |
| MXfold2 | TR0 (10814) | TS0 (1305, 1 skipped) | 0.4786 | 0.6786 | 0.4653 | 0.7583 | 0.3828 |

> Notes:
> - Rivals run did not include recall metric (added in later code revision). MXfold2 run includes recall.
> - Samples with no positive labels are skipped (AUROC/AUPRC undefined). This differs from DeepRNA's `secondary_structure_metircs` which does not skip such samples.
> - DeepRNA has `binary_recall` commented out due to a torcheval bug workaround; we include it.

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
