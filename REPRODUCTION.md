# Reproduction Notes

This document records the exact steps, code changes, and design decisions made to reproduce UFold on the Rivals and MXfold2 benchmark datasets.

## Overview

| Run | Dataset | Train Split | Val Split | Test Split | Script | Date |
|---|---|---|---|---|---|---|
| 1 | Rivals | TrainSetA (3166) | -- | TestSetA (592), TestSetB (430) | `ufold_train_rivals.py` | 2026-04-05 |
| 2 | MXfold2 | TR0-canonicals (10814) | VL0-canonicals (1300) | TS0-canonicals (1305) | `ufold_train_rivals.py` | 2026-04-06 |

---

## Data Format

Both datasets use the same pickle format: list of dicts with keys `{id, seq, label, matrix}`.

- **`seq`** (str): RNA nucleotide sequence (A, U, C, G; some contain N).
- **`label`** (ndarray, int64, NxN): Binary symmetric ground-truth contact map. Verified symmetric via `np.allclose(label, label.T)`.
- **`matrix`** (ndarray, float32, NxN): External method predictions. **Not used** -- only `seq` and `label` are consumed.

### Label verification

```python
# Ground-truth consistency across file variants (verified for rivals data):
addss = pickle.load(open('TrainSetA-addss.pkl', 'rb'))
rnafold = pickle.load(open('TrainSetA-rnafold.pkl', 'rb'))
assert np.array_equal(addss[0]['label'], rnafold[0]['label'])  # True -- same GT

hardlabels = pickle.load(open('TrainSetA-ernierna-hardlabels.pkl', 'rb'))
assert not np.array_equal(addss[0]['label'], hardlabels[0]['label'])  # Different -- model preds
assert not np.allclose(hardlabels[0]['label'], hardlabels[0]['label'].T)  # Asymmetric
```

We use `*-addss.pkl` files which contain verified symmetric ground-truth labels.

### N nucleotide handling

| Dataset | Samples with N |
|---|---|
| Rivals TrainSetA | 2 / 3166 |
| Rivals TestSetA | 14 / 592 |
| Rivals TestSetB | 2 / 430 |
| MXfold2 TR0 | 132 / 10814 |
| MXfold2 VL0 | 14 / 1300 |
| MXfold2 TS0 | 17 / 1305 |

N nucleotides are encoded as all-zero vectors (`seq_dict['N'] = [0,0,0,0]`), producing zero features across all 17 channels. `paired('N', x) = 0` for any `x`. This is consistent with UFold's original handling.

---

## Code Changes

### Files added

| File | Purpose | Lines |
|---|---|---|
| `ufold_train_rivals.py` | Training + evaluation script for benchmark pickle format | ~340 |

### Files from original UFold (unmodified)

| File | Purpose |
|---|---|
| `Network.py` | U-Net architecture |
| `ufold/utils.py` | `creatmat`, `seq_dict`, `seed_torch`, `get_args` |
| `ufold/postprocess.py` | Augmented Lagrangian postprocessing |
| `ufold/data_generator.py` | `get_cut_len` utility function |
| `ufold/config.py`, `ufold/config.json` | Configuration |
| `ufold_train.py` | Original training script (reference only) |

### Key design decisions in `ufold_train_rivals.py`

1. **Data adapter** (`RivalsDataGenerator`): Reads `{seq, label}` dicts instead of `RNA_SS_data` namedtuples. Mimics `RNASSDataGenerator` interface (`get_one_sample()` returns same signature).

2. **creatmat via string input**: The original `data_generator.py:880` defines a GPU-accelerated `creatmat` that converts one-hot arrays to strings internally. However, it crashes on N nucleotides (`[0,0,0,0].index(1)` raises `ValueError`). We use `utils.py:creatmat` with string sequences directly, which handles N correctly. Pre-computed at data load time to amortize cost.

3. **17-channel features**: Identical to original UFold.
   - Channels 0-15: pairwise outer products via `np.matmul(onehot[:, i].reshape(-1, 1), onehot[:, j].reshape(1, -1))` for all 16 `(i, j)` in `product(range(4), range(4))`.
   - Channel 16: `creatmat(seq_string)` thermodynamic feature.

4. **Training loop**: Identical to `ufold_train.py:train()`. Same loss (`BCEWithLogitsLoss(pos_weight=300)`), optimizer (`Adam`), masking (`contact_masks[:, :seq_lens, :seq_lens] = 1`).

5. **Evaluation**: Flattens prediction and label matrices per-sample, computes `binary_precision`, `binary_recall`, `binary_f1_score` (threshold=0.5), `binary_auroc`, `binary_auprc` via `torcheval`, then macro-averages. Postprocessing applied before metrics (same as `ufold_test.py`). Samples with no positive labels are skipped (AUROC/AUPRC undefined). Note: DeepRNA's `secondary_structure_metircs` does not skip such samples (which would produce NaN or crash) and has `binary_recall` commented out due to a torcheval bug workaround.

6. **Postprocessing slicing**: The evaluation slices predictions to `[:seq_len, :seq_len]` before postprocessing, while the original `ufold_test.py` passes the full padded matrix. These are numerically equivalent because the constraint matrix `m` (from `constraint_matrix_batch`) zeros out padding positions via the one-hot outer products.

6. **Padding**: Uses `get_cut_len(data_len, 80)` from `data_generator.py` -- rounds up to multiple of 16, minimum 80. Dynamic re-padding when `l >= seq_max_len` (vs original's hardcoded `l >= 500`).

---

## Hyperparameters

All hyperparameters match UFold official defaults from the codebase:

| Parameter | Value | Source file:line |
|---|---|---|
| `pos_weight` | 300 | `ufold_train.py:34` |
| Optimizer | Adam | `ufold_train.py:37` |
| Learning rate | 0.001 (Adam default) | `ufold_train.py:37` |
| Batch size | 1 | `ufold/config.json:batch_size_stage_1` |
| Epochs | 100 | `ufold/config.json:epoches_first` |
| DataLoader workers | 6 | `ufold_train.py:175` |
| Shuffle | True | `ufold_train.py:174` |
| drop_last | True | `ufold_train.py:176` |
| Seed | 0 | `ufold/utils.py:seed_torch()` |
| PP lr_min | 0.01 | `ufold_test.py` |
| PP lr_max | 0.1 | `ufold_test.py` |
| PP num_itr | 100 | `ufold_test.py` |
| PP rho | 1.6 | `ufold_test.py` |
| PP s | 1.5 | `ufold_test.py` |
| PP with_l1 | True | `ufold_test.py` |

No hyperparameters were tuned or changed from the original codebase.

---

## Data Integrity Checks

- **No test-set training**: Training uses only the designated train split. Test data is loaded separately after training completes.
- **No silent test-set substitution**: Test file names are explicitly passed via `--test_files` CLI argument and printed in the output log.
- **No data leakage**: The `RivalsDataGenerator` loads each pickle file independently. No cross-file data mixing occurs.
- **Label ground truth verified**: Labels in `*-addss.pkl` confirmed as symmetric binary matrices consistent across file variants.
- **Deterministic**: `seed_torch(0)` fixes all random seeds (numpy, torch, CUDA).

---

## Intermediate Files

The following intermediate files are generated during training and excluded from git via `.gitignore`:

| Directory | Contents | Size |
|---|---|---|
| `models_rivals/` | 10 checkpoints (every 10 epochs) | ~340 MB |
| `models_mxfold2/` | 10 checkpoints (every 10 epochs) | ~340 MB |

These can be regenerated by re-running the training commands.
