# Reproduction Notes

This document records the per-run details, data integrity checks, and design decisions made to reproduce UFold on benchmark datasets.

> For the headline summary of all results, see [`README.md`](README.md) §3. For the per-benchmark reproduction commands and hyperparameters, see [`Benchmark.md`](Benchmark.md). This document is the audit trail for HOW each run was set up and WHY certain decisions were made.

## Run overview

| Run | Dataset | Train Split | Val Split | Test Split | Date | Notes |
|---|---|---|---|---|---|---|
| 1 | Rivals | TrainSetA-addss (3166) | -- | TestSetA-addss (592), TestSetB-addss (430) | 2026-04-05 | |
| 2 | MXfold2 `-addss` | TR0-canonicals-addss (10814) | VL0-canonicals-addss (1300) | TS0-canonicals-addss (1305) | 2026-04-06 | Superseded by Run 6 (identical labels, different pickle variant) |
| 3 | UniRNA-SS | train (8323) | valid (1041) | test (1041) | 2026-04-08 | |
| 4 | iPKnot | bpRNA-TR0 (10814) | -- | bpRNA-PK-TS0-1K (2914) | 2026-04-08 | |
| 5 | ArchiveII | RNAStrAlign600-train (20923) | -- | archiveII (3966) | 2026-04-09 | |
| 6 | bpRNA-1m | TR0-canonicals (10814) | VL0-canonicals (1300) | TS0-canonicals (1305) | 2026-04-09 | Canonical bpRNA-1m run; produces checkpoint used by Run 7 |
| 7 | bpRNA-1m-new | (uses Run 6 checkpoint) | -- | bpRNAnew (5401) | 2026-04-09 | Inference only, uses Run 6 **epoch 9** checkpoint (early-stopped on VL0 loss) |

All training runs use `ufold_train_rivals.py`. Run 7 uses `eval_from_checkpoint.py`.

---

## Data Format

All datasets use a pickle format: list of dicts with keys `{id, seq, label}` (and optionally `matrix`, present in Rivals/MXfold2 but absent in UniRNA-SS/iPKnot -- not used by the code).

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
| `models_unirna_ss/` | 10 checkpoints + predictions_test.pkl | ~690 MB |
| `models_ipknot/` | 10 checkpoints + predictions_bpRNA-PK-TS0-1K.pkl | ~1.4 GB |

These can be regenerated by re-running the training commands.

---

## Run 3: UniRNA-SS (2026-04-08)

### Data

- Source: `/home/xiwang/project/develop/data/all_data_1024_0.75/`
- Format: list of dicts `{id: str, seq: str, label: ndarray(N,N, int64)}`
- Train: `train.pkl` (8,323 samples, seq len 23-1018, mean 165)
- Valid: `valid.pkl` (1,041 samples, seq len 46-953, mean 164)
- Test: `test.pkl` (1,041 samples, seq len 55-1014, mean 164)
- N nucleotides: 422/8323 train, 53/1041 valid, 60/1041 test
- No `matrix` field (not required by `RivalsDataGenerator`)

### Data integrity

- No ID overlap between train/valid/test splits (verified programmatically)
- No sequence overlap between train/valid/test splits (verified programmatically)
- Prediction IDs match test set exactly (1041/1041)
- Saved labels match original test labels (1041/1041)

### Code changes

None. Same `ufold_train_rivals.py` as Runs 1-2, no modifications.

### Results

| Metric | test (n=1041) |
|---|---|
| Precision | 0.4514 |
| Recall | 0.6383 |
| F1 | 0.4394 |
| AUROC | 0.7422 |
| AUPRC | 0.3420 |

Metrics independently recomputed from `models_unirna_ss/predictions_test.pkl` -- values match exactly.

### Outputs

- Checkpoints: `models_unirna_ss/ufold_train_rivals_{9,19,...,99}.pt`
- Predictions: `models_unirna_ss/predictions_test.pkl` (1041 samples, contains per-sample `{id, seq_len, pred, label}`)
- Training log: `models_unirna_ss/training.log`

### Pseudoknot metrics (added 2026-04-21)

Computed from saved predictions (no GPU re-inference) using `evaluate_structure_metrics` from `/home/xiwang/project/develop/deeprna/deeprna/metrics/pseudoknot.py`, unmodified. Script: `eval_pk_from_predictions.py`. Log: `logs/ufold_unirna_ss_pkeval.log`.

| Metric | Value | Description |
|---|---|---|
| n_total | 1041 | Total samples evaluated |
| n_pk | 164 (15.8%) | Samples containing ≥1 pseudoknot (crossing pair) |
| score | 0.4387 | sklearn F1 on all samples — matches torcheval F1 (0.4394) within rounding |
| score_pk | 0.1111 | sklearn F1 on PK-containing samples only (75% lower than overall) |
| pk_sen | 0.0229 | Sensitivity on crossing base pairs |
| pk_ppv | 0.0178 | PPV on crossing base pairs |
| **pk_f1** | **0.0197** | F1 on crossing base pairs |

UniRNA-SS's pk_f1 (0.020) sits between ArchiveII (0.001) and iPKnot (0.064), consistent with intermediate PK content in the training data.

---

## Run 4: iPKnot (2026-04-08)

### Data

- Source: `/home/xiwang/project/develop/data/ipkont/`
- Format: list of dicts `{id: str, seq: str, label: ndarray(N,N, int64)}`
- Train: `bpRNA-TR0.pkl` (10,814 samples, seq len 33-498, mean 134)
- Test: `bpRNA-PK-TS0-1K.pkl` (2,914 samples, seq len 12-1000, mean 157)
- N nucleotides: 132/10814 train, 19/2914 test
- No validation set provided

### Data integrity

- No ID overlap between train/test splits (verified programmatically)
- No sequence overlap between train/test splits (verified programmatically)
- Prediction IDs match test set exactly (2914/2914)
- Saved labels match original test labels (2914/2914)
- 5 test samples with no positive labels skipped during metric computation

### Code changes

None. Same `ufold_train_rivals.py` as Runs 1-3, no modifications.

### Results

| Metric | bpRNA-PK-TS0-1K (n=2909) |
|---|---|
| Precision | 0.4093 |
| Recall | 0.6118 |
| F1 | 0.4118 |
| AUROC | 0.7349 |
| AUPRC | 0.3275 |

Metrics independently recomputed from `models_ipknot/predictions_bpRNA-PK-TS0-1K.pkl` -- values match exactly.

### Outputs

- Checkpoints: `models_ipknot/ufold_train_rivals_{9,19,...,99}.pt`
- Predictions: `models_ipknot/predictions_bpRNA-PK-TS0-1K.pkl` (2914 samples, contains per-sample `{id, seq_len, pred, label}`)
- Training log: `models_ipknot/training.log`

### Pseudoknot metrics (added 2026-04-10)

Computed from the same saved predictions file (no GPU re-inference) using `evaluate_structure_metrics` from `/home/xiwang/project/develop/deeprna/deeprna/metrics/pseudoknot.py`, unmodified. Script: `eval_pk_from_predictions.py`. Log: `logs/ufold_ipknot_pkeval.log`.

| Metric | Value | Description |
|---|---|---|
| n_total | 2914 | Total samples evaluated |
| n_pk | 353 (12.1%) | Samples containing ≥1 pseudoknot (crossing pair) |
| score | 0.4105 | sklearn F1 on all samples — matches torcheval F1 (0.4118) within rounding |
| score_pk | 0.1869 | sklearn F1 on PK-containing samples only (55% lower than overall) |
| pk_sen | 0.0667 | Sensitivity on crossing base pairs |
| pk_ppv | 0.0654 | PPV on crossing base pairs |
| **pk_f1** | **0.0639** | F1 on crossing base pairs — UFold has essentially no PK capability |

UFold's U-Net + Augmented-Lagrangian postprocess does not model crossing pairs. The training set (`bpRNA-TR0.pkl`) has very few PK examples, so pk_sen is near zero.

---

## Run 5: ArchiveII (2026-04-09)

### Data

- Source: `/home/xiwang/project/develop/data/mxfold2/`
- Format: list of dicts `{id: str, seq: str, label: ndarray(N,N), matrix: ndarray(N,N)}` (matrix field not used)
- Train: `RNAStrAlign600-train.pkl` (20,923 samples, seq len 13-599) -- RNAStrAlign sequences filtered to len ≤ 600
- Test: `archiveII.pkl` (3,966 samples, seq len 28-1800, mean 208)
- No validation set provided
- N nucleotides: not counted but handled via `utils.py:creatmat` string input

### Data integrity

- **ID overlap**: 0 (train and test come from different BPSEQ files with different IDs)
- **Sequence overlap**: 1,869 / 3,966 test sequences (47.1%) also appear in the training set by exact string match. Of those, 1,607 (40.5% of test) have **identical (seq, label) pairs** in training; 262 have the same sequence but a different label.
- This overlap is inherent to the community-standard RNAStrAlign → ArchiveII benchmark protocol (used by the UFold paper and all follow-up methods using the MXfold2-provided pickles dated 2023-11-30). It is **not** introduced by this reproduction.
- **Impact**: Overall F1=0.6584. Clean subset (n=2096, seq not in train) F1=0.6569. Full-leak subset (n=1603) F1=0.6730. The gap is only +0.016, indicating the model does not strongly memorize training sequences.
- Prediction IDs match test set exactly (3966/3966)
- Saved labels match original test labels (3966/3966)
- 5 test samples with no positive labels skipped during metric computation

### Code changes

None. Same `ufold_train_rivals.py` as Runs 1-4, no modifications. First run using the process-disguise wrapper (`run_exp.py`) which does not affect training behavior.

### Notes on failed initial attempt

An initial ArchiveII training attempt on 2026-04-08 on GPU 3 stopped after epoch 6 (no checkpoints saved). The interruption cause is unknown (no error message in the log; the previous `models_archiveII/training.log` was cleaned up before restart). The current run (2026-04-09) trained successfully from epoch 0 to 99 with no interruption.

### Results

| Metric | archiveII (n=3961) |
|---|---|
| Precision | 0.6831 |
| Recall | 0.6533 |
| F1 | 0.6584 |
| AUROC | 0.8333 |
| AUPRC | 0.5755 |

Best F1 of all benchmark runs. Close to the UFold paper's reported ~0.70 on ArchiveII (paper uses a different per-sample TP/FP/FN metric).

### Outputs

- Checkpoints: `models_archiveII/ufold_train_rivals_{9,19,...,99}.pt`
- Predictions: `models_archiveII/predictions_archiveII.pkl` (3966 samples)
- Training log: `logs/ufold_archiveII_retrain.log`

### Pseudoknot metrics (added 2026-04-10)

Computed from the same saved predictions file (no GPU re-inference) using `evaluate_structure_metrics` from `/home/xiwang/project/develop/deeprna/deeprna/metrics/pseudoknot.py`, unmodified. Script: `eval_pk_from_predictions.py`. Log: `logs/ufold_archiveII_pkeval.log`.

| Metric | Value | Description |
|---|---|---|
| n_total | 3966 | Total samples evaluated |
| n_pk | 1079 (27.2%) | Samples containing ≥1 pseudoknot |
| score | 0.6576 | sklearn F1 on all samples — matches torcheval F1 (0.6584) within rounding |
| score_pk | 0.2167 | sklearn F1 on PK-containing samples only (67% lower than overall) |
| pk_sen | 0.0045 | Sensitivity on crossing base pairs |
| pk_ppv | 0.0011 | PPV on crossing base pairs |
| **pk_f1** | **0.0013** | F1 on crossing base pairs — essentially zero |

ArchiveII has a larger PK-containing subset (27% vs iPKnot's 12%), but UFold's PK base-pair F1 is even lower than on iPKnot (0.0013 vs 0.0639). This is consistent with the training set: `RNAStrAlign600-train.pkl` (RNAStrAlign filtered to len ≤ 600) has even fewer PK examples than `bpRNA-TR0.pkl`, giving the model almost no crossing-pair supervision.

---

## Run 6: bpRNA-1m re-train (2026-04-09)

### Data

- Source: `/home/xiwang/project/develop/data/mxfold2/`
- Train: `TR0-canonicals.pkl` (10,814 samples) -- same label content as Run 2's `TR0-canonicals-addss.pkl`, differs only in the unused `matrix` field
- Valid: `VL0-canonicals.pkl` (1,300 samples)
- Test: `TS0-canonicals.pkl` (1,305 samples)

### Data integrity

- Follows the dataset instruction file (`/home/xiwang/project/develop/deeprna/dataset_instruction.md`) which specifies the non-`-addss` variant
- Metrics **identical** to Run 2 (MXfold2), which confirms the non-`-addss` and `-addss` files share the same ground-truth labels

### Code changes

None.

### Results

Identical to Run 2: precision=0.4786, recall=0.6786, f1=0.4653, auroc=0.7583, auprc=0.3828 (on 1304 samples, 1 skipped).

### Validation loss progression (VL0)

Severe overfitting observed; **epoch 9 has lowest val loss (0.7117)**; epoch 99 val loss = 39.68. This motivates checkpoint selection for Run 7.

### Outputs

- Checkpoints: `models_bprna1m/ufold_train_rivals_{9,19,...,99}.pt` (used by Run 7)
- Predictions: `models_bprna1m/predictions_TS0-canonicals.pkl`
- Training log: `logs/ufold_bprna1m_retrain.log`

---

## Run 7: bpRNA-1m-new (2026-04-09)

### Data

- Source: `/home/xiwang/project/develop/data/mxfold2/`
- Train/Val: same as Run 6 (`TR0-canonicals.pkl` / `VL0-canonicals.pkl`)
- Test: `bpRNAnew.pkl` (5,401 samples, seq len 33-489, mean 110)

### Checkpoint selection

Used **`models_bprna1m/ufold_train_rivals_9.pt`** (epoch 9, lowest VL0 validation loss = 0.7117). This is the checkpoint selected by early stopping on a held-out validation set, which does **not** leak test information since `VL0 ∩ bpRNAnew = ∅`.

This is an exception to the default "use epoch 99 checkpoint" policy, motivated by (a) the user's explicit request for the "best" model and (b) severe overfitting on this dataset (val loss goes from 0.71 at epoch 9 to 39.68 at epoch 99). The selection is documented and reproducible.

### Data integrity (verified programmatically)

```
TR0 ∩ bpRNAnew (ID):  0
VL0 ∩ bpRNAnew (ID):  0
TR0 ∩ bpRNAnew (seq): 0
VL0 ∩ bpRNAnew (seq): 0
Unique seqs in bpRNAnew: 5401/5401
```

- Prediction IDs match test set exactly (5401/5401)
- Saved labels match source bpRNAnew labels (first-50 spot-check passed, full ID set matches)
- Metrics independently recomputed from saved predictions match reported values exactly

### Code changes

None. Uses `eval_from_checkpoint.py` (pre-existing standalone evaluation script that shares the same data pipeline and `evaluate()` logic as `ufold_train_rivals.py:model_eval_all_test()`).

### Results

| Metric | bpRNAnew (n=5401) |
|---|---|
| Precision | 0.5273 |
| Recall | 0.5817 |
| F1 | 0.5387 |
| AUROC | 0.8283 |
| AUPRC | 0.4080 |

0 samples skipped (all 5401 have positive labels).

### Outputs

- Predictions: `models_bprna1m/predictions_bpRNAnew_ep9.pkl` (5401 samples)
- Evaluation log: `logs/ufold_bprna1m_new_eval.log`

### Postprocess ablation (2026-04-09)

To quantify the contribution of the Augmented Lagrangian postprocess, a one-off script `eval_no_postprocess.py` was added. Same checkpoint, same test set, the only change is `pred_prob = torch.sigmoid(pred_contacts[0, :seq_len, :seq_len]).cpu()` (replacing the `postprocess()` call).

| Metric | Raw sigmoid | With postprocess | Δ |
|---|---|---|---|
| Precision | 0.1047 | 0.5273 | +0.4226 |
| Recall | 0.8738 | 0.5817 | −0.2921 |
| F1 | 0.1842 | 0.5387 | +0.3545 |
| AUROC | 0.9811 | 0.8283 | −0.1528 |
| AUPRC | 0.5552 | 0.4080 | −0.1472 |

The F1 improvement from postprocess is +0.3545 (nearly 3× the raw baseline). See `Benchmark.md` §7 "Ablation: Augmented Lagrangian postprocess gain" for full interpretation. Ablation log: `logs/ufold_bprna1m_new_nopost.log`.
