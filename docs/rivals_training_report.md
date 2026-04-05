# UFold Reproducibility Report: Training on Rivals Benchmark

**Date**: 2026-04-05
**Script**: `ufold_train_rivals.py`
**Checkpoints**: `models_rivals/ufold_train_rivals_*.pt`

---

## 1. Objective

Reproduce UFold training and evaluation on the Rivals benchmark dataset. The model uses the full 17-channel UFold feature representation (16 pairwise outer-product channels + 1 creatmat thermodynamic channel), with the ground-truth `label` field as the supervision target. Evaluation metrics are consistent with `DeepRNA/deepprotein/tasks/utils.py:secondary_structure_metircs`.

---

## 2. Environment

| Component | Version |
|---|---|
| Python | 3.13.12 |
| PyTorch | 2.8.0+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA H100 NVL (95830 MiB) |

---

## 3. Data

### 3.1 Format

Each rivals pickle file contains a list of dicts:

```python
{'id': str, 'seq': str, 'label': np.ndarray(N,N, int64), 'matrix': np.ndarray(N,N, float32)}
```

Only `seq` and `label` are used. The `matrix` field (external method predictions) is ignored.

### 3.2 Label Verification

We verified that the `label` field in `*-addss.pkl` is ground truth by cross-checking with other file variants:

```python
# Labels are identical between addss and rnafold files (both are ground truth)
np.array_equal(addss[0]['label'], rnafold[0]['label'])  # True

# Labels differ in ernierna-hardlabels (model predictions, not ground truth)
np.array_equal(addss[0]['label'], hardlabels[0]['label'])  # False
```

All labels are verified to be symmetric binary matrices (`np.allclose(label, label.T)`).

### 3.3 Dataset Statistics

| Dataset | Samples | Seq Length | Mean | Median | Label Density |
|---|---|---|---|---|---|
| **TrainSetA** | 3,166 | [10, 734] | 199.1 | 114 | 1.19% |
| **TestSetA** | 592 | [10, 768] | 164.5 | 74 | 1.39% |
| **TestSetB** | 430 | [27, 244] | 121.2 | 109 | 0.41% |

### 3.4 N Nucleotide Handling

18 samples contain `N` nucleotides (TrainSetA: 2, TestSetA: 14, TestSetB: 2). They are handled consistently across all 17 channels:

- **One-hot**: `seq_dict['N'] = [0,0,0,0]` (from `ufold/utils.py`) -- zero vector, no information.
- **16 pairwise channels**: outer products with a zero vector produce all zeros.
- **creatmat channel**: `paired('N', x)` returns 0 for any `x` (no match in the pairing rules) -- zero thermodynamic score.

All 17 feature channels are zero at N positions, making them effectively invisible to the model.

**Why the utils.py creatmat**: The `data_generator.py` also defines a GPU-accelerated `creatmat` (line 880) that internally converts one-hot arrays to character strings. However, it crashes on N nucleotides because `[0,0,0,0].index(1)` raises `ValueError`. We therefore use the `utils.py` string-based version, which handles N correctly. Pre-computing all creatmat matrices at data loading time amortizes the O(L^2) cost.

---

## 4. Implementation

### 4.1 Minimal-Diff Approach

The script `ufold_train_rivals.py` is derived from the original `ufold_train.py` with only the data-loading layer replaced:

| Component | Original (`ufold_train.py`) | Rivals (`ufold_train_rivals.py`) |
|---|---|---|
| Data loader | `RNASSDataGenerator` + `Dataset_Cut_concat_new_merge_multi` | `RivalsDataGenerator` + `RivalsDataset` |
| Feature channels | 17 (16 pairwise + 1 creatmat) | 17 (identical) |
| Model | `U_Net(img_ch=17)` | `U_Net(img_ch=17)` (identical) |
| Loss | `BCEWithLogitsLoss(pos_weight=300)` | identical |
| Optimizer | `Adam(lr=0.001)` | identical |
| Training loop | `train()` | identical logic |
| Evaluation | N/A (separate script) | Added `model_eval_all_test()` |

### 4.2 Feature Construction

The 17-channel input tensor is constructed identically to the original UFold:

**Channels 0-15** -- Pairwise outer products of one-hot encoded nucleotides:

```python
perm = list(product(np.arange(4), np.arange(4)))  # 16 combinations: (A,A), (A,U), ..., (G,G)
for n, (i, j) in enumerate(perm):
    data_fcn[n, :L, :L] = np.matmul(onehot[:, i].reshape(-1, 1), onehot[:, j].reshape(1, -1))
```

**Channel 16** -- Thermodynamic feature (`creatmat` from `ufold/utils.py`):

```python
# Pairing scores: AU/UA=2, GC/CG=3, GU/UG=0.8, others=0
# For each (i,j), sum scores along stacking direction with Gaussian distance weighting
creatmat_str(seq_string)  # returns (L, L) matrix
```

### 4.3 Training Configuration

All hyperparameters follow UFold defaults:

| Parameter | Value | Source |
|---|---|---|
| Loss | `BCEWithLogitsLoss(pos_weight=300)` | `ufold_train.py:34-36` |
| Optimizer | Adam (lr=0.001) | `ufold_train.py:37` |
| Batch size | 1 | `config.json` |
| Epochs | 100 | `config.json` |
| Workers | 6 | `ufold_train.py:176` |
| Seed | 0 | `utils.py:seed_torch()` |
| Model params | 8,641,089 (33 MB) | |

### 4.4 Postprocessing

Evaluation uses UFold's Augmented Lagrangian postprocessing (`ufold/postprocess.py:postprocess_new`) to enforce the constraint that each nucleotide pairs with at most one other. Parameters match `ufold_test.py`:

| Parameter | Value |
|---|---|
| `lr_min` | 0.01 |
| `lr_max` | 0.1 |
| `num_itr` | 100 |
| `rho` | 1.6 |
| `s` | 1.5 |
| `with_l1` | True |

### 4.5 Evaluation Metrics

Consistent with `DeepRNA/deepprotein/tasks/utils.py:secondary_structure_metircs`:

1. For each test sample, extract the valid (non-padded) region of the prediction and label.
2. Apply postprocessing to get probability values.
3. Flatten both matrices to 1D vectors.
4. Compute `binary_precision`, `binary_recall`, `binary_f1_score` (threshold=0.5), `binary_auroc`, and `binary_auprc` using `torcheval`.
5. Average across all samples (macro average).

Samples with no positive labels are skipped (AUROC/AUPRC are undefined).

---

## 5. Results

### 5.1 Test Metrics

| Metric | TestSetA (592 samples) | TestSetB (430 samples) |
|---|---|---|
| Precision | 0.7084 | 0.5428 |
| F1 | 0.6343 | 0.4145 |
| AUROC | 0.8127 | 0.6890 |
| AUPRC | 0.5167 | 0.2562 |

### 5.2 Training Details

- Total training time: 100.2 minutes (excluding ~14 min creatmat pre-computation)
- Loss at epoch 0: 0.661 (last-batch); epoch 99: 0.020 (last-batch)
- Note: reported loss is the last batch of each epoch (original `ufold_train.py` behavior), hence appears noisy

---

## 6. Reproduction

### Full Run (Train + Evaluate)

```bash
cd /path/to/UFold-repro
PYTHONUNBUFFERED=1 python ufold_train_rivals.py --gpu 0 --data_dir /path/to/rivals
```

### Evaluate Only (Load Checkpoint)

```python
from ufold_train_rivals import RivalsDataGenerator, RivalsDataset, model_eval_all_test
from Network import U_Net
import torch
from torch.utils import data

device = torch.device("cuda:0")
model = U_Net(img_ch=17)
model.load_state_dict(torch.load("models_rivals/ufold_train_rivals_99.pt", map_location=device))
model.to(device)

test_data = RivalsDataGenerator("/path/to/rivals/TestSetA-addss.pkl")
test_loader = data.DataLoader(RivalsDataset([test_data]), batch_size=1, num_workers=6)
results = model_eval_all_test(model, test_loader, device, "TestSetA")
```

### Expected Runtime

| Phase | Duration (H100) |
|---|---|
| creatmat pre-computation (TrainSetA) | ~14 min |
| Training (100 epochs) | ~100 min |
| Evaluation (TestSetA + TestSetB) | ~15 min |

---

## 7. File Inventory

| File | Description |
|---|---|
| `ufold_train_rivals.py` | Training + evaluation script |
| `Network.py` | U-Net architecture (unmodified from UFold) |
| `ufold/utils.py` | Utilities including `creatmat`, `seq_dict` (unmodified) |
| `ufold/postprocess.py` | Augmented Lagrangian postprocessing (unmodified) |
| `ufold/data_generator.py` | Original data pipeline (unmodified, used for `get_cut_len`) |
| `ufold/config.py`, `ufold/config.json` | Configuration (unmodified) |
| `models_rivals/*.pt` | Checkpoints (not in repo; regenerate via training) |
