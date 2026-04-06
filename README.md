# UFold-repro

Reproducibility repository for **UFold** ([paper](https://doi.org/10.1093/nar/gkab1074), [original repo](https://github.com/uci-cbcl/UFold)) on the Rivals benchmark dataset.

UFold is a deep learning method for RNA secondary structure prediction using U-Net fully convolutional networks. It predicts base-pairing contact maps from nucleotide sequences encoded as 17-channel image-like tensors.

## Setup

### Option A: Conda

```bash
# Create the environment (includes PyTorch with CUDA)
conda env create -f environment.yml
conda activate ufold-repro

# Verify everything works
python test_env.py
```

### Option B: Docker

```bash
# Build the image
docker build -t ufold .

# Run interactively with GPU support
docker run --gpus all -it \
    -v /path/to/data:/app/data \
    -v /path/to/models:/app/models \
    ufold bash

# Or run a specific command
docker run --gpus all \
    -v /path/to/data:/app/data \
    ufold python ufold_train_rivals.py --gpu 0 --data_dir /app/data
```

> **Note:** The legacy `UFold.yaml` file pins Python 3.11 / PyTorch 2.0.1 / CUDA 11.8.
> Use `environment.yml` instead for a more flexible setup.

## Quick Start

```bash
# Train on Rivals TrainSetA, evaluate on TestSetA/B
PYTHONUNBUFFERED=1 python ufold_train_rivals.py --gpu 0 --data_dir /path/to/rivals

# With custom save directory
python ufold_train_rivals.py --gpu 0 --data_dir /path/to/rivals --save_dir ./checkpoints
```

## Rivals Data Format

The training script expects pickle files in the data directory:

```
/path/to/rivals/
  TrainSetA-addss.pkl    # 3166 training samples
  TestSetA-addss.pkl     # 592 test samples
  TestSetB-addss.pkl     # 430 test samples
```

Each pickle file contains a list of dicts:

```python
{
    'id':     str,                    # sample identifier
    'seq':    str,                    # RNA sequence (A, U, C, G, N)
    'label':  np.ndarray (N, N),      # binary ground-truth contact map
    'matrix': np.ndarray (N, N),      # external predictions (not used)
}
```

## Feature Representation (17 Channels)

Identical to the original UFold:

| Channels | Description |
|---|---|
| 0-15 | Pairwise outer products of one-hot encoded nucleotides (4x4 = 16 combinations) |
| 16 | Thermodynamic feature from `creatmat()`: base-pair stacking scores (AU=2, GC=3, GU=0.8) with Gaussian distance weighting |

## Training Configuration

All hyperparameters follow UFold official defaults:

| Parameter | Value |
|---|---|
| Model | U_Net (8.6M params) |
| Input channels | 17 |
| Loss | BCEWithLogitsLoss (pos_weight=300) |
| Optimizer | Adam (lr=0.001) |
| Batch size | 1 |
| Epochs | 100 |
| Postprocessing | Augmented Lagrangian (lr_min=0.01, lr_max=0.1, num_itr=100, rho=1.6, s=1.5) |

## Evaluation Metrics

Consistent with [DeepRNA]`secondary_structure_metircs`:

```python
true_label = contacts[0, :seq_len, :seq_len]
# Flatten for binary metrics (matching secondary_structure_metircs)
p = pred_prob.flatten()
t = true_label.flatten().int()

# Skip samples with no positive labels (AUROC/AUPRC undefined)
if t.sum() == 0:
    n_skipped += 1
    continue

all_precision.append(binary_precision(p, t, threshold=0.5).item())
all_recall.append(binary_recall(p, t, threshold=0.5).item())
all_f1.append(binary_f1_score(p, t, threshold=0.5).item())
all_auroc.append(binary_auroc(p, t).item())
all_auprc.append(binary_auprc(p, t).item())
```

- Per-sample: flatten prediction and label matrices, compute binary metrics, then macro-average.
- Metrics: precision, recall, F1 (threshold=0.5), AUROC, AUPRC (via `torcheval`).
- Postprocessing applied before metric computation.

## Results

Trained on TrainSetA (3166 samples), 100 epochs, H100 NVL GPU:

| Metric | TestSetA (n=592) | TestSetB (n=430) |
|---|---|---|
| Precision | 0.7084 | 0.5428 |
| F1 | 0.6343 | 0.4145 |
| AUROC | 0.8127 | 0.6890 |
| AUPRC | 0.5167 | 0.2562 |

## Requirements

- Python >= 3.11
- PyTorch >= 2.0 with CUDA
- torcheval
- numpy, scipy, scikit-learn, munch

## Repository Structure

```
Network.py                  # U-Net architecture (from UFold)
ufold_train_rivals.py       # Training + evaluation script (this work)
ufold/
  config.json, config.py    # Configuration
  data_generator.py         # Original data pipeline (get_cut_len used)
  postprocess.py            # Augmented Lagrangian postprocessing
  utils.py                  # Utilities (creatmat, seq_dict, metrics)
docs/
  rivals_training_report.md # Detailed reproducibility report
```

## Acknowledgments

Based on [UFold](https://github.com/uci-cbcl/UFold) by Liang et al. (2022):

> Liang, K., et al. "UFold: fast and accurate RNA secondary structure prediction with deep learning." *Nucleic Acids Research*, 50(3), e14 (2022).
