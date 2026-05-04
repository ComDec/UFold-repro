# UFold — RNA Secondary Structure Prediction

Reproduction of [UFold (Liang et al., NAR 2022)](https://doi.org/10.1093/nar/gkab1074) on 5 standard benchmarks + cross-dataset evaluation on BIB and CompaRNA.

## Quick Start

```bash
# 1. Install
conda env create -f environment.yml
conda activate ufold-repro
pip install setproctitle "torcheval==0.0.6"

# 2. Download data (see §Data below)

# 3. Evaluate from saved predictions (no GPU, ~1 min)
python eval_from_predictions.py --predictions predictions/archiveII.pkl

# 4. Or: inference from checkpoints (~30 min on 3 GPUs)
bash scripts/eval_all.sh ./data ./checkpoints 0,1,2
```

## Environment

- Python 3.11, PyTorch 2.0.1, NVIDIA GPU with CUDA 11.8+ (tested: H100 NVL)
- ~15 GB disk for data, ~170 MB for checkpoints

```bash
conda env create -f environment.yml
conda activate ufold-repro
pip install setproctitle "torcheval==0.0.6"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.0.1 True
```

## Data

All datasets are on Google Drive:

**[Download Link](https://drive.google.com/drive/folders/15k0o3p1vuRDr5ZMMphIQ6LjbApgsuYc0?usp=drive_link)**

Download and place into `./data/`:

```
./data/
  all_data_1024_0.75/         # UniRNA-SS (train/valid/test.pkl)
  ipkont/                     # iPKnot (bpRNA-TR0.pkl, bpRNA-PK-TS0-1K.pkl)
  mxfold2/                    # bpRNA-1m (TR0/VL0/TS0-canonicals, RNAStrAlign600, archiveII, bpRNAnew)
  rivals/                     # Rivals (TrainSetA/TestSetA/TestSetB-addss.pkl)
  BIB/                        # BIB test sets (test-set-{1,2,3}.pkl)
  CompaRNA/                   # CompaRNA (pdb.pkl, rnastrand.pkl)
```

## Evaluation

### Goal 1: Evaluate from saved predictions (no GPU)

```bash
python eval_from_predictions.py --predictions predictions/archiveII.pkl
python eval_from_predictions.py --predictions predictions/unirna_ss_test.pkl
python eval_from_predictions.py --predictions predictions/ipknot.pkl
python eval_from_predictions.py --predictions predictions/bprna1m_new.pkl
python eval_from_predictions.py --predictions predictions/rivals_TestSetA.pkl
python eval_from_predictions.py --predictions predictions/rivals_TestSetB.pkl
```

For BIB/CompaRNA (requires ground-truth file):

```bash
python eval_from_predictions.py --predictions predictions/bib_unirna_ss_test-1.pkl \
    --bib --truth data/BIB/test-set-1.pkl
```

### Goal 2: Inference from checkpoints (requires GPU)

```bash
bash scripts/eval_all.sh ./data ./checkpoints 0,1,2
```

Or run individually:

```bash
python eval_from_checkpoint.py --gpu 0 --checkpoint checkpoints/archiveII.pt \
    --test_file data/mxfold2/archiveII.pkl --save_predictions predictions/archiveII.pkl
```

### Goal 3: Train from scratch

```bash
bash scripts/train_all.sh ./data ./models_retrain 0
```

Trains all 5 benchmarks sequentially (~30 h total on one H100). See [`RESULTS.md`](RESULTS.md) §5 for per-benchmark wall times.

All results are documented in [`RESULTS.md`](RESULTS.md).

## Code Changes vs Original

| File | Status | Purpose |
|---|---|---|
| `ufold_train_rivals.py` | **new** | Training + evaluation driver for `{id, seq, label}` pickle format |
| `eval_from_checkpoint.py` | **new** | Checkpoint inference + evaluation |
| `eval_from_predictions.py` | **new** | Evaluate from saved predictions (no GPU) |
| `eval_pk_from_predictions.py` | **new** | Pseudoknot metric computation |
| `scripts/eval_all.sh`, `scripts/train_all.sh` | **new** | One-command reproduction scripts |
| `Network.py`, `ufold/utils.py`, `ufold/postprocess.py`, `ufold/data_generator.py` | **unchanged** | Original UFold modules |

## Repository Layout

```
Network.py                      # U-Net architecture (original, unchanged)
ufold/                          # Original modules (unchanged): utils, postprocess, data_generator, config
ufold_train_rivals.py           # Training + evaluation driver
eval_from_checkpoint.py         # Inference from checkpoint (Goal 2)
eval_from_predictions.py        # Evaluate from saved predictions (Goal 1)
eval_pk_from_predictions.py     # Pseudoknot metrics
scripts/
  eval_all.sh                   # One-command evaluation
  train_all.sh                  # One-command training (Goal 3)
checkpoints/                    # 5 trained checkpoints + MD5SUMS
predictions/                    # Saved predictions for all benchmarks
environment.yml                 # Conda environment spec
RESULTS.md                      # All results with reproduction commands
```

## Citation

```bibtex
@article{liang2022ufold,
  title={UFold: fast and accurate RNA secondary structure prediction with deep learning},
  author={Liang, Kai and others},
  journal={Nucleic Acids Research},
  volume={50}, number={3}, pages={e14},
  year={2022}
}
```
