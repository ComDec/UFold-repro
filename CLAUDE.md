# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UFold is a deep learning method for RNA secondary structure prediction using U-Net Fully Convolutional Networks. It predicts base-pairing probability matrices from nucleotide sequences represented as image-like 2D feature maps. Sequences are encoded as 17-channel (L x L) tensors: 16 channels from pairwise outer products of one-hot encoded nucleotides + 1 thermodynamic feature channel.

## Environment Setup

```bash
conda env create -f UFold.yaml
conda activate UFold
# Note: update the `prefix:` path at the end of UFold.yaml to your local conda envs directory
```

Requires CUDA 11.8 and PyTorch 2.0.1. Python 3.11.

## Key Commands

### Training
```bash
python ufold_train.py
```
Uses merged datasets (RNAStralign, ArchiveII, TR0 with data augmentation). Saves checkpoints as `ufold_train_{epoch}.pt` one directory up. Batch size is 1 by default due to variable sequence lengths.

### Testing / Evaluation
```bash
python ufold_test.py --test_files TS2
```
Test set options: `ArchiveII`, `TS0`, `TS1`, `TS2`, `TS3`, `bpnew`, `RNAStralign`. Add `--nc True` for non-canonical base pair evaluation. Loads model from `models/ufold_train.pt`.

### Prediction on New Sequences
```bash
python ufold_predict.py --nc False
```
Reads FASTA input from `data/input.txt`. Outputs CT files to `results/save_ct_file/`.

### Data Preprocessing
```bash
python process_data_newdataset.py /path/to/bpseq/files/
```
Converts BPSEQ format files to pickle files used by the data generator. Filters to sequences <= 600 bp.

## Architecture

### Neural Network (`Network.py`)
- `U_Net`: 5-level encoder-decoder with skip connections. Input: (batch, 17, L, L) -> Output: (batch, L, L) contact map.
- `U_Net_FP`: Variant with Feature Pyramid Network for multi-scale features (defined but not used by default).
- Output symmetry enforced via `torch.transpose(d1, -1, -2) * d1`.

### Data Pipeline (`ufold/data_generator.py`)
- `RNASSDataGenerator`: Loads pickled `RNA_SS_data` namedtuples (seq, ss_label, length, name, pairs).
- `Dataset_Cut_concat_new`: Creates 17-channel feature tensors from sequences. The 16 pairwise channels come from `perm = list(product(np.arange(4), np.arange(4)))`. The 17th channel is a thermodynamic feature matrix (`creatmat()` in `ufold/utils.py`).
- `Dataset_Cut_concat_new_merge_multi`: Merges multiple datasets for training.
- Several dataset variants exist (`_canonicle`, `_merge_two`, etc.) used by different scripts.

### Post-processing (`ufold/postprocess.py`)
Augmented Lagrangian optimization ensures each nucleotide pairs with at most one other position. Key hyperparameters: `lr_min=0.01`, `lr_max=0.1`, `num_itr=100`, `rho=1.6`, `s=1.5`.

### Utilities (`ufold/utils.py`)
- `seq_dict`: Maps nucleotide characters (including IUPAC ambiguity codes) to one-hot vectors.
- `creatmat()`: Generates thermodynamic feature channel using base-pair scores (AU=2, GC=3, GU=0.8) with Gaussian distance weighting.
- `get_args()`: Argparse setup shared across scripts.
- Evaluation metrics: F1, precision, recall computed per-sequence.

### Configuration
- `ufold/config.json`: Runtime config (GPU ID, batch size, epochs, model type).
- `ufold/config.py`: Parses config.json into a Munch object.

## Data Formats

- **Input**: BPSEQ files (position, nucleotide, paired-position) or FASTA for prediction.
- **Intermediate**: Pickle files containing `RNA_SS_data` namedtuples.
- **Output**: CT (connectivity table) files and dot-bracket notation.

## Training Details

- Loss: `BCEWithLogitsLoss` with `pos_weight=300` (extreme class imbalance - most positions are unpaired).
- Optimizer: Adam with default learning rate.
- Contact maps are masked during loss computation to ignore zero-padded regions.
- The model selection between `U_Net` and `U_Net_FP` is done by changing the import in the training/test scripts (commented lines at top of files).
