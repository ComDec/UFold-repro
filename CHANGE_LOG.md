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
