# Results

All results below are reproducible from the provided checkpoints and predictions.

## 1. In-Domain Benchmarks

Metrics: per-sample F1/Precision/Recall/AUROC/AUPRC via `torcheval` (threshold=0.5, macro-averaged).

| Benchmark | Train | Test | Checkpoint | Precision | Recall | **F1** | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|
| Rivals TestSetA | TrainSetA-addss (3166) | TestSetA-addss (592) | `rivals.pt` | 0.7084 | 0.6081 | **0.6343** | 0.8127 | 0.5167 |
| Rivals TestSetB | TrainSetA-addss (3166) | TestSetB-addss (430) | `rivals.pt` | 0.5428 | 0.3595 | **0.4145** | 0.6890 | 0.2562 |
| UniRNA-SS | train (8323) | test (1041) | `unirna_ss.pt` | 0.4514 | 0.6383 | **0.4394** | 0.7422 | 0.3420 |
| bpRNA-1m-new | TR0-canonicals (10814) | bpRNAnew (5401) | `bprna1m.pt` | 0.5386 | 0.5227 | **0.4639** | 0.7315 | 0.3417 |
| ArchiveII | RNAStrAlign600-train (20923) | archiveII (3966) | `archiveII.pt` | 0.6831 | 0.6533 | **0.6584** | 0.8333 | 0.5755 |
| iPKnot | bpRNA-TR0 (10814) | bpRNA-PK-TS0-1K (2909) | `ipknot.pt` | 0.4093 | 0.6118 | **0.4118** | 0.7349 | 0.3275 |

### Evaluate from predictions (no GPU)

```bash
python eval_from_predictions.py --predictions predictions/unirna_ss_test.pkl
python eval_from_predictions.py --predictions predictions/archiveII.pkl
python eval_from_predictions.py --predictions predictions/ipknot.pkl
python eval_from_predictions.py --predictions predictions/bprna1m_new.pkl
python eval_from_predictions.py --predictions predictions/rivals_TestSetA.pkl
python eval_from_predictions.py --predictions predictions/rivals_TestSetB.pkl
```

### Inference from checkpoints (requires GPU)

```bash
python eval_from_checkpoint.py --gpu 0 --checkpoint checkpoints/rivals.pt \
    --test_file data/rivals/TestSetA-addss.pkl --save_predictions predictions/rivals_TestSetA.pkl
python eval_from_checkpoint.py --gpu 0 --checkpoint checkpoints/rivals.pt \
    --test_file data/rivals/TestSetB-addss.pkl --save_predictions predictions/rivals_TestSetB.pkl
python eval_from_checkpoint.py --gpu 0 --checkpoint checkpoints/unirna_ss.pt \
    --test_file data/all_data_1024_0.75/test.pkl --save_predictions predictions/unirna_ss_test.pkl
python eval_from_checkpoint.py --gpu 0 --checkpoint checkpoints/bprna1m.pt \
    --test_file data/mxfold2/bpRNAnew.pkl --save_predictions predictions/bprna1m_new.pkl
python eval_from_checkpoint.py --gpu 0 --checkpoint checkpoints/archiveII.pt \
    --test_file data/mxfold2/archiveII.pkl --save_predictions predictions/archiveII.pkl
python eval_from_checkpoint.py --gpu 0 --checkpoint checkpoints/ipknot.pt \
    --test_file data/ipkont/bpRNA-PK-TS0-1K.pkl --save_predictions predictions/ipknot.pkl
```

---

## 2. Pseudoknot-Aware Metrics (UniRNA-SS, ArchiveII, iPKnot)

Metrics from DeepRNA `deeprna.metrics.pseudoknot`: `score` (overall F1), `score_pk` (PK-containing samples only), `pk_sen/pk_ppv/pk_f1` (crossing base-pair F1).

| Benchmark | n_total | n_pk | F1 (score) | F1_PK (score_pk) | pk_sen | pk_ppv | **pk_f1 (F1_CP)** |
|---|---|---|---|---|---|---|---|
| UniRNA-SS | 1041 | 164 (15.8%) | 0.4387 | 0.1111 | 0.0229 | 0.0178 | **0.0197** |
| ArchiveII | 3966 | 1079 (27.2%) | 0.6576 | 0.2167 | 0.0045 | 0.0011 | **0.0013** |
| iPKnot | 2914 | 353 (12.1%) | 0.4105 | 0.1869 | 0.0667 | 0.0654 | **0.0639** |

### Evaluate from predictions

```bash
python eval_pk_from_predictions.py --predictions predictions/unirna_ss_test.pkl --dataset_name UniRNA-SS
python eval_pk_from_predictions.py --predictions predictions/archiveII.pkl --dataset_name ArchiveII
python eval_pk_from_predictions.py --predictions predictions/ipknot.pkl --dataset_name iPKnot
```

---

## 3. BIB Cross-Dataset Evaluation

Zero-shot inference on BIB test sets. Metrics: F1 and MCC (INF) on lower triangle, per-sample macro-average, sklearn.

### Panel A — UniRNA-SS model (`unirna_ss.pt`)

| Test Set | n | F1 | MCC (INF) |
|---|---|---|---|
| BIB-1 | 496 | **0.5392** | 0.5546 |
| BIB-2 | 989 | **0.6239** | 0.6449 |
| BIB-3 | 16 | **0.3818** | 0.4015 |

### Panel B — bpRNA-1m model (`bprna1m.pt`)

| Test Set | n | F1 | MCC (INF) |
|---|---|---|---|
| BIB-1 | 496 | **0.4938** | 0.5166 |
| BIB-2 | 989 | **0.5279** | 0.5561 |
| BIB-3 | 16 | **0.3634** | 0.3855 |

### Evaluate from predictions

```bash
# UniRNA-SS model
python eval_from_predictions.py --predictions predictions/bib_unirna_ss_test-1.pkl \
    --bib --truth data/BIB/test-set-1.pkl
python eval_from_predictions.py --predictions predictions/bib_unirna_ss_test-2.pkl \
    --bib --truth data/BIB/test-set-2.pkl
python eval_from_predictions.py --predictions predictions/bib_unirna_ss_test-3.pkl \
    --bib --truth data/BIB/test-set-3.pkl

# bpRNA-1m model
python eval_from_predictions.py --predictions predictions/bib_bprna1m_test-1.pkl \
    --bib --truth data/BIB/test-set-1.pkl
python eval_from_predictions.py --predictions predictions/bib_bprna1m_test-2.pkl \
    --bib --truth data/BIB/test-set-2.pkl
python eval_from_predictions.py --predictions predictions/bib_bprna1m_test-3.pkl \
    --bib --truth data/BIB/test-set-3.pkl
```

### Inference from checkpoints

```bash
python eval_from_checkpoint.py --gpu 0 --checkpoint checkpoints/unirna_ss.pt \
    --test_file data/BIB/test-set-1.pkl --save_predictions predictions/bib_unirna_ss_test-1.pkl
python eval_from_checkpoint.py --gpu 0 --checkpoint checkpoints/unirna_ss.pt \
    --test_file data/BIB/test-set-2.pkl --save_predictions predictions/bib_unirna_ss_test-2.pkl
python eval_from_checkpoint.py --gpu 0 --checkpoint checkpoints/unirna_ss.pt \
    --test_file data/BIB/test-set-3.pkl --save_predictions predictions/bib_unirna_ss_test-3.pkl

python eval_from_checkpoint.py --gpu 0 --checkpoint checkpoints/bprna1m.pt \
    --test_file data/BIB/test-set-1.pkl --save_predictions predictions/bib_bprna1m_test-1.pkl
python eval_from_checkpoint.py --gpu 0 --checkpoint checkpoints/bprna1m.pt \
    --test_file data/BIB/test-set-2.pkl --save_predictions predictions/bib_bprna1m_test-2.pkl
python eval_from_checkpoint.py --gpu 0 --checkpoint checkpoints/bprna1m.pt \
    --test_file data/BIB/test-set-3.pkl --save_predictions predictions/bib_bprna1m_test-3.pkl
```

---

## 4. CompaRNA Cross-Dataset Evaluation

Zero-shot inference on CompaRNA. Metrics: F1 via `torcheval` (same as §1).

### UniRNA-SS model (`unirna_ss.pt`)

| Test Set | n | **F1** |
|---|---|---|
| CompaRNA-pdb | 201 | **0.6582** |
| CompaRNA-rnastrand | 1805 | **0.2567** |

### bpRNA-1m model (`bprna1m.pt`)

| Test Set | n | **F1** |
|---|---|---|
| CompaRNA-pdb | 201 | **0.6427** |
| CompaRNA-rnastrand | 1805 | **0.2250** |

### Evaluate from predictions

```bash
python eval_from_predictions.py --predictions predictions/comparna_unirna_ss_pdb.pkl
python eval_from_predictions.py --predictions predictions/comparna_unirna_ss_rnastrand.pkl
python eval_from_predictions.py --predictions predictions/comparna_bprna1m_pdb.pkl
python eval_from_predictions.py --predictions predictions/comparna_bprna1m_rnastrand.pkl
```

### Inference from checkpoints

```bash
python eval_from_checkpoint.py --gpu 0 --checkpoint checkpoints/unirna_ss.pt \
    --test_file data/CompaRNA/pdb.pkl --save_predictions predictions/comparna_unirna_ss_pdb.pkl
python eval_from_checkpoint.py --gpu 0 --checkpoint checkpoints/unirna_ss.pt \
    --test_file data/CompaRNA/rnastrand.pkl --save_predictions predictions/comparna_unirna_ss_rnastrand.pkl
python eval_from_checkpoint.py --gpu 0 --checkpoint checkpoints/bprna1m.pt \
    --test_file data/CompaRNA/pdb.pkl --save_predictions predictions/comparna_bprna1m_pdb.pkl
python eval_from_checkpoint.py --gpu 0 --checkpoint checkpoints/bprna1m.pt \
    --test_file data/CompaRNA/rnastrand.pkl --save_predictions predictions/comparna_bprna1m_rnastrand.pkl
```

---

## 5. Training from Scratch

```bash
bash scripts/train_all.sh ./data ./models_retrain 0
```

Per-benchmark wall time on one H100 NVL:

| Benchmark | Training set | Epochs | Wall time |
|---|---|---|---|
| Rivals | TrainSetA-addss (3166) | 100 | ~100 min |
| UniRNA-SS | train (8323) | 100 | ~5 h |
| bpRNA-1m | TR0-canonicals (10814) | 100 | ~6 h |
| iPKnot | bpRNA-TR0 (10814) | 100 | ~5.5 h |
| ArchiveII | RNAStrAlign600-train (20923) | 100 | ~12.5 h |

All use UFold defaults: `BCEWithLogitsLoss(pos_weight=300)`, Adam(lr=0.001), batch_size=1, seed=0.

After training, evaluate against your checkpoints:
```bash
bash scripts/eval_all.sh ./data ./models_retrain 0,1,2
```

---

## Notes

- CUDA floating-point operations are not bit-deterministic. Max observed drift: 0.0003 on any single metric.
- All checkpoints are epoch 99 (100 epochs total).
- BIB/CompaRNA evaluation uses `sklearn.metrics.f1_score` on the lower triangle (each base pair counted once). In-domain evaluation uses `torcheval` on the full flattened matrix — numerically equivalent for symmetric contact maps.
- Pseudoknot metrics require the DeepRNA package (`deeprna.metrics.pseudoknot`).

### Postprocessing NaN Issue (Algorithm Deficiency)

UFold's Augmented Lagrangian postprocessing suffers from numerical divergence on longer sequences, producing NaN predictions. These samples are **not filtered** — they remain in the evaluation and effectively receive F1 ≈ 0 (torcheval treats NaN as positive predictions, yielding near-zero precision). This is an inherent limitation of UFold's postprocessing, not a data processing bug.

| Benchmark | Total Samples | NaN Samples | NaN % | Onset Length |
|---|---|---|---|---|
| Rivals TestSetA | 592 | 0 | 0% | — |
| Rivals TestSetB | 430 | 0 | 0% | — |
| UniRNA-SS | 1041 | 180 | 17.3% | ~250 nt |
| bpRNA-1m-new | 5401 | 435 | 8.1% | ~270 nt |
| ArchiveII | 3966 | 21 | 0.5% | ~500 nt |
| iPKnot | 2914 | 473 | 16.2% | ~250 nt |

Impact on reported F1 (NaN samples contribute F1 ≈ 0 to the macro-average):

| Benchmark | F1 (all samples, reported) | F1 (excluding NaN) | Δ |
|---|---|---|---|
| UniRNA-SS | 0.4394 | 0.5304 | +0.091 |
| bpRNA-1m-new | 0.4639 | 0.5042 | +0.040 |
| ArchiveII | 0.6584 | 0.6619 | +0.004 |
| iPKnot | 0.4118 | 0.4911 | +0.079 |

The NaN onset length varies by checkpoint because it depends on the training data distribution — checkpoints trained on shorter sequences (UniRNA-SS, bpRNA-1m, iPKnot with max training length ~500 nt) diverge earlier than ArchiveII (trained on sequences up to 1800 nt). All reported results include these NaN samples without exclusion.
