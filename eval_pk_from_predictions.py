"""
Compute pseudoknot-aware metrics on saved UFold predictions.

Uses DeepRNA's pseudoknot evaluation function unmodified:
  /home/xiwang/project/develop/deeprna/deeprna/metrics/pseudoknot.py

Also recomputes the standard torcheval metrics (precision/recall/F1/AUROC/AUPRC)
as a sanity check against REPRODUCTION.md's Run 4 (iPKnot) and Run 5 (ArchiveII).

Saved predictions are the Augmented-Lagrangian postprocessed outputs from
ufold_train_rivals.py:model_eval_all_test() / eval_from_checkpoint.py,
stored as per-sample dicts with keys {id, seq_len, pred, label}.

No GPU required. No model inference. Reads existing pkl, computes metrics.
"""
import sys
import time
import argparse
import _pickle as pickle
import numpy as np
import torch

# Import the user's pseudoknot metric module (unmodified).
sys.path.insert(0, '/home/xiwang/project/develop/deeprna')
from deeprna.metrics.pseudoknot import evaluate_structure_metrics

from torcheval.metrics.functional import (
    binary_auprc, binary_auroc, binary_f1_score,
    binary_precision, binary_recall,
)


def _seq_len(item):
    v = item['seq_len']
    return v.item() if hasattr(v, 'item') else int(v)


def compute_torcheval_metrics(preds):
    """Mirror of ufold_train_rivals.py:model_eval_all_test() metric logic,
    to independently verify the reported numbers in REPRODUCTION.md."""
    all_prec, all_rec, all_f1, all_auroc, all_auprc = [], [], [], [], []
    n_skip = 0
    for item in preds:
        seq_len = _seq_len(item)
        pred = torch.from_numpy(np.asarray(item['pred']))[:seq_len, :seq_len]
        label = torch.from_numpy(np.asarray(item['label']))[:seq_len, :seq_len]
        p = pred.flatten().float()
        t = label.flatten().int()
        if t.sum() == 0:
            n_skip += 1
            continue
        all_prec.append(binary_precision(p, t, threshold=0.5).item())
        all_rec.append(binary_recall(p, t, threshold=0.5).item())
        all_f1.append(binary_f1_score(p, t, threshold=0.5).item())
        all_auroc.append(binary_auroc(p, t).item())
        all_auprc.append(binary_auprc(p, t).item())
    return {
        'precision': float(np.mean(all_prec)),
        'recall': float(np.mean(all_rec)),
        'f1': float(np.mean(all_f1)),
        'auroc': float(np.mean(all_auroc)),
        'auprc': float(np.mean(all_auprc)),
        'n_eval': len(all_f1),
        'n_skip': n_skip,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    print(f'=== {args.dataset_name} ===', flush=True)
    print(f'Loading predictions from {args.predictions}', flush=True)
    t0 = time.time()
    with open(args.predictions, 'rb') as f:
        preds = pickle.load(f)
    print(f'  Loaded {len(preds)} samples in {time.time()-t0:.1f}s', flush=True)

    # Sanity summary
    lens = [_seq_len(x) for x in preds]
    print(f'  seq_len: min={min(lens)}, max={max(lens)}, mean={np.mean(lens):.1f}', flush=True)

    # --- Standard torcheval metrics (sanity check vs REPRODUCTION.md) ---
    print(f'\n[{args.dataset_name}] Standard torcheval metrics (threshold={args.threshold}):', flush=True)
    t0 = time.time()
    std = compute_torcheval_metrics(preds)
    print(f'  n_eval={std["n_eval"]}, n_skip={std["n_skip"]}  ({time.time()-t0:.1f}s)', flush=True)
    for k in ('precision', 'recall', 'f1', 'auroc', 'auprc'):
        print(f'  {k:12s}: {std[k]:.4f}', flush=True)

    # --- Pseudoknot-aware metrics (deeprna.metrics.pseudoknot) ---
    # evaluate_structure_metrics expects dicts with 'pred_prob' and 'label'.
    # Slice to seq_len to discard padding before conversion.
    print(f'\n[{args.dataset_name}] Building PK evaluation input (slicing pred/label to seq_len)...', flush=True)
    t0 = time.time()
    pk_input = []
    for item in preds:
        seq_len = _seq_len(item)
        pred = np.asarray(item['pred'])[:seq_len, :seq_len]
        label = np.asarray(item['label'])[:seq_len, :seq_len]
        pk_input.append({'pred_prob': pred, 'label': label})
    print(f'  Built {len(pk_input)} entries in {time.time()-t0:.1f}s', flush=True)

    print(f'\n[{args.dataset_name}] Pseudoknot-aware metrics (deeprna.metrics.pseudoknot, threshold={args.threshold}):', flush=True)
    t0 = time.time()
    pk = evaluate_structure_metrics(pk_input, threshold=args.threshold)
    print(f'  Computed in {time.time()-t0:.1f}s', flush=True)
    print(f'  n_total      : {pk["n_total"]}', flush=True)
    print(f'  n_pk         : {pk["n_pk"]}', flush=True)
    print(f'  score (F1)   : {pk["score"]:.4f}     # overall F1 (sklearn f1_score, flatten)', flush=True)
    print(f'  score_pk     : {pk["score_pk"]:.4f}     # overall F1 on PK-containing samples only', flush=True)
    print(f'  pk_sen       : {pk["pk_sen"]:.4f}     # PK base-pair sensitivity', flush=True)
    print(f'  pk_ppv       : {pk["pk_ppv"]:.4f}     # PK base-pair PPV', flush=True)
    print(f'  pk_f1        : {pk["pk_f1"]:.4f}     # PK base-pair F1', flush=True)

    print(f'\nDone: {args.dataset_name}', flush=True)


if __name__ == '__main__':
    main()
