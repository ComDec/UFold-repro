"""Evaluate from saved predictions — no GPU required.

Usage:
    python eval_from_predictions.py --predictions predictions/archiveII.pkl
    python eval_from_predictions.py --predictions predictions/bib_unirna_ss_test-1.pkl --bib
"""
import pickle
import argparse
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef

from torcheval.metrics.functional import (
    binary_auprc, binary_auroc, binary_f1_score,
    binary_precision, binary_recall,
)
import torch


def eval_standard(preds):
    all_prec, all_rec, all_f1, all_auroc, all_auprc = [], [], [], [], []
    n_skip = 0
    for p in preds:
        sl = p['seq_len']
        pred_t = torch.tensor(p['pred'][:sl, :sl].flatten(), dtype=torch.float32)
        label_t = torch.tensor(p['label'][:sl, :sl].flatten(), dtype=torch.long)
        if label_t.sum() == 0:
            n_skip += 1
            continue
        all_prec.append(binary_precision(pred_t, label_t, threshold=0.5).item())
        all_rec.append(binary_recall(pred_t, label_t, threshold=0.5).item())
        all_f1.append(binary_f1_score(pred_t, label_t, threshold=0.5).item())
        all_auroc.append(binary_auroc(pred_t, label_t).item())
        all_auprc.append(binary_auprc(pred_t, label_t).item())

    n = len(all_f1)
    print(f'\n{"=" * 60}')
    print(f'  Standard Metrics ({n} samples evaluated)')
    print(f'{"=" * 60}')
    print(f'  precision   : {np.mean(all_prec):.4f}')
    print(f'  recall      : {np.mean(all_rec):.4f}')
    print(f'  f1          : {np.mean(all_f1):.4f}')
    print(f'  auroc       : {np.mean(all_auroc):.4f}')
    print(f'  auprc       : {np.mean(all_auprc):.4f}')
    if n_skip:
        print(f'  (Skipped {n_skip} samples with no positive labels)')
    print(f'{"=" * 60}\n')


def eval_bib(preds, truth_path):
    truth = pickle.load(open(truth_path, 'rb'))
    assert len(preds) == len(truth), f"Length mismatch: {len(preds)} vs {len(truth)}"

    f1_list, mcc_list = [], []
    f1_skip, mcc_skip = 0, 0

    for p, t in zip(preds, truth):
        assert p['id'] == t['id'], f"ID mismatch: {p['id']} vs {t['id']}"
        label_mat = np.asarray(t['label'], dtype=np.float32)
        n = label_mat.shape[0]
        pred_mat = np.asarray(p['pred'][:n, :n], dtype=np.float32)

        tril_i, tril_j = np.tril_indices(n, k=-1)
        label_v = (label_mat[tril_i, tril_j] > 0).astype(int)
        pred_v = (pred_mat[tril_i, tril_j] > 0.5).astype(int)

        if label_v.sum() > 0:
            f1_list.append(float(f1_score(label_v, pred_v, zero_division=0)))
            mcc_list.append(float(matthews_corrcoef(label_v, pred_v)))
        else:
            f1_skip += 1
            mcc_skip += 1

    print(f'\n{"=" * 60}')
    print(f'  BIB/CompaRNA Metrics ({len(f1_list)} samples, {f1_skip} skipped)')
    print(f'{"=" * 60}')
    print(f'  F1  (lower-tri, sklearn) : {np.mean(f1_list):.4f}')
    print(f'  MCC (lower-tri, sklearn) : {np.mean(mcc_list):.4f}')
    print(f'{"=" * 60}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--bib', action='store_true',
                        help='BIB/CompaRNA mode: F1+MCC on lower triangle using ground-truth file')
    parser.add_argument('--truth', type=str, default=None,
                        help='Path to ground-truth pkl (required for --bib mode)')
    args = parser.parse_args()

    with open(args.predictions, 'rb') as f:
        preds = pickle.load(f)
    print(f'Loaded {len(preds)} predictions from {args.predictions}')

    if args.bib:
        if args.truth is None:
            raise ValueError('--truth is required for --bib mode')
        eval_bib(preds, args.truth)
    else:
        eval_standard(preds)


if __name__ == '__main__':
    main()
