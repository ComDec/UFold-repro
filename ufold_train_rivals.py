"""
UFold training on Rivals benchmark dataset.
Minimal modification of the original ufold_train.py -- only data loading is adapted.
Changes vs original are marked with # [RIVALS].

Usage:
    python ufold_train_rivals.py --gpu 6
    python ufold_train_rivals.py --gpu 0 --data_dir /path/to/rivals --save_dir ./checkpoints
"""
import _pickle as pickle
import sys
import os
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils import data

import time
import collections

from Network import U_Net as FCNNet

from ufold.utils import *
from ufold.utils import creatmat as creatmat_str  # string-based creatmat from utils.py
from ufold.config import process_config
from ufold.postprocess import postprocess_new as postprocess

# [RIVALS] Replaced: from ufold.data_generator import RNASSDataGenerator, Dataset, ...
# Instead, use custom data adapter below for rivals pickle format.
from ufold.data_generator import get_cut_len
from itertools import product as iterproduct

from torcheval.metrics.functional import (  # [RIVALS] for evaluation
    binary_auprc,
    binary_auroc,
    binary_f1_score,
    binary_precision,
    binary_recall,
)

RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')

# ======================================================================
# [RIVALS] Data adapter: bridges rivals pkl format to UFold pipeline
# ======================================================================
perm = list(iterproduct(np.arange(4), np.arange(4)))  # 16 pairwise combos


class RivalsDataGenerator(object):
    """
    Drop-in replacement for RNASSDataGenerator that reads rivals pickle format.

    The rivals pickle files contain list of dicts:
        {'id': str, 'seq': str, 'label': ndarray(N,N), 'matrix': ndarray(N,N)}
    We use 'seq' and 'label' only. 'matrix' (external method predictions) is ignored.
    """

    def __init__(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            raw = pickle.load(f)
        self.len = len(raw)
        self.seq = [d['seq'] for d in raw]
        self.seq_length = np.array([len(d['seq']) for d in raw])
        self.data_name = np.array([d['id'] for d in raw])
        self.labels = [d['label'] for d in raw]

        max_len = int(max(self.seq_length))
        self.seq_max_len = max_len

        # One-hot encode using UFold's seq_dict (from ufold/utils.py)
        # N nucleotide maps to [0,0,0,0] -- zero features, consistent with UFold convention
        self.data_x = np.zeros((self.len, max_len, 4), dtype=np.float32)
        for i, s in enumerate(self.seq):
            for j, c in enumerate(s):
                self.data_x[i, j] = seq_dict.get(c.upper(), np.array([0, 0, 0, 0]))

        # Validate label symmetry
        for i in range(self.len):
            label = self.labels[i]
            assert np.allclose(label, label.T), \
                f"Label for sample {self.data_name[i]} is not symmetric"

        # Compatibility fields (unused for single-dataset training)
        self.data_y = np.zeros((self.len, max_len, 3), dtype=np.float32)
        self.pairs = [[] for _ in range(self.len)]
        print(f'  Loaded {self.len} samples from {pkl_path}')

        # Pre-compute creatmat for all samples to avoid per-epoch recomputation.
        #
        # Note on creatmat versions:
        #   - ufold/utils.py:creatmat -- takes string input, pure-Python O(L^2*30)
        #   - ufold/data_generator.py:creatmat (line 880) -- takes one-hot input,
        #     GPU-accelerated, but crashes on N nucleotides ([0,0,0,0].index(1) -> ValueError)
        #
        # We use the utils.py version with string input because:
        #   (a) it correctly handles N nucleotides (paired('N', x) returns 0)
        #   (b) pre-computation amortizes the cost (computed once, not per-epoch)
        print(f'  Pre-computing creatmat features...')
        self.creatmat_cache = [None] * self.len
        for i in range(self.len):
            self.creatmat_cache[i] = creatmat_str(self.seq[i])
            if (i + 1) % 500 == 0:
                print(f'    creatmat: {i + 1}/{self.len}')
        print(f'  creatmat pre-computation done.')

    def get_one_sample(self, index):
        data_seq = self.data_x[index]
        data_len = self.seq_length[index]
        data_name = self.data_name[index]
        label = self.labels[index]
        # Build contact map from label (already NxN binary symmetric matrix)
        contact = np.zeros((self.seq_max_len, self.seq_max_len), dtype=np.float32)
        contact[:data_len, :data_len] = label[:data_len, :data_len].astype(np.float32)
        matrix_rep = np.zeros(contact.shape)
        return contact, data_seq, matrix_rep, data_len, data_name


class RivalsDataset(data.Dataset):
    """
    Equivalent to Dataset_Cut_concat_new_merge_multi from ufold/data_generator.py,
    adapted for RivalsDataGenerator. Produces 17-channel input tensors:
      - Channels 0-15: pairwise outer products of one-hot encoded nucleotides
      - Channel 16: creatmat thermodynamic feature
    """

    def __init__(self, data_list):
        # Same single-dataset path as Dataset_Cut_concat_new_merge_multi
        self.data = data_list[0]

    def __len__(self):
        return self.data.len

    def __getitem__(self, index):
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        l = get_cut_len(data_len, 80)

        # Pad if sequence length exceeds pre-allocated array size
        # Original uses `if l >= 500:` (hardcoded for 600-padded data);
        # we check against actual seq_max_len for generality.
        if l >= self.data.seq_max_len:
            contact_adj = np.zeros((l, l))
            contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj

        # 16 pairwise outer product channels (identical to original data_generator.py:651-653)
        data_fcn = np.zeros((16, l, l))
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(
                data_seq[:data_len, i].reshape(-1, 1),
                data_seq[:data_len, j].reshape(1, -1))

        # Channel 17: creatmat thermodynamic feature (pre-computed in RivalsDataGenerator)
        data_fcn_1 = np.zeros((1, l, l))
        data_fcn_1[0, :data_len, :data_len] = self.data.creatmat_cache[index]

        # Concatenate to 17 channels (identical to original data_generator.py:664)
        data_fcn_2 = np.concatenate((data_fcn, data_fcn_1), axis=0)  # (17, l, l)
        return contact[:l, :l], data_fcn_2, matrix_rep, data_len, data_seq[:l], data_name


# ======================================================================
# Training -- identical to ufold_train.py:31-105
# ======================================================================
def train(contact_net, train_merge_generator, epoches_first, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weight = torch.Tensor([300]).to(device)
    criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight)
    u_optimizer = optim.Adam(contact_net.parameters())

    steps_done = 0
    print('start training...')
    for epoch in range(epoches_first):
        contact_net.train()
        for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name in train_merge_generator:
            contacts_batch = torch.Tensor(contacts.float()).to(device)
            seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)

            pred_contacts = contact_net(seq_embedding_batch)

            contact_masks = torch.zeros_like(pred_contacts)
            contact_masks[:, :seq_lens, :seq_lens] = 1

            # Compute loss (identical to ufold_train.py:85)
            loss_u = criterion_bce_weighted(pred_contacts * contact_masks, contacts_batch)

            # Optimize the model
            u_optimizer.zero_grad()
            loss_u.backward()
            u_optimizer.step()
            steps_done = steps_done + 1

        print('Training log: epoch: {}, step: {}, loss: {}'.format(
            epoch, steps_done - 1, loss_u))
        # [RIVALS] Save every 10 epochs + final epoch (original saves every epoch)
        if epoch > -1:
            if (epoch + 1) % 10 == 0 or epoch == epoches_first - 1:
                torch.save(contact_net.state_dict(),
                           os.path.join(save_dir, f'ufold_train_rivals_{epoch}.pt'))


# ======================================================================
# [RIVALS] Evaluation with metrics matching DeepRNA secondary_structure_metircs
# ======================================================================
def model_eval_all_test(contact_net, test_generator, device, dataset_name):
    """
    Evaluate model on test set. Metrics are consistent with
    DeepRNA/deepprotein/tasks/utils.py:secondary_structure_metircs:
      - Per-sample: flatten pred & label, compute binary metrics, then average.
      - Postprocessing is applied before metric computation (same as ufold_test.py).
    """
    contact_net.eval()
    all_precision = []
    all_recall = []
    all_f1 = []
    all_auroc = []
    all_auprc = []
    n_postprocess_fallback = 0
    n_skipped = 0

    with torch.no_grad():
        for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name in test_generator:
            seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
            seq_len = seq_lens[0].item()

            pred_contacts = contact_net(seq_embedding_batch)

            # Postprocess with Augmented Lagrangian (identical params to ufold_test.py)
            pred_u = pred_contacts[:, :seq_len, :seq_len]
            seq_tmp = seq_ori[:, :seq_len, :].float().to(device)
            try:
                u_no_train = postprocess(pred_u, seq_tmp,
                                         lr_min=0.01, lr_max=0.1,
                                         num_itr=100, rho=1.6,
                                         with_l1=True, s=1.5)
                pred_prob = u_no_train[0].cpu()
            except RuntimeError:
                # Fallback for edge cases (e.g., very short sequences)
                pred_prob = torch.sigmoid(pred_contacts[0, :seq_len, :seq_len].cpu())
                n_postprocess_fallback += 1

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

    results = {
        'precision': np.mean(all_precision) if all_precision else 0.0,
        'recall': np.mean(all_recall) if all_recall else 0.0,
        'f1': np.mean(all_f1) if all_f1 else 0.0,
        'auroc': np.mean(all_auroc) if all_auroc else 0.0,
        'auprc': np.mean(all_auprc) if all_auprc else 0.0,
    }
    n = len(all_f1)
    print(f'\n{"=" * 60}')
    print(f'  {dataset_name} Results ({n} samples evaluated)')
    print(f'{"=" * 60}')
    print(f'  Precision : {results["precision"]:.4f}')
    print(f'  Recall    : {results["recall"]:.4f}')
    print(f'  F1        : {results["f1"]:.4f}')
    print(f'  AUROC     : {results["auroc"]:.4f}')
    print(f'  AUPRC     : {results["auprc"]:.4f}')
    if n_skipped > 0:
        print(f'  (Skipped {n_skipped} samples with no positive labels)')
    if n_postprocess_fallback > 0:
        print(f'  (Postprocess fallback to sigmoid: {n_postprocess_fallback} samples)')
    print(f'{"=" * 60}\n')

    return results


# ======================================================================
# Main
# ======================================================================
def parse_rivals_args():
    parser = argparse.ArgumentParser(description='UFold training on Rivals dataset')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (default: 0)')
    parser.add_argument('--data_dir', type=str,
                        default='/home/xiwang/project/develop/data/rivals',
                        help='Path to rivals data directory')
    parser.add_argument('--save_dir', type=str,
                        default='./models_rivals',
                        help='Directory to save checkpoints')
    parser.add_argument('-c', '--config', type=str,
                        default='ufold/config.json',
                        help='UFold config file')
    return parser.parse_args()


def main():
    rivals_args = parse_rivals_args()
    GPU_ID = rivals_args.gpu
    DATA_DIR = rivals_args.data_dir
    SAVE_DIR = rivals_args.save_dir
    os.makedirs(SAVE_DIR, exist_ok=True)

    torch.cuda.set_device(GPU_ID)

    config = process_config(rivals_args.config)
    print('#####Stage 1#####')
    print('Here is the configuration of this run: ')
    print(config)

    BATCH_SIZE = config.batch_size_stage_1
    epoches_first = config.epoches_first

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_torch()

    # [RIVALS] Load rivals data instead of UFold-format pickle files
    print('Loading rivals dataset...')
    train_data = RivalsDataGenerator(os.path.join(DATA_DIR, 'TrainSetA-addss.pkl'))

    # DataLoader params (identical to ufold_train.py:173-177)
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 6,
              'drop_last': True}

    train_merge = RivalsDataset([train_data])
    train_merge_generator = data.DataLoader(train_merge, **params)

    # img_ch=17: 16 pairwise + 1 creatmat (identical to original UFold)
    contact_net = FCNNet(img_ch=17)
    contact_net.to(device)

    t0 = time.time()
    train(contact_net, train_merge_generator, epoches_first, SAVE_DIR)
    train_time = time.time() - t0
    print(f'\nTraining completed in {train_time / 60:.1f} minutes')

    # [RIVALS] Evaluate on TestSetA and TestSetB
    test_params = {'batch_size': 1, 'shuffle': False,
                   'num_workers': 6, 'drop_last': False}

    all_results = {}
    for test_name, test_file in [('TestSetA', 'TestSetA-addss.pkl'),
                                  ('TestSetB', 'TestSetB-addss.pkl')]:
        test_data = RivalsDataGenerator(os.path.join(DATA_DIR, test_file))
        test_set = RivalsDataset([test_data])
        test_generator = data.DataLoader(test_set, **test_params)
        results = model_eval_all_test(contact_net, test_generator, device, test_name)
        all_results[test_name] = results

    # Final summary
    print('\n' + '=' * 60)
    print('FINAL SUMMARY')
    print('=' * 60)
    print(f'Model: U_Net (img_ch=17, 16 pairwise + 1 creatmat)')
    print(f'Training: {train_data.len} samples, {epoches_first} epochs, {train_time / 60:.1f} min')
    print(f'Loss: BCEWithLogitsLoss(pos_weight=300), Optimizer: Adam')
    print(f'Postprocessing: Augmented Lagrangian (official UFold params)')
    print()
    for test_name, r in all_results.items():
        print(f'{test_name}: precision={r["precision"]:.4f}, recall={r["recall"]:.4f}, '
              f'f1={r["f1"]:.4f}, auroc={r["auroc"]:.4f}, auprc={r["auprc"]:.4f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
