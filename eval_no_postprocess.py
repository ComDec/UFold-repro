"""
One-off: evaluate checkpoint on bpRNAnew WITHOUT Augmented Lagrangian postprocess.
Compute metrics directly on sigmoid(logits) to measure the postprocess gain.

Reuses DataGenerator/EvalDataset from eval_from_checkpoint.py verbatim.
"""
import _pickle as pickle
import os
import argparse
import numpy as np
import torch
from torch.utils import data

from Network import U_Net as FCNNet
from ufold.utils import *
from ufold.utils import creatmat as creatmat_str
from ufold.data_generator import get_cut_len
from itertools import product as iterproduct

from torcheval.metrics.functional import (
    binary_auprc, binary_auroc, binary_f1_score,
    binary_precision, binary_recall,
)

perm = list(iterproduct(np.arange(4), np.arange(4)))


class DataGenerator(object):
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
        self.data_x = np.zeros((self.len, max_len, 4), dtype=np.float32)
        for i, s in enumerate(self.seq):
            for j, c in enumerate(s):
                self.data_x[i, j] = seq_dict.get(c.upper(), np.array([0, 0, 0, 0]))
        print(f'  Loaded {self.len} samples from {pkl_path}')
        print(f'  Pre-computing creatmat features...')
        self.creatmat_cache = [None] * self.len
        for i in range(self.len):
            self.creatmat_cache[i] = creatmat_str(self.seq[i])
            if (i + 1) % 1000 == 0:
                print(f'    creatmat: {i + 1}/{self.len}')
        print(f'  creatmat pre-computation done.')

    def get_one_sample(self, index):
        data_seq = self.data_x[index]
        data_len = self.seq_length[index]
        data_name = self.data_name[index]
        label = self.labels[index]
        contact = np.zeros((self.seq_max_len, self.seq_max_len), dtype=np.float32)
        contact[:data_len, :data_len] = label[:data_len, :data_len].astype(np.float32)
        matrix_rep = np.zeros(contact.shape)
        return contact, data_seq, matrix_rep, data_len, data_name


class EvalDataset(data.Dataset):
    def __init__(self, data_gen):
        self.data = data_gen
    def __len__(self):
        return self.data.len
    def __getitem__(self, index):
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        l = get_cut_len(data_len, 80)
        if l >= self.data.seq_max_len:
            contact_adj = np.zeros((l, l))
            contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        data_fcn = np.zeros((16, l, l))
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(
                data_seq[:data_len, i].reshape(-1, 1),
                data_seq[:data_len, j].reshape(1, -1))
        data_fcn_1 = np.zeros((1, l, l))
        data_fcn_1[0, :data_len, :data_len] = self.data.creatmat_cache[index]
        data_fcn_2 = np.concatenate((data_fcn, data_fcn_1), axis=0)
        return contact[:l, :l], data_fcn_2, matrix_rep, data_len, data_seq[:l], data_name


def evaluate_raw(contact_net, test_generator, device, dataset_name):
    """Evaluate with raw sigmoid output (NO postprocess).
    Also enforces output symmetry (since Network.py already applies transpose*d1,
    sigmoid of the raw output is already symmetric-ish; we keep as-is)."""
    contact_net.eval()
    all_prec, all_rec, all_f1, all_auroc, all_auprc = [], [], [], [], []
    n_skip = 0

    with torch.no_grad():
        for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name in test_generator:
            seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
            seq_len = seq_lens[0].item()
            pred_contacts = contact_net(seq_embedding_batch)

            # NO postprocess - just sigmoid on the raw output
            pred_prob = torch.sigmoid(pred_contacts[0, :seq_len, :seq_len]).cpu()

            true_label = contacts[0, :seq_len, :seq_len]
            p = pred_prob.flatten()
            t = true_label.flatten().int()
            if t.sum() == 0:
                n_skip += 1
                continue
            all_prec.append(binary_precision(p, t, threshold=0.5).item())
            all_rec.append(binary_recall(p, t, threshold=0.5).item())
            all_f1.append(binary_f1_score(p, t, threshold=0.5).item())
            all_auroc.append(binary_auroc(p, t).item())
            all_auprc.append(binary_auprc(p, t).item())

    n = len(all_f1)
    print(f'\n{"=" * 60}')
    print(f'  {dataset_name} Results — NO POSTPROCESS (n={n}, skipped={n_skip})')
    print(f'{"=" * 60}')
    print(f'  Precision : {np.mean(all_prec):.4f}')
    print(f'  Recall    : {np.mean(all_rec):.4f}')
    print(f'  F1        : {np.mean(all_f1):.4f}')
    print(f'  AUROC     : {np.mean(all_auroc):.4f}')
    print(f'  AUPRC     : {np.mean(all_auprc):.4f}')
    print(f'{"=" * 60}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda")
    seed_torch()

    print(f'Loading checkpoint: {args.checkpoint}')
    contact_net = FCNNet(img_ch=17)
    contact_net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    contact_net.to(device)
    print('Model loaded.')

    print(f'Loading test data: {args.test_file}')
    test_data = DataGenerator(args.test_file)
    test_set = EvalDataset(test_data)
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=6, drop_last=False)

    dataset_name = os.path.basename(args.test_file).replace('.pkl', '')
    evaluate_raw(contact_net, test_loader, device, dataset_name)


if __name__ == '__main__':
    main()
