# UFold Reproduction Checkpoints

All checkpoints are UFold `U_Net(img_ch=17)` with 8.6M parameters, trained with official default hyperparameters (see `../Benchmark.md` for details). Each is a standalone PyTorch state dict loadable via:

```python
from Network import U_Net as FCNNet
net = FCNNet(img_ch=17)
net.load_state_dict(torch.load('checkpoints/<name>.pt', map_location='cuda'))
```

## Files

| File | Training set | Epoch | Evaluate on | Expected F1 (torcheval, threshold=0.5) |
|---|---|---|---|---|
| `rivals_ep99.pt` | rivals/TrainSetA-addss.pkl (3166) | 99 | rivals/TestSetA-addss.pkl (592), TestSetB-addss.pkl (430) | 0.6343 (A), 0.4145 (B) |
| `unirna_ss_ep99.pt` | all_data_1024_0.75/train.pkl (8323) | 99 | all_data_1024_0.75/test.pkl (1041) | 0.4394 |
| `bprna1m_ep99.pt` | mxfold2/TR0-canonicals.pkl (10814) | 99 | mxfold2/TS0-canonicals.pkl (1304) | 0.4653 |
| `bprna1m_ep9.pt` | mxfold2/TR0-canonicals.pkl (10814) | 9 (early stop on VL0) | mxfold2/bpRNAnew.pkl (5401) | 0.5387 |
| `archiveII_ep99.pt` | mxfold2/RNAStrAlign600-train.pkl (20923) | 99 | mxfold2/archiveII.pkl (3961) | 0.6584 |
| `ipknot_ep99.pt` | ipkont/bpRNA-TR0.pkl (10814) | 99 | ipkont/bpRNA-PK-TS0-1K.pkl (2909) | 0.4118 |

**Total: 6 files, ~204 MB.**

## Why epoch 9 for bpRNA-1m-new?

`bprna1m_ep9.pt` is the lowest-validation-loss checkpoint (VL0 loss = 0.7117 at epoch 9, vs 39.68 at epoch 99 — severe overfitting). Used only for the bpRNAnew benchmark. VL0 ∩ bpRNAnew = ∅, so this is legitimate early-stopping, not test-set cherry-picking. See `../REPRODUCTION.md` Run 7 for details.

## Verification (2026-04-10)

All 6 checkpoints re-evaluated via `eval_from_checkpoint.py` on the documented test sets. Results match `REPRODUCTION.md` within CUDA floating-point rounding tolerance (max observed drift: 0.0003 on Rivals TestSetB precision; all F1 values match to 0.0001). Verification logs in `../logs/verify_*.log`.

## MD5 sums

See `MD5SUMS`.
