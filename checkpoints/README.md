# Checkpoints

All checkpoints are UFold `U_Net(img_ch=17)` (8.6M params), trained for 100 epochs with default hyperparameters: `BCEWithLogitsLoss(pos_weight=300)`, Adam(lr=0.001), batch_size=1.

```python
from Network import U_Net as FCNNet
net = FCNNet(img_ch=17)
net.load_state_dict(torch.load('checkpoints/<name>.pt', map_location='cuda'))
```

| File | Training set | Test set | F1 |
|---|---|---|---|
| `rivals.pt` | TrainSetA-addss (3166) | TestSetA (592) / TestSetB (430) | 0.6343 / 0.4145 |
| `unirna_ss.pt` | UniRNA-SS train (8323) | UniRNA-SS test (1041) | 0.4394 |
| `bprna1m.pt` | TR0-canonicals (10814) | bpRNAnew (5401) | 0.4639 |
| `archiveII.pt` | RNAStrAlign600-train (20923) | archiveII (3956) | 0.6584 |
| `ipknot.pt` | bpRNA-TR0 (10814) | bpRNA-PK-TS0-1K (2909) | 0.4118 |

Total: 5 files, ~170 MB. Verify integrity: `cd checkpoints && md5sum -c MD5SUMS`
