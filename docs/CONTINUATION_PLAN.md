# UFold 整理发布计划 (2026-04-10)

> 因 GPU 2 进入 `GPU requires reset` 状态导致全局 cuInit 失败，需重启机器后继续。

## 当前状态

- GPU 2 (PCI 3B:00.0) 硬件错误，需重启修复
- 所有文档已更新到 PK metrics 评测完成（Run 4 iPKnot + Run 5 ArchiveII 的 pseudoknot 指标）
- ArchiveII/iPKnot 的 PK metrics 已从 saved predictions 验证过（torcheval F1 bit-for-bit 匹配 REPRODUCTION.md）
- 验证脚本 `scripts/run_verification.sh` 已写好，重启后可直接用
- `eval_pk_from_predictions.py` 已创建并通过 code review

## 阶段 A：验证所有 checkpoint（需要 GPU，重启后执行）

**目标**：确保每个 benchmark 的 checkpoint 推理结果和文档完全一致。

**脚本**：`scripts/run_verification.sh`（已写好，可直接运行）

**步骤**：
1. 重启机器后确认 `nvidia-smi` 正常，所有 8 张 H100 无 ERR
2. 激活 ufold-repro env：`conda activate ufold-repro`
3. 运行：`bash /data/xiwang_home/project/workspace/UFold/scripts/run_verification.sh`
4. 等待 ~30 min，3 条 chain 并行（GPU 1/5/7）
5. 对比 7 个 log 文件中的 metrics 和下面的期望值

**7 个验证任务及期望值**：

| exp | GPU | Checkpoint | Test file | Expected P | Expected R | Expected F1 | Expected AUROC | Expected AUPRC | Log |
|---|---|---|---|---|---|---|---|---|---|
| exp35 | 1 | models_archiveII/ufold_train_rivals_99.pt | mxfold2/archiveII.pkl | 0.6831 | 0.6533 | 0.6584 | 0.8333 | 0.5755 | logs/verify_archiveII.log |
| exp36 | 5 | models_ipknot/ufold_train_rivals_99.pt | ipkont/bpRNA-PK-TS0-1K.pkl | 0.4093 | 0.6118 | 0.4118 | 0.7349 | 0.3275 | logs/verify_ipknot.log |
| exp37 | 7 | models_unirna_ss/ufold_train_rivals_99.pt | all_data_1024_0.75/test.pkl | 0.4514 | 0.6383 | 0.4394 | 0.7422 | 0.3420 | logs/verify_unirna_ss.log |
| exp38 | 7 | models_bprna1m/ufold_train_rivals_9.pt | mxfold2/bpRNAnew.pkl | 0.5273 | 0.5817 | 0.5387 | 0.8283 | 0.4080 | logs/verify_bprna1m_new.log |
| exp39 | 5 | models_bprna1m/ufold_train_rivals_99.pt | mxfold2/TS0-canonicals.pkl | 0.4786 | 0.6786 | 0.4653 | 0.7583 | 0.3828 | logs/verify_bprna1m.log |
| exp40 | 5 | models_rivals/ufold_train_rivals_99.pt | rivals/TestSetA-addss.pkl | 0.7084 | -- | 0.6343 | 0.8127 | 0.5167 | logs/verify_rivals_testA.log |
| exp41 | 5 | models_rivals/ufold_train_rivals_99.pt | rivals/TestSetB-addss.pkl | 0.5428 | -- | 0.4145 | 0.6890 | 0.2562 | logs/verify_rivals_testB.log |

> 注：Rivals 原始 Run 1 未报 recall（旧版代码），fresh eval 会多出 recall 值，不算不一致。

## 阶段 B：组织 checkpoint 到统一目录

验证全部通过后：

```bash
mkdir -p checkpoints
cp models_rivals/ufold_train_rivals_99.pt       checkpoints/rivals_ep99.pt
cp models_unirna_ss/ufold_train_rivals_99.pt    checkpoints/unirna_ss_ep99.pt
cp models_bprna1m/ufold_train_rivals_99.pt      checkpoints/bprna1m_ep99.pt
cp models_bprna1m/ufold_train_rivals_9.pt       checkpoints/bprna1m_ep9.pt
cp models_archiveII/ufold_train_rivals_99.pt    checkpoints/archiveII_ep99.pt
cp models_ipknot/ufold_train_rivals_99.pt       checkpoints/ipknot_ep99.pt
md5sum checkpoints/*.pt > checkpoints/MD5SUMS
```

共 6 个文件 ~204 MB。models_mxfold2/ 丢弃（与 bprna1m 重复）。

## 阶段 C：文档整理 + REVIEWER_README.md

### 文档去重原则
- **Benchmark.md** = 纯结果表格 + 复现命令，不含方法论解释
- **REPRODUCTION.md** = 每个 Run 的详细记录（数据、代码改动、合法性检查）
- **CHANGE_LOG.md** = 时间线日志，每条简短
- 重复内容用交叉引用（"详见 Benchmark.md §X"）代替复制

### REVIEWER_README.md 要点
- 一键 setup（conda env create + 从 Google Drive 下载数据/ckpt）
- 一键 eval（eval_from_checkpoint.py 对每个 benchmark 跑一遍）
- 一键 train（ufold_train_rivals.py 复现训练）
- 额外 PK 评测（eval_pk_from_predictions.py 对 ArchiveII + iPKnot）
- 每个 benchmark 列出期望的 metric 值，审稿人可直接比对

## 阶段 D：打包上传（rclone gdrive_xw:）

### 数据集 → gdrive_xw:UniRNA/ss_dataset/

| 数据集目录 | 要上传的文件 | 预计大小 |
|---|---|---|
| all_data_1024_0.75/ | 原样全部（train.pkl, valid.pkl, test.pkl） | 3.4 GB |
| ipkont/ | 原样全部（bpRNA-TR0.pkl, bpRNA-PK-TS0-1K.pkl） | 3.0 GB |
| rivals/ | 仅 6 个文件：{TrainSetA,TestSetA,TestSetB}-{addss,eternafold}.pkl | ~TBD |
| mxfold2/ | 仅 6 个文件：TR0-canonicals.pkl, VL0-canonicals.pkl, TS0-canonicals.pkl, RNAStrAlign600-train.pkl, archiveII.pkl, bpRNAnew.pkl | ~12 GB |

### 代码+权重 → gdrive_xw:UniRNA/baselines/UFold/

整个 git repo + checkpoints/ 目录。

### rclone 命令模板
```bash
# 数据集
rclone copy /home/xiwang/project/develop/data/all_data_1024_0.75/ gdrive_xw:UniRNA/ss_dataset/all_data_1024_0.75/ -P
rclone copy /home/xiwang/project/develop/data/ipkont/ gdrive_xw:UniRNA/ss_dataset/ipkont/ -P
# rivals 只传 6 个文件
for f in TrainSetA-addss TrainSetA-eternafold TestSetA-addss TestSetA-eternafold TestSetB-addss TestSetB-eternafold; do
    rclone copy "/home/xiwang/project/develop/data/rivals/${f}.pkl" gdrive_xw:UniRNA/ss_dataset/rivals/ -P
done
# mxfold2 只传 6 个文件
for f in TR0-canonicals VL0-canonicals TS0-canonicals RNAStrAlign600-train archiveII bpRNAnew; do
    rclone copy "/home/xiwang/project/develop/data/mxfold2/${f}.pkl" gdrive_xw:UniRNA/ss_dataset/mxfold2/ -P
done
```

## 阶段 E：Git commit + push

### 要 commit 的文件
- `eval_from_checkpoint.py` — standalone checkpoint evaluation
- `eval_no_postprocess.py` — postprocess ablation
- `eval_pk_from_predictions.py` — PK metric evaluation
- `run_exp.py` — process disguise wrapper
- `scripts/run_verification.sh` — checkpoint verification launcher
- `REVIEWER_README.md` — 审稿人复现指南（阶段 C 产出）
- `Benchmark.md` — 更新后的结果表
- `REPRODUCTION.md` — 更新后的 Run 详情
- `CHANGE_LOG.md` — 更新后的时间线
- `.gitignore` — 更新后
- `ufold/UFold_example_output.png` — 原仓库资源
- `ufold/UFold_logonew1.png`
- `ufold/UFold—logo.png`

### 不 commit
- `docs/superpowers/` — 排除
- `logs/` — 已在 .gitignore
- `models_*/` — 已在 .gitignore
- `checkpoints/` — 需加到 .gitignore，上传到 GDrive 而非 git

### Git 命令
```bash
git add eval_from_checkpoint.py eval_no_postprocess.py eval_pk_from_predictions.py run_exp.py
git add scripts/run_verification.sh
git add REVIEWER_README.md Benchmark.md REPRODUCTION.md CHANGE_LOG.md .gitignore
git add ufold/UFold_example_output.png ufold/UFold_logonew1.png "ufold/UFold—logo.png"
git commit -m "Add checkpoint verification, PK evaluation, reviewer README, and doc cleanup"
git push origin main
```

## UFold 全数据集结果总结（供参考）

### 标准指标

| Benchmark | Train | Test | Ckpt | P | R | F1 | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|
| Rivals / TestSetA | TrainSetA-addss (3166) | TestSetA-addss (592) | rivals/ep99 | 0.7084 | -- | 0.6343 | 0.8127 | 0.5167 |
| Rivals / TestSetB | TrainSetA-addss (3166) | TestSetB-addss (430) | rivals/ep99 | 0.5428 | -- | 0.4145 | 0.6890 | 0.2562 |
| UniRNA-SS | train (8323) | test (1041) | unirna_ss/ep99 | 0.4514 | 0.6383 | 0.4394 | 0.7422 | 0.3420 |
| bpRNA-1m | TR0-canonicals (10814) | TS0-canonicals (1304) | bprna1m/ep99 | 0.4786 | 0.6786 | 0.4653 | 0.7583 | 0.3828 |
| bpRNA-1m-new | TR0-canonicals (10814) | bpRNAnew (5401) | bprna1m/ep9 | 0.5273 | 0.5817 | 0.5387 | 0.8283 | 0.4080 |
| ArchiveII | RNAStrAlign600-train (20923) | archiveII (3961) | archiveII/ep99 | 0.6831 | 0.6533 | 0.6584 | 0.8333 | 0.5755 |
| iPKnot | bpRNA-TR0 (10814) | bpRNA-PK-TS0-1K (2909) | ipknot/ep99 | 0.4093 | 0.6118 | 0.4118 | 0.7349 | 0.3275 |

### Pseudoknot 指标（ArchiveII + iPKnot）

| Benchmark | n_pk | score | score_pk | pk_sen | pk_ppv | pk_f1 |
|---|---|---|---|---|---|---|
| ArchiveII | 1079 (27.2%) | 0.6576 | 0.2167 | 0.0045 | 0.0011 | 0.0013 |
| iPKnot | 353 (12.1%) | 0.4105 | 0.1869 | 0.0667 | 0.0654 | 0.0639 |

## 已知问题

1. **GPU 2 ERR** — 需重启修复。重启后验证 `nvidia-smi` 无 ERR 再开始阶段 A
2. **Claude Code 沙盒无法使用 GPU** — 所有 GPU 推理命令必须在独立终端运行（不通过 Claude Code 的 Bash 工具或 `!` 前缀）
3. **Rivals 缺 recall** — Run 1 旧版代码未计算 recall。fresh eval 会产生 recall 值但不影响其他指标的验证
