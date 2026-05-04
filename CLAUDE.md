# CLAUDE.md

## Project Overview

UFold reproduction for RNA secondary structure prediction on 5 benchmarks + BIB/CompaRNA cross-dataset evaluation.

## Environment

```bash
conda env create -f environment.yml
conda activate ufold-repro
pip install setproctitle "torcheval==0.0.6"
```

## Key Commands

### Evaluate from predictions (no GPU)
```bash
python eval_from_predictions.py --predictions predictions/archiveII.pkl
```

### Inference from checkpoints
```bash
python eval_from_checkpoint.py --gpu 0 --checkpoint checkpoints/archiveII.pt \
    --test_file data/mxfold2/archiveII.pkl
```

### Evaluate all benchmarks
```bash
bash scripts/eval_all.sh ./data ./checkpoints 0,1,2
```

### Train all benchmarks
```bash
bash scripts/train_all.sh ./data ./models_retrain 0
```

## Architecture

- `Network.py`: U-Net (img_ch=17, 8.6M params). Input: (batch, 17, L, L) → Output: (batch, L, L) contact map.
- `ufold/data_generator.py`: Data loading and 17-channel feature construction.
- `ufold/postprocess.py`: Augmented Lagrangian post-processing.
- `ufold/utils.py`: `creatmat()`, `seq_dict`, evaluation helpers.

## Checkpoints

5 checkpoints in `checkpoints/`, all epoch 99:
- `rivals.pt`, `unirna_ss.pt`, `bprna1m.pt`, `archiveII.pt`, `ipknot.pt`

## Results

All results in `RESULTS.md`. Key F1 values:
- Rivals TestSetA: 0.6343, TestSetB: 0.4145
- UniRNA-SS: 0.4394
- bpRNA-1m-new: 0.4639
- ArchiveII: 0.6584
- iPKnot: 0.4118

### 复现的核心注意事项和学术正直
- 维护CHANGE_LOG.md文件用于记录你的环境创建和代码修改，实验运行行为
- 尽可能按照原始仓库的ReadME来构建环境，自定义的适应操作是允许的，但是必须记录在CHANGE_LOG.md中
- 如果用户要求使用额外数据集进行训练和评测，最佳做法是写一个转化函数，将用户数据集转化成原始仓库的支持格式，并完整记录转化函数
- 最佳评测工作流是直接使用用户指明的验证和测试集，并尽可能不做修改的使用用户的评测函数，对于最终结果需要推理的到最终结果，便于用户后续直接评测
- 核心的训练集，验证集（如果有）和测试集必须由用户指明，你永远不允许修改训练中使用的数据集，或者对数据集做任何改动，每一次实验都需要调用subagents来判断实验的合法性
- 每次完成环境配置，原始论文结果复现，新数据集训练和测试后，都需要更新到REPRODUCTION.md文档中，所有的subagents都需要遵循这个原则
- 只使用Opus High Effort模型作为你的Subagents
- 每次新增的REPRODUCTION.md文档的内容，都需要额外调用Subagents做代码Review，着重评估代码Bug，评测策略和数据是否符合要求，是否存在任何数据集泄露，测试集训练，静默替换测试集的行为
- 维护Benchmark.md文档，记录官方数据集复现结果（和论文数值对比），以及所有额外数据集的结果，每一个数据集的结果都需要记录详细的日志，额外的代码改动，超参数
- 复现尽可能少添加和修改代码，所有的代码改动必须直观展示在git中，便于用户review，在未经允许的情况下，不要做git add/commit/push
- 复现过程中可能会出现中间数据文件，你可以添加到gitignore中，但必须明确告知用户
- Benchmark.md文档必须要写清楚新的Benchmark要怎么做，以及注意事项和禁止开展的行为
