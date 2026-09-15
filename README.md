# KVForge

A small decoder-only language model with training, KV caching, and reproducible inference experiments.

基于 PyTorch 的 Decoder-only 语言模型与推理实验项目。实现现代注意力模块、可恢复训练、增量 KV Cache，以及固定预算下的质量与存储对照。

## 功能

- **模型**：RMSNorm、RoPE、SwiGLU、MHA/GQA/MQA、causal attention、embedding/output权重共享。
- **训练**：字符级数据处理、训练/验证隔离、AdamW、学习率调度、梯度裁剪和完整checkpoint恢复。
- **推理**：预分配紧凑KV、增量生成、位置偏移；recent与sink_recent预算策略。
- **评测**：固定输入性能对照、共同预测位置的NLL/PPL、JSON/CSV与图表。

## 快速开始

已验证：Python 3.11 / PyTorch 2.5.1+cpu / Windows。以下命令均在项目根目录运行。

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test*.py" -v
```

预期30项测试通过。使用样例文本训练小模型：

```powershell
python prepare_data.py --input examples/day2_sample.txt --output data/quickstart
python train.py --data data/quickstart --out runs/quickstart_train --steps 200 --warmup 10 --seq-len 32 --batch-size 4 --dim 64 --layers 2 --heads 4 --kv-heads 2 --hidden-dim 176 --eval-every 20
python generate.py --checkpoint runs/quickstart_train/last.pt --prompt "The " --max-new-tokens 20 --temperature 0 --use-cache
```

去掉 `--use-cache` 可运行无缓存基线。训练恢复和数据格式见 [训练指南](docs/training.md)。

## 运行实验

性能对照使用固定token回放，可直接创建随机权重小模型：

```powershell
python benchmark.py --kv-heads 4 2 1 --out runs/benchmark
```

质量对照使用同一checkpoint的真实验证数据：

```powershell
python evaluate_budget.py --checkpoint runs/quickstart_train/last.pt --data data/quickstart --out runs/budget
```

绘图：

```powershell
python -m pip install -r requirements-benchmark.txt
python plot_day4.py --input runs/benchmark/results.json
python plot_day5.py --input runs/budget/results.json
```

输出目录非空时请选择新的 `--out`，保留各次原始结果。详细口径见 [性能评测](docs/benchmarking.md) 和 [预算评测](docs/budget-cache.md)。

## 参考结果

| 实验 | 配置 | 观察 |
|---|---|---|
| 完整KV缓存 | CPU FP32，SDPA/GQA，dim128，2层，batch1，提示词256，后续16步 | decode 216.3 → 734.7 token/s，3.396× |
| recent预算缓存 | 已训练小模型，窗口32，预算8，16个共同评分位置 | 常驻KV 16 → 4 KiB；NLL增加0.001358 |

性能数据是固定输入的模型路径计时，不是端到端服务成绩。质量实验样本很小，不能据此证明策略普遍有效。
完整配置、原始样本及图表见 [实验记录](reports/README.md)。

## 目录

```text
kvforge/             模型、训练辅助、缓存与评测实现
tests/               模型、恢复、缓存和评测的回归测试
docs/                架构与运行指南
  learning/          学习问答与项目复习
examples/            最小文本样例
reports/             可复核的参考结果
*.py                 数据准备、训练、生成、评测和绘图入口
```

本地数据放在 `data/`，训练和实验输出放在 `runs/`；二者默认不纳入Git。
文档入口见 [docs/README.md](docs/README.md)，整体流程见 [架构说明](docs/architecture.md)。

## 范围与限制

- 已完成CPU验证；CUDA接口已提供，GPU性能尚未实测。
- 样例模型与语料较小，不代表成熟的语言生成能力。
- 完整生成超过窗口后重建缓存；预算缓存仅支持逐token输入，不超过模型原窗口。
- 预算数值只限制常驻KV槽数，位置表和临时张量另有开销。
- 尚无变长请求调度、PagedAttention、多卡或自定义CUDA kernel。

目录组织参考 [nanoGPT](https://github.com/karpathy/nanoGPT) 的轻量运行入口与 [minGPT](https://github.com/karpathy/minGPT) 的独立模型包。
