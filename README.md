# KVForge

> Modern Decoder LLM Training and KV-Efficient Inference Framework

**当前进度：Day 2 CPU 训练闭环已验证。** 文本预处理、训练与验证、checkpoint 恢复和无缓存生成全部跑通，13 项测试通过。分步操作见 [DAY2.md](DAY2.md)。

KVForge 是一个基于 PyTorch 从零构建的模块化 Decoder-only 语言模型项目，目标是实现现代 Decoder 架构、增量 KV Cache，以及可复现的推理性能评测。

项目已完成现代 Decoder 架构与样例文本训练验证。增量 KV Cache、较大语料训练及推理 benchmark 将在后续阶段加入。下文保留 Day 1 架构基线及 Day 2 实测记录。

## Day 2：训练、恢复与生成闭环

验证日期：2026-09-10。运行环境：Windows / CPU / FP32，注意力后端为 SDPA。

### 已完成的功能

- 字符级 tokenizer：先切分文本，再仅使用训练段建立词表；保存 token 张量、词表与数据指纹。
- 训练循环：AdamW、线性 warmup、余弦学习率衰减、梯度范数裁剪，支持随机窗口与固定 batch 训练。
- 验证日志：独立采样器选择固定验证窗口，记录 train loss、validation loss 和学习率。
- 断点恢复：保存模型、优化器、GradScaler、配置、步数和随机状态；恢复时核对数据指纹。
- 自回归生成：实现贪心解码、temperature 和 top-k 采样，作为 Day 3 的无缓存基线。
- CUDA FP16 接口已实现；本次验证仅覆盖 CPU FP32，尚未进行 GPU 实测。

### 实验配置

使用仓库自带原创短文本 `examples/day2_sample.txt` 验证完整流程。

| 配置项 | 本次取值 |
|---|---:|
| 训练 / 验证 token 数 | 859 / 96 |
| 词表大小（含未知字符） | 35 |
| 验证集未知字符数 | 0 |
| 模型维度 / 层数 | 64 / 2 |
| Q 头数 / KV 头数 | 4 / 2 |
| SwiGLU 中间维度 | 176 |
| Batch size / 序列长度 | 4 / 32 |
| 可训练参数量 | 94,720 |
| 正常训练总步数 / warmup | 200 / 10 |
| 峰值 / 最低学习率 | 3e-4 / 3e-5 |
| 验证间隔 / 验证 batch 数 | 20 / 5 |

该配置用于快速验证，与下文 Day 1 默认模型规模不同。

### 固定 batch：可学习性检查

使用峰值学习率 0.003、warmup 5 步，重复学习同一批样本，共 100 步。

| Step | Train loss | Validation loss |
|---:|---:|---:|
| 20 | 1.8439 | 2.6603 |
| 40 | 0.3838 | 3.2587 |
| 60 | 0.0940 | 3.8405 |
| 80 | 0.0609 | 4.0366 |
| 100 | 0.0571 | 4.0615 |

训练 loss 明显下降，验证 loss 后期上升，符合固定样本过拟合现象。本实验用于验证模型能够学习，不作为泛化成绩。

### 正常训练与恢复

总计划 200 步，在第 100 步保存并暂停，再从 checkpoint 恢复。

| 阶段 / Step | Train loss | Validation loss |
|---|---:|---:|
| 初始化 | — | 3.6104 |
| 20 | 3.2348 | 3.2128 |
| 60 | 2.9612 | 2.7631 |
| 100（暂停） | 2.6482 | 2.5482 |
| 恢复后、继续训练前 | — | 2.5482 |
| 140 | 2.5080 | 2.4534 |
| 180 | 2.5229 | 2.4169 |
| 200 | 2.5216 | 2.4062 |

暂停前与恢复后的验证 loss 均为 `2.5482325553894043`。最终 checkpoint 记录 `step=200`、`best_val=2.40616455078125`，日志连续保存 20～200 步共 10 条记录。

Train loss 来自当前 batch 更新前的前向计算；validation loss 来自更新后固定抽样窗口的平均值。两者不要求同步下降。验证文本只有 96 个字符，这些结果主要验证工程链路，不代表广泛的语言泛化能力。

### 复现命令

在项目根目录执行；以下输出目录应当是新目录。再次从头实验请更换目录名。

```powershell
python -m unittest discover -s tests -p "test*.py" -v
python prepare_data.py --input examples/day2_sample.txt

# 固定 batch 检查。
python train.py --out runs/day2_overfit_check --overfit --steps 100 --warmup 5 --seq-len 32 --dim 64 --layers 2 --heads 4 --kv-heads 2 --hidden-dim 176 --lr 0.003 --eval-every 20

# 总计划 200 步，在第 100 步暂停。
python train.py --out runs/day2_run_check --steps 200 --warmup 10 --seq-len 32 --dim 64 --layers 2 --heads 4 --kv-heads 2 --hidden-dim 176 --eval-every 20 --stop-after 100

# 恢复原配置，继续至第 200 步。
python train.py --resume runs/day2_run_check/last.pt --out runs/day2_run_check

# 随机采样与贪心生成。
python generate.py --checkpoint runs/day2_run_check/best.pt --prompt "The " --max-new-tokens 100 --temperature 0.8 --top-k 20
python generate.py --checkpoint runs/day2_run_check/best.pt --prompt "The " --max-new-tokens 100 --temperature 0
```

### 生成结果与当前边界

随机采样输出节选：

```text
The lsthe fooii.qCdtea sn.
```

贪心输出节选：

```text
The the the t the the the the the the the the t
```

两种路径均正常完成，但文本尚不连贯，贪心生成出现重复。短样例和小模型用于验证流程，不以语言生成质量达标为结论。

当前采用字符级 tokenizer，无 EOS 停止规则；超长上下文保留最近窗口，并重置窗口内位置。KV Cache 尚未实现，因此不报告推理加速或缓存显存收益。

### 测试与输出文件

本次实测：

```text
Ran 13 tests in 13.356s

OK
```

新增的 4 项测试覆盖字符编码往返、词表隔离与标签错位、生成边界，以及连续训练与恢复一致性。恢复测试比较同一 CPU 环境下连续 8 步与暂停后恢复至 8 步的最终权重，要求数值完全相等；不代表跨设备或混合精度的一致性保证。

- `last.pt`：最近保存的训练现场，用于继续训练。
- `best.pt`：按验证 loss 选择的最佳已保存模型。
- `metrics.jsonl`：训练步数、训练/验证损失与学习率日志。

数据和 checkpoint 按 `.gitignore` 保留在本地。

## 项目目标

- 从底层理解现代 Decoder-only Transformer 的数据流与张量形状。
- 实现 RMSNorm、RoPE、SwiGLU 和 GQA 等现代架构组件。
- 实现紧凑的增量 KV Cache，减少自回归解码中的重复计算。
- 对比 MHA/GQA、Cache/No-Cache、Naive Attention/SDPA 的延迟、吞吐和缓存占用。
- 通过固定配置、正确性测试和 benchmark 脚本保证实验可复现。

## Day 1：现代 Decoder 基线

### 已实现

- `ModelConfig`：集中管理模型规模、注意力头数、最大序列长度及实现后端，并在模型创建前验证维度约束。
- `RMSNorm`：沿 token 特征维度进行均方根归一化，关键统计量使用 FP32 计算。
- `RoPE`：对 Query 和 Key 应用旋转位置编码，并支持 `start_pos` 位置偏移，为增量解码预留接口。
- `SwiGLU`：使用 SiLU 门控的前馈网络，将中间表示扩展后投影回残差维度。
- `GQA`：8 个 Query 头共享 2 个 Key/Value 头；同一实现也支持 MHA 和 MQA 配置。
- `Causal Self-Attention`：提供可读的手写实现和 PyTorch SDPA 实现，严格阻止未来信息泄漏。
- `Pre-Norm Transformer Block`：使用两组 RMSNorm 和两条残差连接组合 Attention 与 FFN。
- `MiniLLM`：完成 token embedding、多层 Decoder、语言模型输出头、权重共享和 next-token loss。
- `Smoke Test`：验证前向传播、有限梯度及一次 AdamW 参数更新。
- `Unit Tests`：验证公式、因果性、后端一致性、GQA 等价性和小样本可学习性。

### 模型数据流

```text
token ids [B, T]
        │
        ▼
Token Embedding
        │ [B, T, C]
        ▼
┌───────────────────────────────┐
│ RMSNorm → Causal GQA → Residual│
│ RMSNorm → SwiGLU     → Residual│ × N layers
└───────────────────────────────┘
        │ [B, T, C]
        ▼
Final RMSNorm → LM Head
        │
        ▼
logits [B, T, V]
```

其中 `B` 为 batch size，`T` 为序列长度，`C` 为模型维度，`V` 为词表大小。

### 默认配置

| 配置 | 数值 | 含义 |
|---|---:|---|
| `vocab_size` | 4096 | 词表大小 |
| `dim` | 256 | token 隐藏向量维度 |
| `n_layers` | 4 | Transformer Block 数量 |
| `n_heads` | 8 | Query 头数 |
| `n_kv_heads` | 2 | Key/Value 头数 |
| `head_dim` | 32 | 每个注意力头的维度 |
| `hidden_dim` | 704 | SwiGLU 中间维度 |
| `max_seq_len` | 512 | 最大输入长度 |

默认模型共有 **3,868,928 个可训练参数**。Embedding 与 LM Head 共享权重，因此共享参数只统计一次。

## 快速开始

### 1. 安装环境

建议使用 Python 3.10 或更高版本，并安装 PyTorch 2.2+：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "torch>=2.2"
```

如需 CUDA，请根据本机驱动和系统环境使用 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)安装对应构建，并用下面的命令确认环境：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Day 1 的所有正确性测试均可在 CPU 上运行。

### 2. 运行完整训练链路检查

手写 Attention：

```powershell
python smoke.py
```

PyTorch SDPA：

```powershell
python smoke.py --backend sdpa
```

当前 Day 1 基线的实测输出：

```text
torch=2.5.1+cpu, device=cpu, backend=naive
parameters=3,868,928
logits=(2, 32, 4096), loss=8.3772; backward + optimizer OK
```

这里的输入是随机 token，`loss=8.3772` 仅用于验证数值与训练链路正常，不代表模型的语言能力或训练效果。

### 3. 运行正确性测试

```powershell
python -m unittest discover -s tests -p "test*.py" -v
```

当前测试命令会运行 13 项测试。以下保留 Day 1 当日的 9 项测试记录：

```text
Ran 9 tests in 1.245s

OK
```

测试覆盖：

1. 非法配置与维度约束。
2. RMSNorm 与直接公式的一致性。
3. RoPE 的零位置、保长性质及位置偏移。
4. 因果注意力不存在未来信息泄漏。
5. Naive Attention 与 SDPA 的前向、梯度一致性。
6. GQA 与显式共享 KV 权重的 MHA 等价性。
7. next-token 标签对齐、有限梯度和 Weight Tying。
8. 最短、最长及非法序列边界。
9. 固定短序列上的小批次过拟合能力。

## 项目结构

```text
KVForge/
├── kvforge/
│   ├── __init__.py       # 包入口
│   ├── config.py         # 模型配置与维度检查
│   ├── layers.py         # RMSNorm、RoPE、SwiGLU
│   ├── attention.py      # GQA、因果遮罩、Naive/SDPA
│   ├── model.py          # Transformer Block 与完整语言模型
│   ├── data.py           # 字符词表、数据切分和 batch 采样
│   └── training.py       # 学习率、验证与 checkpoint 保存
├── tests/
│   ├── test_day1.py      # Day 1 正确性与可学习性测试
│   └── test_day2.py      # 数据、生成与恢复一致性测试
├── examples/
│   └── day2_sample.txt   # 原创流程检查文本
├── prepare_data.py       # 数据准备入口
├── train.py              # 训练与恢复入口
├── generate.py           # 无缓存生成入口
├── DAY2.md               # Day 2 分步复现指南
├── smoke.py              # 前向、反向和优化器链路检查
├── requirements.txt
└── README.md
```

## 实现说明

### GQA 的张量形状

默认配置中 `C=256`、`Hq=8`、`Hkv=2`、`D=32`：

```text
Input:  [B, T, 256]
Q:      [B, 8, T, 32]
K/V:    [B, 2, T, 32]
Scores: [B, 8, T, T]
Output: [B, T, 256]
```

Day 1 为便于理解，在注意力计算前显式扩展 K/V 头。后续 KV Cache 将保存扩展前的紧凑 `[B, Hkv, T, D]` 张量，因此当前版本尚不宣称获得 KV Cache 显存收益。

### 两种 Attention 后端

- `naive`：显式执行 `QKᵀ / √D`、因果遮罩、Softmax 和 Value 聚合，便于理解与调试。
- `sdpa`：使用 `torch.nn.functional.scaled_dot_product_attention`，为后续性能实验提供统一接口。

调用 SDPA 并不自动等同于使用 FlashAttention；实际内核由设备、dtype、张量形状和 PyTorch 后端共同决定。

## Roadmap

- [x] **Day 1**：现代 Decoder 组件、GQA 因果注意力、完整模型与正确性测试。
- [x] **Day 2**：字符级 Tokenizer、文本数据管线、训练/验证、warmup、checkpoint 恢复和生成样例；CPU 验证通过。
- [ ] **Day 3**：紧凑增量 KV Cache，以及 full forward 与逐 token decode 的 logits 一致性测试。
- [ ] **Day 4**：Cache/No-Cache、MHA/GQA、Naive/SDPA 的延迟、吞吐与缓存占用 benchmark。
- [ ] **Day 5**：缓存预算策略、实验图表、结果分析与项目文档整理。

## 参考实现

- [karpathy/llama2.c — model.py](https://github.com/karpathy/llama2.c/blob/master/model.py)
- [karpathy/nanoGPT — model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py)
- [Meta Llama — model.py](https://github.com/meta-llama/llama/blob/main/llama/model.py)
- [PyTorch Scaled Dot Product Attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

KVForge 以学习、实现和可复现实验为目标，当前代码为独立编写的教学实现，不兼容官方 Llama 模型权重。
