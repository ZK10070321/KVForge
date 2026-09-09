# KVForge

> Modern Decoder LLM Training and KV-Efficient Inference Framework

KVForge 是一个基于 PyTorch 从零构建的模块化 Decoder-only 语言模型项目，目标是实现现代 Decoder 架构、增量 KV Cache，以及可复现的推理性能评测。

项目目前完成 **Day 1：模型架构与正确性基线**。当前版本实现了完整的前向传播、语言模型损失、反向传播和参数更新，并通过 9 项单元测试验证关键数学性质。KV Cache、真实文本训练和性能 benchmark 将在后续阶段加入。

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

Day 1 验证结果：

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
│   └── model.py          # Transformer Block 与完整语言模型
├── tests/
│   └── test_day1.py      # Day 1 正确性与可学习性测试
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
- [ ] **Day 2**：Tokenizer、文本数据管线、训练/验证划分、warmup、checkpoint 和生成样例。
- [ ] **Day 3**：紧凑增量 KV Cache，以及 full forward 与逐 token decode 的 logits 一致性测试。
- [ ] **Day 4**：Cache/No-Cache、MHA/GQA、Naive/SDPA 的延迟、吞吐与缓存占用 benchmark。
- [ ] **Day 5**：缓存预算策略、实验图表、结果分析与项目文档整理。

## 参考实现

- [karpathy/llama2.c — model.py](https://github.com/karpathy/llama2.c/blob/master/model.py)
- [karpathy/nanoGPT — model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py)
- [Meta Llama — model.py](https://github.com/meta-llama/llama/blob/main/llama/model.py)
- [PyTorch Scaled Dot Product Attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

KVForge 以学习、实现和可复现实验为目标，当前代码为独立编写的教学实现，不兼容官方 Llama 模型权重。
