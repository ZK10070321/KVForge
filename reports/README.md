# 参考实验

这些文件是固定运行的原始快照；重新执行请使用新的runs/目录，不覆盖参考数据。
CPU、FP32、模型与数据规模等条件记录在各JSON的environment和配置字段中。

| 目录 | 实验 | 文件 |
|---|---|---|
| day4_cpu | 随机权重模型，18组头数/后端/长度对照 | [JSON](day4_cpu/results.json)、[CSV](day4_cpu/summary.csv)、[Naive图](day4_cpu/naive.png)、[SDPA图](day4_cpu/sdpa.png) |
| day4_checkpoint | 已训练checkpoint的短上下文速度 | [JSON](day4_checkpoint/results.json)、[CSV](day4_checkpoint/summary.csv) |
| day5_cpu | 完整与预算缓存，16个共同验证位置 | [JSON](day5_cpu/results.json)、[CSV](day5_cpu/summary.csv)、[质量与存储图](day5_cpu/quality_memory.png) |

解读规则：

- 速度比较只适用于记录的硬件和工作量，不代表GPU或线上服务成绩。
- KV张量字节数不是整个模型的内存/显存峰值。
- 预算质量样本只有16个评分位置，适合流程诊断，不支持普遍优越性结论。
- JSON中的路径标识原始运行来源，移动报告目录不改写历史配置。

复现：[性能指南](../docs/benchmarking.md) · [预算指南](../docs/budget-cache.md)。
