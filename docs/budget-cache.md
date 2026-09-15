# 缓存预算与预测质量

本文命令均在项目根目录执行。

目标：限制常驻KV位置数，观察预测质量与存储开销的关系。
本实现是可解释的小型实验，使用已有Day 2 checkpoint，不需要重新训练。

## 1. 新增与调整的代码

| 文件 | 内容 |
|---|---|
| kvforge/budget_cache.py | 最近位置、起始位置+最近位置两种淘汰策略 |
| kvforge/cache.py | next_position/key_positions/commit统一接口 |
| kvforge/attention.py | 按缓存提供的真实位置建立因果mask |
| kvforge/model.py | 创建预算缓存，全部层完成后统一提交 |
| kvforge/budget_evaluation.py | 固定验证文本、共同预测位置、NLL/PPL |
| evaluate_budget.py | 数据/权重指纹检查，CLI与JSON/CSV |
| plot_day5.py | NLL、预测一致率与KV存储图 |
| tests/test_day5.py | 7项新增测试，包含独立数学参考 |

训练、Day 3完整缓存、Day 4速度评测和旧checkpoint格式仍可使用。
现有generate.py默认生成行为不变；预算策略由显式缓存API和质量评测入口使用。

## 2. 策略与预算口径

预算包含当前token。容量4意味着注意力最多访问4个K/V位置，而不是4个历史再加当前。

- full：保留整个窗口，作为质量参考。
- recent：淘汰最旧位置，例如[0,1,2,3]+新4变成[1,2,3,4]。
- sink_recent：保留开头sink_size个位置，其余留给最近位置；sink_size=1时，[0,5,6,7]+新8变成[0,6,7,8]。

length是保留数量；seen_tokens是已经处理数量。它们在淘汰后不同。
新Q/K按seen_tokens旋转；已缓存K不重复旋转、不重新编号。
位置表是额外int64张量，单独报告，不混进KV字节公式。

只支持逐token调用，包括提示词逐token预填充。这样每个query使用明确预算，避免多token块同时淘汰带来的歧义。
只在模型原max_seq_len范围内评估，不宣称位置外推或无限长流式推理。
这里借鉴起始+最近的选择形式，未复现完整StreamingLLM，也未证明起始token具有普遍优势。

## 3. 运行测试

```powershell
python -m unittest discover -s tests -p "test_day5.py" -v
python -m unittest discover -s tests -p "test*.py" -v
```

预期分别为 `Ran 7 tests ... OK`、`Ran 30 tests ... OK`。
独立参考测试对整段输入使用按策略构造的稀疏因果mask，不调用BudgetKVCache的选择逻辑。
它验证淘汰后的输出是否符合所定义策略；预算未满时，预算缓存还应与完整前向一致。
另测容量1、MHA/GQA/MQA、两种后端、batch=2、重置、半次写入失败、位置上限、评分标签与CLI。

## 4. 用原Day 2 checkpoint运行

```powershell
python evaluate_budget.py --checkpoint runs/day2_run_check/last.pt --data data/prepared --out runs/day5_verify
python plot_day5.py --input runs/day5_verify/results.json
```

预期第一行：

```text
windows=2, scored_tokens=16, window=32, score_from=24
```

然后7行：full，以及预算8/16/24的recent和sink_recent各一行。
生成results.json、summary.csv、quality_memory.png。
目录非空时换一个--out；不要覆盖原始结果。
数据指纹必须与checkpoint一致，本次实际匹配目录是data/prepared，不是data/day2_check。
如果之后换过模型或数据，窗口数和分数可能不同，不能要求照抄下面的数值。

## 5. 为什么只有16个评分位置？

原验证集96个字符，每个窗口取33个token（32输入+下一token标签），窗口起点0、32。
起点64处只剩32个token，不足33，因此丢弃尾部。
每个窗口位置0..23只用于提供上下文，位置24..31的8个预测参与打分。
两个窗口共16个预测位置，所有策略完全相同。
每个输入位置预测真实文本的下一token，不把预测结果接回输入。
默认共同评分起点取最大预算24，使所有压缩预算在评分时已开始淘汰。

这个样本太小，只是可运行的质量诊断；不能声称统计显著或普遍最优。
可在同一报告中调整--score-from 0来评估全部位置，但必须重新运行所有策略，并标明新评分口径。
若要形成更可靠结论，需要更大的独立验证文本及相应模型实验，不应只把当前16个位置重复多次。

## 6. 指标

- NLL：正确下一token的负对数概率均值，越低越好，单位为nat/token。
- PPL：exp(NLL)，同一词表与评分文本内越低越好；这里是字符级PPL。
- delta_nll：预算策略NLL减full NLL。负值只表示本次这些位置上较低，不证明总体质量提升。
- argmax_agreement：与full选出的最大分数token相同的比例，不是对真实标签的准确率。
- kv_allocated_bytes：单个窗口、batch=1的常驻K/V张量存储。多个窗口依次复用评测流程，不按窗口数相乘。
- position_metadata_bytes：预算位置表开销。压缩移动和注意力会产生临时张量，未计入常驻KV公式。

Day 4测速度；Day 5测质量与存储，不声称这里的淘汰实现必然更快。

## 7. 2026-09-15 实测

Windows / PyTorch2.5.1+cpu / FP32；旧checkpoint、词表35、窗口32、sink_size=2。

| 策略 | 预算 | KV KiB | NLL | PPL | 相对full的NLL变化 |
|---|---:|---:|---:|---:|---:|
| full | 32 | 16 | 2.186747 | 8.9062 | 0 |
| recent | 8 | 4 | 2.188105 | 8.9183 | +0.001358 |
| sink_recent | 8 | 4 | 2.202081 | 9.0438 | +0.015334 |
| recent | 16 | 8 | 2.185462 | 8.8948 | -0.001285 |
| sink_recent | 16 | 8 | 2.185364 | 8.8939 | -0.001383 |
| recent | 24 | 12 | 2.183936 | 8.8812 | -0.002811 |
| sink_recent | 24 | 12 | 2.185470 | 8.8948 | -0.001276 |

预算8的KV张量少75%，但不能写成总显存少75%。保留开头在该预算下反而损失更大，数据原样记录。
原始记录见[reports/day5_cpu/results.json](../reports/day5_cpu/results.json)。

## 8. 直接调用预算缓存

```python
import torch
from kvforge import ModelConfig, MiniLLM

model = MiniLLM(ModelConfig()).eval()
cache = model.create_budget_cache(1, capacity=4, policy='sink_recent', sink_size=1)
ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
with torch.no_grad():
    for i in range(ids.size(1)):
        logits, _ = model(ids[:, i:i+1], cache=cache)
print(cache.positions[:cache.length].tolist())  # [0,3,4,5]
print(cache.length, cache.next_position)         # 4,6
cache.reset()  # 新文本重新开始；模型权重更新后也应重新创建或重置缓存。
```

如果某层在写缓存之后报错，缓存标记为未完成；再次使用会明确报错，需要reset并从窗口开头重算。

## 9. 三次学习与收尾

1. 预算、真实位置、淘汰与Day 3重建的区别：budget_cache.py。
2. 真实token回放、标签、共同后缀、NLL/PPL：budget_evaluation.py和evaluate_budget.py。
3. 独立参考测试、图表、局限和项目表达：test_day5.py、plot_day5.py、README。

项目交付记录与简历表述见[项目总结](learning/project-summary.md)。
