# 推理性能评测

本文命令均在项目根目录执行。

Day 3 证明缓存算得对；Day 4 测量它在具体配置下的时间与存储开销。

## 1. 验收测试

```powershell
python -m unittest discover -s tests -p "test*.py" -v
```

预期 `Ran 30 tests` 和 `OK`。新增 5 项检查工作量、指标分母、容量限制及输出文件。
测试不会断言“缓存一定更快”，因为速度与机器、长度、精度有关。

## 2. 先跑小规模 CPU 示例

```powershell
python benchmark.py --contexts 16 32 --decode-steps 8 --kv-heads 4 2 --warmup 1 --repeats 3 --out runs/day4_quick
```

预期每个组合输出一行，格式类似（数字以本机实际结果为准）：

```text
naive Hkv=4 P=  16 no_cache=... tok/s cache=... tok/s speedup=...x
...
Saved: runs\day4_quick\results.json
```

一共 2 后端 × 2 KV 头配置 × 2 提示词长度 = 8 组，每组内部比较两种模式。
输出目录已有内容时程序会拒绝覆盖；重新运行请换 `--out runs/day4_quick_v2`。

## 3. 完整 CPU 对照

```powershell
python benchmark.py --kv-heads 4 2 1 --out runs/day4_cpu
python plot_day4.py --input runs/day4_cpu/results.json
```

默认参数：dim=128、层数=2、Q头数=4、词表=512、batch=1、上下文=32/128/256、
decode=16步、预热2轮、正式5轮、CPU线程数2、FP32。
4/2/1 个 KV 头分别对应 MHA/GQA/MQA，共 18 组比较。
随机权重只用于性能实验，不代表完成了这三个架构的训练或质量对比。

若绘图提示缺少 matplotlib，在同一个 Python 环境执行：

```powershell
python -m pip install -r requirements-benchmark.txt
```

生成文件：

- `results.json`：环境、配置、模型来源、正确性误差、容差及每轮原始耗时。
- `summary.csv`：每组 cache/no-cache 的汇总指标。
- `naive.png`、`sdpa.png`：吞吐、加速比、prefill 与缓存字节数四张子图。

## 4. 用 Day 2 的真实 checkpoint 检查

```powershell
python benchmark.py --checkpoint runs/day2_run_check/last.pt --contexts 8 16 --decode-steps 8 --out runs/day4_checkpoint
```

保持 checkpoint 原有头数和窗口，不接受 `--kv-heads`。
Day 2 的窗口为32，不能直接套用默认256提示词。P+N超出窗口时明确报错。
这里仍回放固定 token ID，用于控制工作量，不测自然文本质量。

## 5. CUDA（接口已提供，CPU 验收不要求执行）

只有 `torch.cuda.is_available()` 为 True 时执行：

```powershell
python benchmark.py --device cuda --dtype fp16 --kv-heads 4 2 1 --out runs/day4_cuda_fp16
```

显式转换模型精度，不开启 autocast。运行前后同步 CUDA；FP32 测量关闭 TF32。
本次没有可用 GPU，CUDA/FP16 路径和速度尚未实测，不能把 CPU 结果换算成 GPU 结果。

## 6. 指标的准确口径

设 P 为提示词长度、N 为 decode_steps、B 为 batch。
预先创建 P+N 个固定 token ID；prefill 处理前 P 个，然后执行 N 次模型调用。
第 i 次（从0开始）：基线处理前 P+i+1 个位置，缓存版只处理位置 P+i。
两者预测分数对应相同前缀。计时不采样，不做 tokenizer，不打印，不搬运输入。
因此 N 表示额外处理的 token 步数，不是 generate.py 中包含 prefill 首个预测的输出 token 口径。

| 字段 | 定义 |
|---|---|
| prefill_ms_median | 正式轮次 prefill 耗时中位数；不含缓存申请 |
| decode_ms_median | 每轮 N 步 decode 总耗时的中位数 |
| decode_step_ms | 上一项除以 N，是整个 batch 一步的平均耗时 |
| decode_tokens_per_second | B×N / decode秒数，是 batch 总吞吐 |
| decode_speedup | 无缓存 decode 中位时间 / 缓存 decode 中位时间 |
| cache_allocated_bytes | 预分配 K/V 张量字节数；本实验容量固定为 P+N |
| cuda_peak_allocated_bytes | 一轮含缓存申请、prefill、decode 的 PyTorch 张量分配峰值 |
| cuda_peak_extra_bytes | 上述峰值减去申请本轮缓存前的常驻张量基线 |

GPU峰值取各正式轮次最大值，时间取中位数，原始数据都保留。
CUDA统计不是 nvidia-smi 的进程总显存，也不是 reserved allocator 指标。
CPU 的 CUDA 字段是 null，CSV 中为空，表示没有测量；不是0字节。
时钟是 perf_counter，CUDA在阶段边界同步，所以包含Python调度，不是纯kernel时间。

## 7. 公平性与限制

- 同一 cache/no-cache 对照使用同一个模型对象与相同输入，计时前先检查所有位置 logits。
- 同一头配置的 naive/SDPA 加载同一份权重；Day 1 的后端一致性回归测试继续保留。
- 不同头数的权重形状、参数量可能不同，因此 MHA/GQA/MQA 性能比较不是同一模型的无损开关，也没有质量结论。
- 每轮创建新缓存，预热不计分，正式轮次交替模式执行顺序，不挑最快一次。
- P+N 必须不超过窗口，不允许窗口重建混入增量 benchmark。
- 同一上下文内两种模式工作量可比较，但基线计算全部位置 logits，缓存版只计算新增位置 logits；这是现有实现的整体路径比较，不是单独 attention kernel 对比。
- 小模型、小窗口可能无加速甚至变慢；保留小于1的比值，不能修改测试或挑选数据隐藏它。
- 不包含采样、排队、网络和真实请求调度，不能宣称端到端服务 TTFT/TPOT 或并发能力。

## 8. 三次学习安排

1. 实验设计与指标：P/N/B、固定输入、prefill/decode、吞吐与延迟；阅读 benchmark.py 的参数和 benchmarking.py 的 validate_workload。
2. 计时实现：正确性门槛、同步、预热、交替顺序、中位数、缓存内存；阅读 benchmarking.py 的其余函数。
3. 结果解读：JSON/CSV、绘图、测试、架构对照与性能限制；阅读 plot_day4.py 和 tests/test_day4.py，整理面试表达。

官方参考：
- https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution
- https://docs.pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html
