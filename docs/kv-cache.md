# 增量 KV Cache

本文命令均在项目根目录执行。

目标：复用历史 token 的 K/V，并证明计算结果与完整前向一致。
不用重新训练模型，也不需要先安装 CUDA。

## 1. 跑全部回归测试

在项目根目录执行：

```powershell
python -m unittest discover -s tests -p "test*.py" -v
```

预期末尾出现 `Ran 30 tests` 和 `OK`。Day 1 的 9 项、Day 2 的 4 项仍需通过；
Day 3 新增 5 项，内部包含不同后端、头数与输入分块的多组检查。
测试比较每个位置的 logits，不以“程序没有报错”代替正确性验证。

## 2. 使用昨天的 checkpoint 生成

先运行无缓存基线：

```powershell
python generate.py --checkpoint runs/day2_run_check/last.pt --prompt "The " --max-new-tokens 20 --temperature 0
```

再运行缓存版本：

```powershell
python generate.py --checkpoint runs/day2_run_check/last.pt --prompt "The " --max-new-tokens 20 --temperature 0 --use-cache
```

预期：两条命令输出相同文本。文本仍可能重复、不流畅，因为没有重新训练，
KV Cache 改变的是计算方式，不会提高模型本身的语言能力。
采用 temperature=0 是为了直接比较贪心结果；随机采样容易受到数值微小差异影响。

然后把两条命令的 `--max-new-tokens` 都改成 60，检查窗口满后的生成。
原 Day 2 checkpoint 的窗口为 32，因此此时缓存版会重建窗口，不能用这段耗时宣称增量加速。

## 3. 阅读顺序

1. `kvforge/cache.py`：缓存空间、有效长度、写入和重置。
2. `kvforge/attention.py`：新 Q 查询历史 K/V，RoPE 偏移及矩形因果遮罩。
3. `kvforge/model.py`：每层使用自己的缓存，全部成功后统一推进长度。
4. `generate.py`：第一次 prefill，之后每次输入一个新 token。
5. `tests/test_day3.py`：用无缓存结果作为参考检查缓存实现。

## 4. 直接调用接口

```python
import torch
from kvforge import MiniLLM, ModelConfig

model = MiniLLM(ModelConfig()).eval()
cache = model.create_cache(batch_size=1)
with torch.no_grad():
    # prefill：一次处理提示词，缓存长度变为 3。
    logits, _ = model(torch.tensor([[1, 2, 3]]), cache=cache)
    # decode：只输入新增 token；通过缓存看见前面三个位置。
    logits, _ = model(torch.tensor([[4]]), cache=cache)
    print(logits.shape)   # torch.Size([1, 1, 4096])
    print(cache.length)   # 4
cache.reset()             # 新文本必须重新开始，不能混入之前的历史。
```

低层接口超过容量会报错；`generate()` 会自动重建最近窗口。
缓存只支持等长、无 padding 的 batch。创建缓存后不要更改模型权重或设备；
改变模型后应重新创建缓存。此版本缓存推理不支持混用 autocast，默认 CPU FP32 即可。

## 5. 缓存大小与边界

常驻 K/V 字节数：`2 × 层数 × batch × KV头数 × 容量 × 每头维度 × 每元素字节数`。
K 已做 RoPE，V 没有做 RoPE；两者都在 KV 头扩展之前保存。
GQA 能减少这里的常驻缓存，但注意力计算仍会临时扩展 KV 头，不能把这个公式当成总显存。

完整前向与缓存前向有不同的矩阵形状，浮点结果允许合理误差。
性能测量和指标定义见 [推理评测](benchmarking.md)。
