# Day 2：数据、训练、恢复与生成

先运行代码，后续再逐段学习。新增 Python 文件含中文注释。
使用字符级 tokenizer：一个 Unicode 字符对应一个 token；它容易检查且不需要外部分词库。
这与子词 BPE 不同，后续可替换。Day 2 的词表大小从文本推导，不再固定为 4096。

## 1. 在项目根目录检查环境

```powershell
python -c "import sys,torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
python -m unittest discover -s tests -p "test*.py" -v
```

现有 Day 1 的 9 项测试加 Day 2 的 4 项测试，共 13 项。
CPU 可以完成整条流程；只有 CUDA 可用时才添加 `--device cuda --precision fp16`。
FP16 使用 GradScaler，梯度先还原再裁剪。默认 FP32，未自动修改你的 Python 环境。

## 2. 准备数据

先用项目自带原创短文本验证链路：

```powershell
python prepare_data.py --input examples/day2_sample.txt
```

写入 `data/prepared/train.pt`、`val.pt`、`tokenizer.json`。
原始文本前 90% 用于训练，后 10% 用于验证；词表仅从训练段构建。
验证段未见字符编码为 0，准备输出中的 `val_unknown` 记录它们的数量。
这份样例很短，仅用于流程检查；真实实验请换成有使用许可、更充足的 UTF-8 文本：

```powershell
python prepare_data.py --input "你的文本路径.txt" --output data/corpus
```

如果文本包含多个文档，当前按字符位置切分可能让同一文档跨训练/验证段，正式质量评测应另做按文档划分与去重。

## 3. 先用固定 batch 检查能否学习

```powershell
python train.py --out runs/overfit --overfit --steps 100 --warmup 5 --seq-len 32 --dim 64 --layers 2 --heads 4 --kv-heads 2 --hidden-dim 176 --lr 0.003 --eval-every 20
```

关注打印的 `train_loss` 是否明显下降。它来自更新前的固定 batch；验证 loss 不保证随之下降。

## 4. 正常采样训练与暂停

```powershell
python train.py --out runs/day2 --steps 200 --warmup 10 --seq-len 32 --dim 64 --layers 2 --heads 4 --kv-heads 2 --hidden-dim 176 --eval-every 20 --stop-after 100
```

这里学习率按总共 200 步计算，在第 100 步保存并退出。
样例文本过短，生成质量有限。真实文本可使用默认 128 维配置，并按显存调整 batch size 和 seq_len。

输出文件：

- `last.pt`：最近一次验证时的模型、优化器、GradScaler、步数、配置和随机状态。
- `best.pt`：验证 loss 最低的已保存 checkpoint。
- `metrics.jsonl`：每行一条 step/train_loss/val_loss/lr 记录。

## 5. 恢复训练

```powershell
python train.py --resume runs/day2/last.pt --out runs/day2
```

恢复使用 checkpoint 保存的模型和训练配置，完成剩余 100 步；无需再次填写 dim 等参数。
数据路径默认仍为 data/prepared，其他数据要显式提供相同的 `--data`。
恢复不会改变原来的总步数/学习率计划；若已达到总步数，则不再训练。
正常 Ctrl+C 后从最近一次保存恢复，会丢失尚未保存的几步；可减小 eval-every。
同一 CPU 环境的连续/恢复一致性由测试覆盖，跨设备不保证逐位一致。

## 6. 生成文本

```powershell
python generate.py --checkpoint runs/day2/best.pt --prompt "The " --max-new-tokens 200 --temperature 0.8 --top-k 20
```

temperature=0 使用贪心解码；top-k=0 表示不进行 top-k 过滤。
提示词必须由训练词表中的字符组成。当前没有 EOS，按指定数量生成。
超过上下文长度时裁剪最早 token，并在保留窗口中重置位置；这是 Day 2 无缓存基线。
Day 3 做严格缓存对照时需要明确位置与裁剪规则，不能把此行为与绝对位置缓存混用。

## 文件阅读顺序

`kvforge/data.py → prepare_data.py → kvforge/training.py → train.py → generate.py → tests/test_day2.py`

完成复现的标准：13 项测试通过；固定 batch loss 下降；正常训练有日志；checkpoint 能恢复；生成入口输出文字。
生成可运行不等于模型已经获得良好语言能力。CUDA 和真实语料质量需在你的训练环境进一步验证。
