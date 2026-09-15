"""
在明确的条件下用固定的工作量测量程序表现
批量测试后端、KV 头数和上下文，输出可追溯的 JSON/CSV
默认是随机权重的小模型，只评测计算，不评价语言质量
提供 --checkpoint 时使用已有权重，不改变原模型头数或窗口
"""
import argparse
import csv
import hashlib
import json
import platform
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import torch
from kvforge import MiniLLM, ModelConfig
from kvforge.benchmarking import benchmark_pair


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # type=int：把输入转成整数；nargs="+"：允许提供一个或多个数字
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--dtype', choices=['fp32', 'fp16'], default='fp32')
    p.add_argument('--contexts', type=int, nargs='+', default=[32, 128, 256])
    p.add_argument('--decode-steps', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--backends', nargs='+', choices=['naive', 'sdpa'], default=['naive', 'sdpa'])
    # 表示分别测试两种配置，假设当前 Ｑ 头数为 4，则 KV 头数 4 时是 MHA；KV 头数 2 时是 GQA
    p.add_argument('--kv-heads', type=int, nargs='+', default=None)
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--vocab-size', type=int, default=512)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--out', type=Path, default=Path('runs/day4_cpu'))
    args = p.parse_args()
    if min(args.contexts + [args.decode_steps, args.batch_size, args.repeats, args.threads]) < 1 or args.warmup < 0:
        p.error('长度、batch、重复次数和线程数必须为正；warmup 不能为负')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('当前 PyTorch 无可用 CUDA；先使用 --device cpu')
    if args.device == 'cpu' and args.dtype != 'fp32':
        p.error('本实验 CPU 固定使用 fp32；fp16 留给 CUDA 单独测量')
    # 避免不同实验悄悄覆盖已有原始数据，重新测量时换一个 --out
    if args.out.exists() and any(args.out.iterdir()):
        p.error('输出目录非空，请换一个 --out 以保留原始结果')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == 'fp32' else torch.float16
    if device.type == 'cuda':
        # 明确 FP32 的设置，避免不同机器默认 TF32 开关影响比较
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    checkpoint = None
    checkpoint_hash = None
    if args.checkpoint:
        if args.kv_heads is not None:
            p.error('checkpoint 模式不允许改 KV 头数，否则权重形状不匹配')
        checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        base = ModelConfig(**checkpoint['model_config'])
        kv_heads = [base.n_kv_heads]
    else:
        base = ModelConfig(vocab_size=args.vocab_size, dim=args.dim,
            n_layers=args.layers, n_heads=args.heads, n_kv_heads=args.heads,
            hidden_dim=args.dim * 3, max_seq_len=max(args.contexts)+args.decode_steps)
        kv_heads = args.kv_heads if args.kv_heads is not None else [args.heads, 1]
    if max(args.contexts) + args.decode_steps > base.max_seq_len:
        p.error('prompt + decode 超过 checkpoint 窗口；例如 Day 2 用 --contexts 8 16 --decode-steps 8')

    report = {
        'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'environment': {'torch': torch.__version__, 'python': platform.python_version(),
            'platform': platform.platform(), 'device': str(device), 'dtype': args.dtype,
            'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
            'cuda_runtime': torch.version.cuda, 'threads': torch.get_num_threads()},
        'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'model_source': 'checkpoint' if checkpoint else 'random_weights',
        'checkpoint_sha256': checkpoint_hash,
        'method': 'fixed-token replay; prefill P then N forward steps; no sampling, no window rebuild; allocation excluded from timing',
        'results': [],
    }
    # 实验循环
    # 不同 kv 头数
    for heads in kv_heads:
        # 先创建一份 CPU 权重，两个后端加载完全相同的 state_dict
        torch.manual_seed(args.seed)
        config = replace(base, n_kv_heads=heads, attention_backend='naive')
        reference = MiniLLM(config).eval()
        if checkpoint:
            reference.load_state_dict(checkpoint['model'])
        weights = reference.state_dict()
        # 不同后端
        for backend in args.backends:
            config = replace(config, attention_backend=backend)
            model = MiniLLM(config).to(device=device, dtype=dtype).eval()
            model.load_state_dict(weights)
            # 不同提示词长度
            for context in args.contexts:
                # 独立 CPU 随机数生成器使输入不受模型初始化或实验执行顺序影响，若输入依赖共享随机状态，改变实验顺序可能意外改变测试输入
                # 随机种子能帮助固定输入，但不能保证每次耗时完全相同，系统负载和硬件状态仍会变化
                generator = torch.Generator().manual_seed(args.seed + context)
                # 从0到vocab_size-1生成合法token ID，形状为[B,P+N]
                ids = torch.randint(0, config.vocab_size,
                    (args.batch_size, context+args.decode_steps), generator=generator).to(device)
                # 每组实验运行后，代码得到一个字典
                row = benchmark_pair(model, ids, context, args.decode_steps, args.warmup, args.repeats)
                row.update({'backend': backend, 'kv_heads': heads, 'config': asdict(config),
                            'parameters': sum(t.numel() for t in model.parameters())})
                # 每组结果追加进去
                report['results'].append(row)
                print(f'{backend:5s} Hkv={heads} P={context:4d} '
                      f'no_cache={row["no_cache"]["decode_tokens_per_second"]:.1f} tok/s '
                      f'cache={row["cache"]["decode_tokens_per_second"]:.1f} tok/s '
                      f'speedup={row["decode_speedup"]:.3f}x')
            del model, ids
        del reference, weights
    args.out.mkdir(parents=True, exist_ok=True)
    # JSON 保存完整结构：report：要保存的Python数据
    #                  ensure_ascii=False：中文正常显示
    #                  indent=2：缩进排版，方便阅读
    # JSON 里除了成绩，还保存环境、配置和原始样本
    (args.out / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    # CSV 保存容易比较的平面表格
    # 每组分成 一行 no_cache、一行 cache，每行对应一种模式，便于直接用 Excel 查看；JSON 保留全部原始计时
    fields = ['backend', 'kv_heads', 'prompt_length', 'decode_steps', 'batch_size', 'mode',
              'prefill_ms_median', 'decode_ms_median', 'decode_step_ms',
              'decode_tokens_per_second', 'cache_allocated_bytes',
              'cuda_peak_allocated_bytes', 'cuda_peak_extra_bytes']
    with (args.out / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in report['results']:
            for mode in ('no_cache', 'cache'):
                flat = {k: row[k] for k in fields[:5]}
                flat.update({k: row[mode][k] for k in fields[6:]})
                writer.writerow({**flat, 'mode': mode})
    print(f'Saved: {args.out / "results.json"}')


if __name__ == '__main__':
    main()
