"""
将 benchmark 的 JSON 画成独立 PNG；绘图与计时分开，避免互相干扰
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('runs/day4_cpu/results.json'))
    args = parser.parse_args()
    # matplotlib 仅绘图需要；核心测试和 benchmark 不依赖它
    import matplotlib
    # 直接保存图片，不弹出交互窗口
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 把 JSON 文件读回 Python 字典
    report = json.loads(args.input.read_text(encoding='utf-8'))
    rows = report['results']
    # 遍历每条实验结果，取出 backend，用集合去重，得到 naive、sdpa，排序后分别画图
    for backend in sorted({r['backend'] for r in rows}):
        # 每个后端创建四个子图：
        # 左上：decode吞吐   右上：decode加速比
        # 左下：prefill耗时  右下：KV缓存大小
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        selected = [r for r in rows if r['backend'] == backend]
        # 然后根据 KV 头数分组，再按提示词长度排序
        # 为什么排序:因为画线时要按照 32 → 128 → 256 连接，而不是按照文件中可能出现的任意顺序连接
        #          这里的排序只改变展示顺序，不改变原始成绩
        for index, heads in enumerate(sorted({r['kv_heads'] for r in selected})):
            group = sorted([r for r in selected if r['kv_heads'] == heads], key=lambda r: r['prompt_length'])
            x = [r['prompt_length'] for r in group]
            color = f'C{index % 10}'
            for mode, style in [('no_cache', '--'), ('cache', '-')]:
                label = f'Hkv={heads} {mode}'
                axes[0, 0].plot(x, [r[mode]['decode_tokens_per_second'] for r in group],
                                style, marker='o', color=color, label=label)
                axes[1, 0].plot(x, [r[mode]['prefill_ms_median'] for r in group],
                                style, marker='o', color=color, label=label)
            axes[0, 1].plot(x, [r['decode_speedup'] for r in group], marker='o', color=color, label=f'Hkv={heads}')
            axes[1, 1].plot(x, [r['cache']['cache_allocated_bytes']/1024 for r in group],
                            marker='o', color=color, label=f'Hkv={heads}')
        axes[0, 1].axhline(1, color='gray', linestyle=':', label='Equal speed')
        titles = [('Decode throughput (higher is better)', 'tokens / second'),
                  ('Decode speedup: no-cache time / cache time', 'ratio'),
                  ('Prefill only (allocation excluded)', 'milliseconds'),
                  ('Allocated KV tensors, not total device memory', 'KiB')]
        for axis, (title, ylabel) in zip(axes.flat, titles):
            axis.set_title(title, fontsize=10)
            axis.set_xlabel('Prompt tokens P')
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
            axis.set_ylim(bottom=0)
        env = report['environment']
        figure.suptitle(f'KVForge | {backend} | {env["device"]} {env["dtype"]} | '
                        f'{report["model_source"]} | fixed-token replay')
        target = args.input.parent / f'{backend}.png'
        # 保存图片，关闭绘图对象
        figure.savefig(target, dpi=160)
        plt.close(figure)
        print(f'Saved: {target}')


if __name__ == '__main__':
    main()
