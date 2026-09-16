"""
绘制共同评分位置上的质量—存储关系，不把不同文本的PPL放在一起比较
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path('runs/day5_budget/results.json'))
    args = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report = json.loads(args.input.read_text(encoding='utf-8'))
    rows = report['results']
    full = next(r for r in rows if r['policy'] == 'full')
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for policy in ('recent', 'sink_recent'):
        selected = sorted([r for r in rows if r['policy'] == policy], key=lambda r: r['capacity'])
        x = [r['kv_allocated_bytes']/1024 for r in selected]
        axes[0].plot(x, [r['nll'] for r in selected], marker='o', label=policy)
        axes[1].plot(x, [r['argmax_agreement'] for r in selected], marker='o', label=policy)
    axes[0].axhline(full['nll'], color='gray', linestyle=':', label='full NLL')
    for axis, field in [(axes[0], 'nll'), (axes[1], 'argmax_agreement')]:
        axis.scatter([full['kv_allocated_bytes']/1024], [full[field]], marker='*', s=130, color='black', label='full cache')
        axis.set_xlabel('Allocated KV tensors (KiB), excludes metadata/temporaries')
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel('Mean NLL (lower is better)')
    axes[1].set_ylabel('Argmax agreement with full (not target accuracy)')
    axes[1].set_ylim(-0.02, 1.05)
    figure.suptitle(f'KVForge budget experiment | {full["scored_tokens"]} scored tokens | '
                   f'{report["environment"]["device"]} FP32 | small-sample diagnostic')
    target = args.input.parent/'quality_memory.png'
    figure.savefig(target, dpi=160)
    plt.close(figure)
    print(f'Saved: {target}')


if __name__ == '__main__':
    main()
