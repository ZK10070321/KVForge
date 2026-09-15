"""Day 5入口：用已有checkpoint的真实验证文本做缓存预算实验。"""
import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
from kvforge import ModelConfig, MiniLLM
from kvforge.data import load_data
from kvforge.budget_evaluation import make_windows, evaluate_policy, compare_with_full


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('runs/day2_run_check/last.pt'))
    p.add_argument('--data', type=Path, default=Path('data/prepared'))
    p.add_argument('--budgets', type=int, nargs='+', default=[8, 16, 24])
    p.add_argument('--sink-size', type=int, default=2)
    p.add_argument('--window', type=int, default=None)
    p.add_argument('--score-from', type=int, default=None)
    p.add_argument('--max-windows', type=int, default=0)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--backend', choices=['naive', 'sdpa'], default=None)
    p.add_argument('--out', type=Path, default=Path('runs/day5_budget'))
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        p.error('输出目录非空，请选择新的--out，保留原始实验')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA不可用，请先在CPU完成实验')
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    _, val, tokenizer, fingerprint = load_data(args.data)
    if checkpoint['data_fingerprint'] != fingerprint or checkpoint['chars'] != tokenizer.chars:
        p.error('数据或词表与checkpoint不同，请使用原训练对应的数据目录')
    config_values = dict(checkpoint['model_config'])
    if args.backend:
        config_values['attention_backend'] = args.backend
    config = ModelConfig(**config_values)
    window = config.max_seq_len if args.window is None else args.window
    budgets = sorted(set(args.budgets))
    if not 1 <= window <= config.max_seq_len or not budgets or min(budgets) < 1 or max(budgets) > window:
        p.error('预算与window必须在模型窗口范围内')
    if not 0 <= args.sink_size < min(budgets):
        p.error('sink-size必须非负，且小于最小预算')
    # 默认只打分所有预算都已进入淘汰阶段的共同后缀；不给小预算额外挑容易样本。
    score_from = min(max(budgets), window-1) if args.score_from is None else args.score_from
    if not 0 <= score_from < window or args.max_windows < 0:
        p.error('score-from或max-windows不合法')
    windows = make_windows(val, window, args.max_windows)
    model = MiniLLM(config).to(args.device).eval()
    model.load_state_dict(checkpoint['model'])
    full = evaluate_policy(model, windows, score_from=score_from)
    rows = [compare_with_full(full, full)]
    for budget in budgets:
        for policy in ('recent', 'sink_recent'):
            result = evaluate_policy(model, windows, policy, budget,
                args.sink_size if policy == 'sink_recent' else 0, score_from)
            rows.append(compare_with_full(result, full))
    report = {
        'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'environment': {'torch': torch.__version__, 'python': platform.python_version(),
                        'device': args.device, 'dtype': 'fp32', 'threads': 2},
        'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'data_fingerprint': fingerprint, 'model_config': config_values,
        'window': window, 'window_stride': window, 'score_from': score_from,
        'validation_tokens': len(val), 'validation_unknown_tokens': int((val == 0).sum()),
        'evaluated_windows': windows.size(0),
        'scored_unknown_tokens': int((windows[:, score_from+1:] == 0).sum()),
        'method': 'fixed true validation tokens; score common suffix; reset each window; absolute positions; capacity includes current token',
        'limitations': 'small checkpoint/sample; KV tensor bytes exclude temporary tensors and metadata; no long-context or quality superiority claim',
        'results': rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    fields = ['policy', 'capacity', 'sink_size', 'scored_tokens', 'nll', 'ppl', 'delta_nll',
              'argmax_agreement', 'kv_allocated_bytes', 'position_metadata_bytes', 'kv_saving_fraction']
    with (args.out/'summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f'windows={len(windows)}, scored_tokens={full["scored_tokens"]}, window={window}, score_from={score_from}')
    for row in rows:
        ppl_text = f'{row["ppl"]:.4f}' if row['ppl'] is not None else 'overflow'
        print(f'{row["policy"]:11s} budget={row["capacity"]:3d} NLL={row["nll"]:.6f} '
              f'PPL={ppl_text} delta={row["delta_nll"]:+.6f} '
              f'KV={row["kv_allocated_bytes"]} bytes')
    print(f'Saved: {args.out / "results.json"}')


if __name__ == '__main__':
    main()
