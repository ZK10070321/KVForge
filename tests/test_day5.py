"""Day 5：用独立的稀疏因果全序列参考验证淘汰后的计算，而不只检查队列长度。"""
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F
from kvforge import MiniLLM, ModelConfig
from kvforge.data import prepare, load_data
from kvforge.budget_evaluation import make_windows, evaluate_policy, compare_with_full

ROOT = Path(__file__).resolve().parents[1]


class Day5Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(15)
        self.config = ModelConfig(vocab_size=17, dim=32, n_layers=2, n_heads=4,
                                  n_kv_heads=2, hidden_dim=64, max_seq_len=12)

    @torch.no_grad()
    def test_retained_positions_and_storage(self):
        model = MiniLLM(self.config).eval()
        for policy, sink, expected in [('recent', 0, [5, 6, 7, 8]),
                                       ('sink_recent', 1, [0, 6, 7, 8])]:
            cache = model.create_budget_cache(2, 4, policy, sink)
            for i in range(9):
                model(torch.full((2, 1), i, dtype=torch.long), cache=cache)
                self.assertLessEqual(cache.length, 4)
            self.assertEqual(cache.positions[:cache.length].tolist(), expected)
            self.assertEqual(cache.next_position, 9)
            self.assertEqual(cache.allocated_bytes, 2*2*2*2*4*8*4)
            self.assertEqual(cache.metadata_bytes, 4*8)

    @torch.no_grad()
    def test_budget_matches_independent_dense_mask(self):
        """参考模型一次处理整段，但每行只准看策略规定的位置；不调用预算缓存逻辑。"""
        ids = torch.randint(0, 17, (2, 9))
        original_sdpa = F.scaled_dot_product_attention
        for backend in ('naive', 'sdpa'):
            for heads in (1, 2, 4):
                for capacity, sink in [(1, 0), (4, 0), (4, 1)]:
                    with self.subTest(backend=backend, heads=heads, capacity=capacity, sink=sink):
                        config = replace(self.config, attention_backend=backend, n_kv_heads=heads)
                        model = MiniLLM(config).eval()
                        reference = MiniLLM(replace(config, attention_backend='sdpa')).eval()
                        reference.load_state_dict(model.state_dict())
                        mask = torch.zeros(9, 9, dtype=torch.bool)
                        for query in range(9):
                            # 前缀未满时看全部；满后按原始位置直接构造，不复用cache的选择代码。
                            if query+1 <= capacity:
                                allowed = list(range(query+1))
                            else:
                                allowed = list(range(sink)) + list(range(query-(capacity-sink)+1, query+1))
                            mask[query, allowed] = True
                        def sparse_sdpa(q, k, v, **kwargs):
                            kwargs['attn_mask'] = mask
                            return original_sdpa(q, k, v, **kwargs)
                        with patch('torch.nn.functional.scaled_dot_product_attention', side_effect=sparse_sdpa):
                            expected = reference(ids)[0]
                        cache = model.create_budget_cache(2, capacity, 'sink_recent' if sink else 'recent', sink)
                        actual = torch.cat([model(ids[:, i:i+1], cache=cache)[0] for i in range(9)], dim=1)
                        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    @torch.no_grad()
    def test_no_eviction_matches_full(self):
        ids = torch.randint(0, 17, (2, 9))
        for backend in ('naive', 'sdpa'):
            model = MiniLLM(replace(self.config, attention_backend=backend)).eval()
            for policy, sink in [('recent', 0), ('sink_recent', 2)]:
                cache = model.create_budget_cache(2, 9, policy, sink)
                actual = torch.cat([model(ids[:, i:i+1], cache=cache)[0] for i in range(9)], dim=1)
                torch.testing.assert_close(actual, model(ids)[0], atol=1e-6, rtol=1e-5)

    def test_invalid_usage_and_failed_write_reset(self):
        model = MiniLLM(self.config).eval()
        ids = torch.ones(1, 1, dtype=torch.long)
        cache = model.create_budget_cache(1, 4)
        with self.assertRaises(ValueError):
            model(ids, cache=cache)  # 梯度仍开启。
        with self.assertRaises(ValueError):
            model.create_budget_cache(1, 4, 'sink_recent', 4)
        with torch.no_grad():
            with self.assertRaises(ValueError):
                model(ids.expand(1, 2), cache=cache)
            with self.assertRaises(ValueError):
                MiniLLM(self.config).eval()(ids, cache=cache)
            # 故意在第一层写入后让第二层失败，不能悄悄复用半次写入的历史。
            with patch.object(model.blocks[1], 'forward', side_effect=RuntimeError('injected failure')):
                with self.assertRaises(RuntimeError):
                    model(ids, cache=cache)
            with self.assertRaises(ValueError):
                model(ids, cache=cache)
            cache.reset()
            torch.testing.assert_close(model(ids, cache=cache)[0], model(ids)[0])

    @torch.no_grad()
    def test_absolute_position_limit(self):
        model = MiniLLM(self.config).eval()
        cache = model.create_budget_cache(1, 2)
        ids = torch.ones(1, 1, dtype=torch.long)
        for _ in range(self.config.max_seq_len):
            model(ids, cache=cache)
        with self.assertRaises(ValueError):
            model(ids, cache=cache)
        self.assertEqual(cache.next_position, self.config.max_seq_len)

    def test_quality_labels_and_common_suffix(self):
        model = MiniLLM(self.config)
        tokens = torch.arange(17).repeat(2)
        windows = make_windows(tokens, 8, max_windows=2)
        full = evaluate_policy(model, windows, score_from=4)
        self.assertTrue(model.training)
        self.assertEqual(full['scored_tokens'], 8)
        with torch.no_grad():
            logits = model(windows[:, :-1])[0][:, 4:]
            expected = F.cross_entropy(logits.reshape(-1, 17), windows[:, 5:].reshape(-1)).item()
        self.assertAlmostEqual(full['nll'], expected, places=6)
        same = evaluate_policy(model, windows, 'recent', 8, score_from=4)
        row = compare_with_full(same, full)
        self.assertAlmostEqual(row['delta_nll'], 0, places=6)
        self.assertEqual(row['argmax_agreement'], 1)
        with self.assertRaises(ValueError):
            make_windows(torch.arange(8), 8)

    def test_cli_report_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            text = root/'text.txt'
            text.write_text('abcde '*50, encoding='utf-8')
            prepare(text, root/'data', val_fraction=0.3)
            _, _, tokenizer, fingerprint = load_data(root/'data')
            config = replace(self.config, vocab_size=tokenizer.vocab_size)
            checkpoint = root/'model.pt'
            torch.save({'model': MiniLLM(config).state_dict(), 'model_config': asdict(config),
                        'chars': tokenizer.chars, 'data_fingerprint': fingerprint}, checkpoint)
            command = [sys.executable, 'evaluate_budget.py', '--checkpoint', str(checkpoint),
                '--data', str(root/'data'), '--window', '8', '--score-from', '4', '--budgets', '2', '4',
                '--sink-size', '1', '--max-windows', '2', '--out', str(root/'result')]
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
            report = json.loads((root/'result/results.json').read_text(encoding='utf-8'))
            self.assertEqual(len(report['results']), 5)
            self.assertTrue(all(r['scored_tokens'] == 8 for r in report['results']))
            self.assertTrue((root/'result/summary.csv').is_file())
            self.assertNotEqual(subprocess.run(command, cwd=ROOT, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
