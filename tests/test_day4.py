"""
只检查评测逻辑和文件内容，不断言缓存必须更快（速度取决于机器）
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from kvforge import ModelConfig, MiniLLM
from kvforge.benchmarking import benchmark_pair, run_trial, summarize

ROOT = Path(__file__).resolve().parents[1]


class Day4Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.model = MiniLLM(ModelConfig(vocab_size=16, dim=16, n_layers=1,
            n_heads=2, n_kv_heads=1, hidden_dim=32, max_seq_len=8))
        self.ids = torch.randint(0, 16, (2, 7))

    def test_pair_schema_correctness_and_mode(self):
        """
        原始样本数量、缓存存储和 CPU 未测 GPU 指标必须如实报告
        """
        result = benchmark_pair(self.model, self.ids, 4, 3, warmup=0, repeats=2)
        self.assertTrue(self.model.training)
        for mode in ('cache', 'no_cache'):
            self.assertEqual(len(result[mode]['samples']), 2)
            self.assertGreater(result[mode]['decode_tokens_per_second'], 0)
            self.assertIsNone(result[mode]['cuda_peak_allocated_bytes'])
        self.assertEqual(result['cache']['cache_allocated_bytes'], 2*1*2*1*7*8*4)
        self.assertEqual(result['no_cache']['cache_allocated_bytes'], 0)
        self.assertLess(result['correctness']['max_abs_error'], 1e-5)

    def test_same_growing_prefix_workload(self):
        """
        基线不能偷换成固定短输入；缓存不能在每轮继续用上次的历史
        """
        self.model.eval()
        for enabled, lengths in [(False, [4, 5, 6, 7]), (True, [4, 1, 1, 1])]:
            for _ in range(2):
                with torch.no_grad(), patch.object(self.model, 'forward', wraps=self.model.forward) as forward:
                    run_trial(self.model, self.ids, 4, 3, enabled)
                self.assertEqual([call.args[0].size(1) for call in forward.call_args_list], lengths)

    def test_reject_overflow_and_invalid_repeats(self):
        """
        超过容量必须失败，不能转成窗口重建后仍称作增量 benchmark
        """
        with self.assertRaises(ValueError):
            benchmark_pair(self.model, torch.ones(2, 9, dtype=torch.long), 6, 3)
        with self.assertRaises(ValueError):
            benchmark_pair(self.model, self.ids, 4, 3, repeats=0)

    def test_metric_denominators(self):
        """
        batch=2、3步、总耗时6秒：吞吐1 token/s，批次每步2秒
        """
        samples = [{'prefill_seconds': 1, 'decode_seconds': 6, 'cache_allocated_bytes': 0,
                    'cuda_peak_allocated_bytes': None, 'cuda_peak_extra_bytes': None}]
        result = summarize(samples, batch_size=2, decode_steps=3)
        self.assertEqual(result['decode_tokens_per_second'], 1)
        self.assertEqual(result['decode_step_ms'], 2000)

    def test_cli_exports_and_checkpoint_window_guard(self):
        """
        真实运行入口，检查报告
        checkpoint 模式不允许越过原训练窗口
        """
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'report'
            command = [sys.executable, 'benchmark.py', '--contexts', '4', '--decode-steps', '2',
                       '--dim', '16', '--layers', '1', '--heads', '2', '--kv-heads', '1',
                       '--vocab-size', '16', '--backends', 'naive', '--warmup', '0',
                       '--repeats', '1', '--out', str(out)]
            subprocess.run(command, cwd=ROOT, capture_output=True, check=True)
            report = json.loads((out / 'results.json').read_text(encoding='utf-8'))
            self.assertEqual(len(report['results']), 1)
            self.assertTrue((out / 'summary.csv').is_file())
            # 非空目录不能被另一次运行覆盖
            repeated = subprocess.run(command, cwd=ROOT, capture_output=True)
            self.assertNotEqual(repeated.returncode, 0)
            checkpoint = Path(folder) / 'model.pt'
            from dataclasses import asdict
            torch.save({'model_config': asdict(self.model.config),
                        'model': self.model.state_dict()}, checkpoint)
            guarded = subprocess.run([sys.executable, 'benchmark.py', '--checkpoint', str(checkpoint),
                '--contexts', '7', '--decode-steps', '2', '--out', str(Path(folder)/'overflow')],
                cwd=ROOT, capture_output=True)
            self.assertNotEqual(guarded.returncode, 0)
            self.assertFalse((Path(folder)/'overflow/results.json').exists())


if __name__ == '__main__':
    unittest.main()
