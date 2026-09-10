"""Day 2 回归测试：真实调用 CLI，检查断点恢复、数据隔离和生成。"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import torch
from kvforge import MiniLLM, ModelConfig
from kvforge.data import CharTokenizer, prepare, load_data, get_batch
from generate import generate

ROOT = Path(__file__).resolve().parents[1]


class Day2Tests(unittest.TestCase):
    def test_tokenizer_roundtrip(self):
        """中文、空格、换行也按字符编码；未知字符固定为 0。"""
        text = '你好 world\n'
        tokenizer = CharTokenizer.fit(text)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertEqual(tokenizer.encode('★'), [0])

    def test_split_and_shift(self):
        """验证集独有字符不会偷偷进入词表；目标恰好错开一位。"""
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source.txt'
            source.write_text('a' * 90 + 'z' * 10, encoding='utf-8')
            prepare(source, folder)
            train, val, tokenizer, _ = load_data(folder)
            self.assertNotIn('z', tokenizer.stoi)
            self.assertTrue((val == 0).all())
            sequence = torch.arange(20)
            x, y = get_batch(sequence, 2, 5, torch.Generator().manual_seed(1), 'cpu')
            torch.testing.assert_close(y, x + 1)
            with self.assertRaises(ValueError):
                get_batch(train, 1, 100, torch.Generator(), 'cpu')

    def test_generation_boundaries(self):
        """生成可以跨越上下文上限；模型会裁剪输入，但返回完整 token 序列。"""
        model = MiniLLM(ModelConfig(vocab_size=8, dim=16, n_layers=1, n_heads=2,
                                    n_kv_heads=1, hidden_dim=32, max_seq_len=4))
        ids = torch.tensor([[1, 2, 3]])
        output = generate(model, ids, 6, temperature=0)
        self.assertEqual(output.shape, (1, 9))
        torch.testing.assert_close(output[:, :3], ids)
        self.assertTrue(model.training)
        self.assertTrue((output[:, 3:] != 0).all())

    def test_checkpoint_resume_matches_uninterrupted(self):
        """同 CPU 环境，8 步连续训练应等于 4 步保存后恢复至 8 步。"""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source.txt'
            source.write_text('The cat reads a book. The dog plays a game.\n' * 40, encoding='utf-8')
            prepare(source, root / 'data')
            base = [sys.executable, str(ROOT / 'train.py'), '--data', str(root / 'data'),
                    '--steps', '8', '--warmup', '2', '--seq-len', '8', '--batch-size', '2',
                    '--dim', '16', '--layers', '1', '--heads', '2', '--kv-heads', '1',
                    '--hidden-dim', '32', '--eval-every', '4', '--eval-batches', '1']
            def run(arguments):
                result = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            run(base + ['--out', str(root / 'full')])
            run(base + ['--out', str(root / 'split'), '--stop-after', '4'])
            run(base + ['--out', str(root / 'split'), '--resume', str(root / 'split/last.pt')])
            full = torch.load(root / 'full/last.pt', weights_only=True)
            resumed = torch.load(root / 'split/last.pt', weights_only=True)
            self.assertEqual(resumed['step'], 8)
            for name in full['model']:
                torch.testing.assert_close(full['model'][name], resumed['model'][name], rtol=0, atol=0)
            run([sys.executable, str(ROOT / 'generate.py'), '--checkpoint', str(root / 'split/last.pt'),
                 '--prompt', 'The ', '--max-new-tokens', '10'])


if __name__ == '__main__':
    unittest.main()
