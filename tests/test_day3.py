"""Day 3：验证缓存是否算得对，而不只是验证程序能运行。"""
import unittest
import torch
from kvforge import ModelConfig, MiniLLM
from generate import generate


class Day3Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(123)

    def make_model(self, backend='naive', kv_heads=2):
        # 用小模型快速检查数学逻辑；两层能够暴露跨层缓存混用的问题。
        return MiniLLM(ModelConfig(vocab_size=35, dim=32, n_layers=2,
            n_heads=4, n_kv_heads=kv_heads, hidden_dim=64,
            max_seq_len=16, attention_backend=backend)).eval()

    @torch.no_grad()
    def test_full_matches_token_and_chunk_cache(self):
        """MHA/GQA/MQA、两种后端：缓存后的每个位置都应接近完整前向。"""
        ids = torch.randint(1, 35, (2, 12))
        for backend in ('naive', 'sdpa'):
            for heads in (1, 2, 4):
                for chunks in ([1] * 12, [5, 3, 4]):
                    with self.subTest(backend=backend, heads=heads, chunks=chunks):
                        model = self.make_model(backend, heads)
                        expected, _ = model(ids)
                        cache = model.create_cache(2)
                        outputs = []
                        start = 0
                        for size in chunks:
                            actual, _ = model(ids[:, start:start+size], cache=cache)
                            outputs.append(actual)
                            start += size
                        torch.testing.assert_close(torch.cat(outputs, dim=1), expected,
                                                   atol=1e-6, rtol=1e-5)
                        self.assertEqual(cache.length, 12)

    @torch.no_grad()
    def test_chunk_cannot_see_future(self):
        """已有历史时，新块内部也必须遮住未来，不能让整个新块互相可见。"""
        for backend in ('naive', 'sdpa'):
            model = self.make_model(backend)
            ids = torch.randint(1, 35, (1, 8))
            altered = ids.clone()
            altered[:, 6:] = (altered[:, 6:] % 34) + 1
            results = []
            for sequence in (ids, altered):
                cache = model.create_cache(1)
                model(sequence[:, :3], cache=cache)
                logits, _ = model(sequence[:, 3:], cache=cache)
                results.append(logits[:, :3])
            torch.testing.assert_close(*results, atol=1e-6, rtol=1e-5)

    @torch.no_grad()
    def test_compact_storage_reset_and_capacity(self):
        """常驻缓存按 KV 头保存；重置后可复用空间，越界不会推进长度。"""
        model = self.make_model()
        cache = model.create_cache(2, capacity=8)
        # 2(K和V) × 2层 × 2样本 × 2个KV头 × 8位置 × 8头维度 × 4字节。
        self.assertEqual(cache.allocated_bytes, 2*2*2*2*8*8*4)
        self.assertEqual(tuple(cache.keys[0].shape), (2, 2, 8, 8))
        model(torch.ones(2, 8, dtype=torch.long), cache=cache)
        with self.assertRaises(ValueError):
            model(torch.ones(2, 1, dtype=torch.long), cache=cache)
        self.assertEqual(cache.length, 8)
        cache.reset()
        ids = torch.full((2, 3), 2, dtype=torch.long)
        torch.testing.assert_close(model(ids, cache=cache)[0], model(ids)[0])
        self.assertEqual(cache.length, 3)

    def test_invalid_usage_and_weights_unchanged(self):
        """训练、不同模型、不同 batch 不应悄悄使用错误缓存。"""
        model = self.make_model()
        names = set(model.state_dict())
        cache = model.create_cache(1)
        ids = torch.ones(1, 2, dtype=torch.long)
        with self.assertRaises(ValueError):
            model(ids, cache=cache)  # 没有禁用梯度。
        with torch.no_grad():
            model.train()
            with self.assertRaises(ValueError):
                model(ids, cache=cache)
            model.eval()
            with self.assertRaises(ValueError):
                self.make_model()(ids, cache=cache)
            with self.assertRaises(ValueError):
                model(ids.expand(2, -1), cache=cache)
            with self.assertRaises(ValueError):
                model(ids, ids, cache=cache)
        self.assertEqual(cache.length, 0)
        self.assertEqual(names, set(model.state_dict()))

    def test_generation_matches_and_restores_mode(self):
        """贪心结果一致；覆盖未满窗口、越过窗口、长提示词和零生成。"""
        for backend in ('naive', 'sdpa'):
            model = self.make_model(backend).train()
            for prompt_length, new_tokens in ((4, 6), (14, 7), (20, 4), (4, 0)):
                ids = torch.randint(1, 35, (2, prompt_length))
                expected = generate(model, ids, new_tokens, temperature=0)
                actual = generate(model, ids, new_tokens, temperature=0, use_cache=True)
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(model.training)
                self.assertTrue(torch.equal(actual[:, :prompt_length], ids))


if __name__ == '__main__':
    unittest.main()
