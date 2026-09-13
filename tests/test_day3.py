"""
Day 3：验证缓存是否算得对，而不只是验证程序能运行
"""
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
        """
        比较一次完整输入、逐 token 输入、分成多个块输入，并覆盖 naive / SDPA 和 MHA / GQA / MQA
        它能发现位置偏移、头扩展、缓存写入和遮罩等问题
        """
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
        """
        保留前面的 token，修改后面的 token，早位置的输出发生明显变化，说明可能看到了未来
        """
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
        """
        检查缓存确实按 Hkv 保存、分配字节数符合公式、越界会报错、越界后有效长度不变、重置后新文本计算正确
        """
        model = self.make_model()
        cache = model.create_cache(2, capacity=8)
        # 2(K和V) × 2层 × 2样本 × 2个KV头 × 8位置 × 8头维度 × 4字节
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
        """
        检查开启梯度时拒绝缓存调用、训练模式下拒绝、其他模型不能使用该缓存、
        batch 不一致时报错、缓存调用不接收训练 targets、创建缓存不改变 state_dict 的键集合
        这个测试并不是完整验证所有可能的错误用法(例如不能据此声称它能自动识别同一模型对象的所有权重变更)
        """
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
        """
        检查缓存与无缓存的贪心结果一致、可以跨窗口生成、长提示词可按窗口处理、
        生成0个token时保持输入、保留提示词前缀、返回后恢复原模型模式
        """
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
