"""Day 1 正确性回归测试：每项检查一个具体性质，而不只验证 shape。

运行：python -m unittest discover -s tests -v。
使用很小的配置，使 CPU 也能快速验证；未覆盖 CUDA 或混合精度。
"""

import unittest
from dataclasses import replace
import torch
from torch.nn import functional as F
from kvforge import ModelConfig, MiniLLM
from kvforge.layers import RMSNorm, RotaryEmbedding


class Day1Tests(unittest.TestCase):
    """unittest 自动发现 test_ 开头的方法；断言失败会报告对应测试名。"""

    def setUp(self):
        """每项测试之前重置随机种子与配置，减少测试之间的相互影响。"""
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.c = ModelConfig(vocab_size=32, dim=32, n_layers=2, n_heads=4,
                             n_kv_heads=2, hidden_dim=64, max_seq_len=16)

    def test_config_rejects_invalid_shapes(self):
        """分别检查拆头失败、KV 分组失败、奇数头维度、零层和未知后端。"""
        for kwargs in ({"dim": 31}, {"n_kv_heads": 3}, {"dim": 12},
                       {"n_layers": 0}, {"attention_backend": "bad"}):
            with self.assertRaises(ValueError):
                # replace 创建新 dataclass，并再次执行 __post_init__ 验证。
                replace(self.c, **kwargs)

    def test_rmsnorm_reference(self):
        """用直接除以平方根的公式对照实现；初始可学习缩放为 1。"""
        x = torch.randn(2, 3, 32)
        layer = RMSNorm(32)
        expected = x / torch.sqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        # 浮点运算可能存在舍入误差，用 assert_close 而非逐位相等判断。
        torch.testing.assert_close(layer(x), expected)

    def test_rope_rotation_and_offset(self):
        """检查零位置、保长性质、分段绝对位置和已知旋转角度。"""
        rope = RotaryEmbedding(8)
        x = torch.randn(2, 4, 6, 8)
        y = rope(x)
        # 位置 0 的角度为 0，所以应该保持输入不变。
        torch.testing.assert_close(y[:, :, 0], x[:, :, 0])
        # 正交旋转保持最后一维的平方和，不应改变向量长度。
        torch.testing.assert_close(y.square().sum(-1), x.square().sum(-1))
        # 只处理后半段时 start_pos=3，应与整段对应位置的旋转结果相同。
        torch.testing.assert_close(rope(x[:, :, 3:], start_pos=3), y[:, :, 3:])
        # 第一对通道的频率为 1，位置 1 旋转 1 弧度，独立验证符号与角度。
        angle = torch.tensor(1.0)
        expected = x[:, :, 1, 0] * angle.cos() - x[:, :, 1, 1] * angle.sin()
        torch.testing.assert_close(y[:, :, 1, 0], expected)

    def test_no_future_leakage(self):
        """修改后 4 个 token 后，前 4 个位置的 logits 必须保持不变。"""
        for backend in ("naive", "sdpa"):
            # eval 设置评估模式；no_grad 则关闭梯度记录，两者作用不同。
            m = MiniLLM(replace(self.c, attention_backend=backend)).eval()
            x = torch.randint(32, (2, 8))
            # clone 避免改到原输入；加 1 再取模确保每个后缀 token 都发生变化。
            changed = x.clone()
            changed[:, 4:] = (changed[:, 4:] + 1) % 32
            with torch.no_grad():
                a, _ = m(x)
                b, _ = m(changed)
            torch.testing.assert_close(a[:, :4], b[:, :4], atol=1e-6, rtol=1e-5)

    def test_naive_sdpa_outputs_and_gradients(self):
        """相同权重和输入下，两种注意力后端的前向和反向都应接近。"""
        for heads in (1, 2, 4):  # MQA, GQA, MHA
            a = MiniLLM(replace(self.c, n_kv_heads=heads))
            b = MiniLLM(replace(self.c, n_kv_heads=heads, attention_backend="sdpa"))
            # 必须复制权重，否则随机初始化不同会掩盖后端计算差异。
            b.load_state_dict(a.state_dict())
            x = torch.randint(32, (2, 9))
            ya, la = a(x[:, :-1], x[:, 1:])
            yb, lb = b(x[:, :-1], x[:, 1:])
            torch.testing.assert_close(ya, yb, atol=2e-6, rtol=2e-5)
            # 不仅看输出：错误的梯度也可能让模型在训练中悄悄出问题。
            la.backward()
            lb.backward()
            for pa, pb in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(pa.grad, pb.grad, atol=2e-6, rtol=2e-4)

    def test_gqa_matches_expanded_mha(self):
        """把 GQA 的 KV 投影权重按头复制到 MHA，输出应完全等价。

        这是人为构造共享权重的对照，不代表独立训练的 MHA/GQA 会相同。
        """
        gqa = MiniLLM(self.c)
        mha = MiniLLM(replace(self.c, n_kv_heads=4))
        # state_dict 包含命名参数；仅替换 K/V 投影，其余参数形状不变。
        state = gqa.state_dict()
        for key in list(state):
            if key.endswith(("k_proj.weight", "v_proj.weight")):
                # Linear 权重是 [out,in]=[16,32]，按 [Hkv,D,C] 拆开，
                # 每头复制两次，再合并为 MHA 的 [32,32] 权重矩阵。
                state[key] = state[key].view(2, 8, 32).repeat_interleave(2, dim=0).reshape(32, 32)
        mha.load_state_dict(state)
        x = torch.randint(32, (2, 8))
        torch.testing.assert_close(gqa(x)[0], mha(x)[0])

    def test_loss_shift_backward_and_weight_tying(self):
        """检查标签对齐、真正的参数共享、有限梯度以及优化器实际更新。"""
        model = MiniLLM(self.c)
        tokens = torch.randint(32, (2, 9))
        logits, loss = model(tokens[:, :-1], tokens[:, 1:])
        # 独立计算同一分类损失，确认模型内部没有偷偷再次移位。
        expected = F.cross_entropy(logits.reshape(-1, 32), tokens[:, 1:].reshape(-1))
        torch.testing.assert_close(loss, expected)
        # assertIs 检查对象身份；仅数值相等不能证明共享同一个参数。
        self.assertIs(model.embedding.weight, model.lm_head.weight)
        # detach 脱离计算图，clone 保留更新前快照，避免共享存储一起改变。
        before = model.embedding.weight.detach().clone()
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        opt.step()
        self.assertFalse(torch.equal(before, model.embedding.weight))

    def test_boundaries(self):
        """最短和最大合法序列可用；空序列和超长序列应明确报错。"""
        model = MiniLLM(self.c)
        for length in (1, 16):
            # ID 必须是整型，这里用全 0 token 验证形状，不测试文本含义。
            y, loss = model(torch.zeros(1, length, dtype=torch.long))
            self.assertEqual(y.shape, (1, length, 32))
            self.assertIsNone(loss)
        for length in (0, 17):
            with self.assertRaises(ValueError):
                model(torch.zeros(1, length, dtype=torch.long))

    def test_tiny_batch_overfit(self):
        """让模型记住一条固定短序列，检查训练链路确实能降低损失。

        这是可学习性检查，不是验证集指标，也不能说明泛化能力。
        """
        model = MiniLLM(self.c)
        tokens = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8]])
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        initial = model(tokens[:, :-1], tokens[:, 1:])[1].item()
        for _ in range(40):
            # PyTorch 默认累积梯度；每轮清空，避免把历史梯度误加进来。
            # set_to_none=True 直接置空梯度，下一次 backward 再创建。
            opt.zero_grad(set_to_none=True)
            _, loss = model(tokens[:, :-1], tokens[:, 1:])
            loss.backward()
            opt.step()
        # 使用更新后重新计算的损失；要求降到初始值的 30% 以下。
        final = model(tokens[:, :-1], tokens[:, 1:])[1].item()
        self.assertLess(final, initial * 0.3)


if __name__ == "__main__":
    # 支持直接作为脚本启动 unittest；推荐从项目根目录使用 discover 命令。
    unittest.main()
        # C=32，Hq=4，D=8，Hkv=2；词表和序列很小以节省测试时间。
