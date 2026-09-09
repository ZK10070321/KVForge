"""
完整序列的因果自注意力
支持 MHA/GQA/MQA 与 naive/SDPA 两种计算路径
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .layers import RotaryEmbedding

class CausalSelfAttention(nn.Module):
    """
    输入输出 [B,T,C]，每个位置只能关注自己和过去的位置
    Hkv=Hq 是 MHA,每个 Q 头都有自己的 K/V；
    1<Hkv<Hq 是 GQA，一组 Q 头共享 K/V；
    HKV=1 是 MQA，所有 Q 头共享一组 K/V
    Day 1 不保存历史 K/V，每次调用都会重新计算整个输入序列
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Linear 只变换最后一轴，Q 输出 Hq*D=C，K/V 输出 Hkv*D
        # 默认 Q 投影为 256 -> 256,而 K/V 分别为 256 -> 64
        self.q_proj = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        # 合并各头后，通过输出投影混合他们的信息，保持残差所需的 C 维
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.rope = RotaryEmbedding(config.head_dim, config.rope_theta)

    def forward(self, x):
        """
        :param x: x 是隐藏状态，而不是 token ID
        :return: 返回与 x 同形状的注意力输出
        """
        b, t, _ = x.shape
        c = self.config
        # 先投影 [B,T,H*D]，再 view 为 [B,T,H,D]，最后换轴到 [B,H,T,D]
        # 头数轴在前，便于矩阵乘法在每个 batch / head 内独立完成
        q = self.q_proj(x).view(b, t, c.n_heads, c.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, c.n_kv_heads, c.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, c.n_kv_heads, c.head_dim).transpose(1, 2)
        # Q/K 旋转后，点积会感知位置，V 保留作为被聚合的内容向量
        q, k = self.rope(q), self.rope(k)
        # 显式扩展 KV 头，方便观察共享关系(默认 groups=8//2=4)
        # 第 0 个 KV 头重复 4 次，第 1 个 KV 头再重复 4 次
        # repeat_interleave 与普通 repeat 的排列顺序不同，不能随意替换
        # 这一步会扩展计算用的张量，不代表已实现显存优化,Day 3 缓存要保存
        # 扩展前的紧凑 K/V(K 已经过 RoPE)，形状 [B,Hkv,T,D]
        groups = c.n_heads // c.n_kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        if c.attention_backend == "sdpa":
            # SDPA 内部包含 1/sqrt(D) 缩放，不能在调用前再次缩放 Q
            # 当前 Q/K 长度相等，is_causal=True 正好表示下三角可见
            # dropout_p 显式设为 0，便于与手写分支对照
            # 调用 SDPA 不保证采用 FlashAttention,具体后端取决于设备、dtype 等条件
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0,is_causal=True)
        else:
            # [B,Hq,T,D] @ [B,Hq,D,T] -> [B,Hq,T,T]
            # 行是 query 位置，列是 key 位置；缩放避免点积随 D 增大而过大
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(c.head_dim)
            # triu(1) 只保留严格上三角的 True，即 key 位置大于 query 的未来
            # 对角线不遮挡,模型可看当前输入 token 来预测下一个 token
            future = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
            # [T,T] mask 广播到所有 batch/head，-inf 经 softmax 后变为 0
            scores = scores.masked_fill(future, float("-inf"))
            # 沿 key 轴归一化，每行概率和为 1,FP32 softmax 后恢复 dtype
            weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
            # 按注意力概率加权汇总 V：[B,Hq,T,T] @ [B,Hq,T,D]
            out = weights @ v
        # 换回 [B,T,Hq,D],transpose 后内存通常不连续，contiguous 后才能安全地 view 并合并 Hq/D 轴 成 C
        # 最后应用可学习的输出投影
        return self.out_proj(out.transpose(1, 2).contiguous().view(b, t, c.dim))












