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
    不传 cache 时计算完整序列;传 cache 时只计算新输入并复用历史 K/V
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

    def forward(self, x, cache=None, layer_index=0):
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
        # 假如历史已有 5 个 token，新输入的位置从 5 开始而不是再次从 0 开始
        start = 0 if cache is None else cache.next_position
        q, k = self.rope(q, start_pos=start), self.rope(k, start_pos=start)
        if cache is not None:
            k, v = cache.write(layer_index, k, v)
        # 新query位于[start,start+t)，完整缓存的key从0开始。
        # 预算淘汰后key位置可能不连续，以缓存提供的原始位置为准。
        # 广播形成[T,S]允许矩阵：key位置 <= query位置。
        q_positions = torch.arange(start, start + t, device=x.device)
        # 预算淘汰后，存储槽可能对应[0,1,7,8]，不能把它们重新编号为[0,1,2,3]。
        k_positions = (torch.arange(k.size(2), device=x.device) if cache is None
                       else cache.key_positions(k.size(2), x.device))
        allowed = k_positions[None, :] <= q_positions[:, None]
        # 显式扩展 KV 头，方便观察共享关系(默认 groups=8//2=4)
        # 第 0 个 KV 头重复 4 次，第 1 个 KV 头再重复 4 次
        # repeat_interleave 与普通 repeat 的排列顺序不同，不能随意替换
        # 这一步仍会扩展计算用的临时张量；常驻缓存已在上方保存了
        # 扩展前的紧凑 K/V(K 已经过 RoPE)，形状 [B,Hkv,T,D]
        groups = c.n_heads // c.n_kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        if c.attention_backend == "sdpa":
            # SDPA 内部包含 1/sqrt(D) 缩放，不能在调用前再次缩放 Q
            # dropout_p 显式设为 0，便于与手写分支对照
            # 调用 SDPA 不保证采用 FlashAttention,具体后端取决于设备、dtype 等条件
            # 显式mask处理带历史的矩形注意力，因果约束不依赖is_causal开关。
            # SDPA 的 attn_mask 中 True 表示允许关注,所以传入的是 allowed，不是 future
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, attn_mask=allowed, is_causal=False)
        else:
            # T 是新输入长度，S 是历史加新输入的长度
            # [B,Hq,T,D] @ [B,Hq,D,S] -> [B,Hq,T,S]
            # 行是 query 位置，列是 key 位置，缩放避免点积随 D 增大而过大
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(c.head_dim)
            # 取反后 True 表示需要遮住的未来位置
            future = ~allowed
            # masked_fill 的规则是:条件为 True 的位置替换成指定数值
            # [T,S] mask 广播到所有 batch/head，-inf 经 softmax 后变为 0
            # 把未来位置设为负无穷后，这些位置的概率就变成 0，不再参与 V 的加权汇总
            scores = scores.masked_fill(future, float("-inf"))
            # 沿 key 轴归一化，每行概率和为 1,FP32 softmax 后恢复 dtype
            weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
            # 按注意力概率加权汇总 V [B,Hq,T,S] @ [B,Hq,S,D] -> [B,Hq,T,D]
            out = weights @ v
        # 换回 [B,T,Hq,D],transpose 后内存通常不连续，contiguous 后才能安全地 view 并合并 Hq/D 轴 成 C
        # 最后应用可学习的输出投影
        return self.out_proj(out.transpose(1, 2).contiguous().view(b, t, c.dim))
