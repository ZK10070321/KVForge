"""
把基础层组装成 Decoder-only 语言模型，并计算下一 token 预测损失
"""

import torch
from torch import nn
from torch.nn import functional as F
from .attention import CausalSelfAttention
from .layers import RMSNorm, SwiGLU

class TransformerBlock(nn.Module):
    """
    Pre-Norm（先归一化） Block:归一化在子层之前，两个子层分别带残差连接
    """
    def __init__(self, config):
        super().__init__()
        # attention 和 FFN 使用独立的 RMSNorm 缩放参数
        # 数据流经历的计算:输入->RMSNorm->Attention->RMSNorm->SwiGLU->输出
        self.attn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn = SwiGLU(config.dim, config.hidden_dim)

    def forward(self, x, cache=None, layer_index=0):
        # 残差主分支直接传递 x，注意力分支学习需要添加的跨位置信息
        # 所有中间子层输出均为 [B,T,C]，才能与主分支逐元素相加
        x = x + self.attention(self.attn_norm(x), cache=cache, layer_index=layer_index)
        # 使用更新后的 x 进入 FFN，而不是最初的输入
        # FFN 逐位置处理特征
        return x + self.ffn(self.ffn_norm(x))

class MiniLLM(nn.Module):
    """
    小型因果语言模型，返回 (logits, loss)
    input_ids：整型 token ID，[B,T]，取值范围 [0,vocab_size)
    targets：可选的同形状目标，由调用者提前向后错开一个 token
    当前面向无 padding 的等长文本块，没有实现 padding attention mask
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Embedding 是 [V,C] 的查找表，输入 [B,T],输出 [B,T,C]
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        # ModuleList 注册所有子层到模型中，使参数能被优化器、to(device)、state_dict 找到
        # 普通 Python list 不会自动完成这种注册,列表推导式创建独立的层
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        # 最后一个 Block 之后再归一化，再映射为每个词的分数
        self.norm = RMSNorm(config.dim, config.norm_eps)
        # 对最后一维做变换，产生 logits 分数
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        # apply 递归访问模型的所有子模块并调用初始化函数，RMSNorm 保持其全 1 初值
        self.apply(self._init_weights)
        # 以下是 Weight tying 代码，让输入 token 使用的向量表和输出 token 使用的分类权重共享同一个 Parameter，不是复制数值
        # Embedding/Linear 的权重形状都是 [V,C],因此可以直接共享，可以减少参数量，且输入输出使用同一个 token 表示空间
        self.lm_head.weight = self.embedding.weight

    @staticmethod
    def _init_weights(module):
        """
        初始化线性层与词嵌入，RMSNorm保留默认的全1缩放
        """
        # 如果 module 是 Linear 或 Embedding，就执行下面的初始化
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # 把权重初始化为服从正态分布的随机数，均值 mean=0.0，标准差 std=0.02
            # 随机初始化打破神经元对称性
            # 函数名末尾的 "_" 表示 "直接修改传入张量本身"
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def create_cache(self, batch_size, capacity=None):
        # 缓存由调用者持有，不注册为模型参数，因此旧 checkpoint 仍能直接加载
        from .cache import KVCache
        return KVCache(self, batch_size, capacity)

    def create_budget_cache(self, batch_size, capacity, policy='recent', sink_size=0):
        # 显式创建预算缓存，不改变Day 3完整缓存的默认行为与旧checkpoint格式。
        from .budget_cache import BudgetKVCache
        return BudgetKVCache(self, batch_size, capacity, policy, sink_size)

    def forward(self, input_ids, targets=None, *, cache=None):
        # 先检查二维形状，Python 的 or 短路求值能避免对低维输入访问 size(1)
        if input_ids.ndim != 2 or not 0 < input_ids.size(1) <= self.config.max_seq_len:
            raise ValueError("input_ids must be [B, T], with 1 <= T <= max_seq_len")
        # 本接口不在内部移位，因此 targets 必须与 input_ids 等长
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must match input_ids shape and be shifted by caller")
        if cache is not None:
            if targets is not None:
                raise ValueError('KV Cache 只用于推理，不接收训练 targets')
            cache.validate(self, input_ids)
        x = self.embedding(input_ids)
        # 每层输出保持 [B,T,C]，但其中编码的信息会逐层更新
        # 模型把同一个缓存交给不同层，enumerate 同时提供 layer_index：层编号、block：这一层对象
        for layer_index, block in enumerate(self.blocks):
            x = block(x, cache=cache, layer_index=layer_index)
        # [B,T,C] -> [B,T,V]，logits 尚未经过 softmax，不是概率而是对词表第 v 个 token 给出的预测分数
        logits = self.lm_head(self.norm(x))
        # targets由调用者错位；推理时不计算loss
        loss = None
        if targets is not None:
            # 调用者传 tokens[:,:-1] 和 tokens[:,1:]，这里不能再次移位
            # logits [B,T,V]和targets [B,T]展平为交叉熵接口需要的形状
            # 内部已包含log_softmax；混合精度下先转FP32以提高数值稳定性
            loss = F.cross_entropy(logits.float().reshape(-1, self.config.vocab_size), targets.reshape(-1))
        # loss.shape = [], 综合表示当前模型在这一批所有 token 位置上预测得有多差,训练的目标就是不断降低这个损失
        if cache is not None:
            # 所有层共用同一个起始位置，必须等全部层成功后才增加 length
            # 各层处理同一批位置，不能随层数重复增加历史长度
            # 完整缓存中途失败时，length不变，重试覆盖未提交区域
            # 预算缓存可能已移动旧数据，失败后会标记dirty，必须reset并从窗口开头重算
            cache.commit(input_ids.size(1))
        return logits, loss
