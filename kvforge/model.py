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
    Pre-Norm(前向传播) Block:归一化在子层之前，两个子层分别带残差连接
    """
    def __init__(self, config):
        super().__init__()
        # attention 和 FFN 使用独立的 RMSNorm 缩放参数
        # 数据流经历的计算:输入->RMSNorm->Attention->RMSNorm->SwiGLU->输出
        self.attn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn = SwiGLU(config.dim, config.hidden_dim)

    def forward(self, x):
        # 残差主分支直接传递 x，注意力分支学习需要添加的跨位置信息
        # 所有中间子层输出均为 [B,T,C]，才能与主分支逐元素相加
        x = x + self.attention(self.attn_norm(x))
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
        因为 _init_weights 不需要访问当前 MiniLLM 对象，只需要检查传入的 module ，所以写成静态辅助函数，
        无需 self
        """
        # 如果 module 是 Linear 或 Embedding，就执行下面的初始化
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # 把权重初始化为服从正态分布的随机数，均值 mean=0.0，标准差 std=0.02
            # 为什么不全初始化为 0:如果同一层的神经元初始状态完全相同，会得到相同的梯度，难以学习不同功能，随机初始化可以打破这种对称性
            # 函数名末尾的 "_" 表示 "直接修改传入张量本身"
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        # 先检查二维形状，Python 的 or 短路求值能避免对低维输入访问 size(1)
        if input_ids.ndim != 2 or not 0 < input_ids.size(1) <= self.config.max_seq_len:
            raise ValueError("input_ids must be [B, T], with 1 <= T <= max_seq_len")
        # 本接口不在内部移位，因此 targets 必须与 input_ids 等长
        # 等长原因:原始 token 为 [a,b,c,d],切片后 input_ids = [a,b,c],targets = [b,c,d],两者长度都为 3
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must match input_ids shape and be shifted by caller")
        x = self.embedding(input_ids)
        # 每层输出保持 [B,T,C]，但其中编码的信息会逐层更新
        for block in self.blocks:
            x = block(x)
        # [B,T,C] -> [B,T,V]，logits 尚未经过 softmax，不是概率而是对词表第 v 个 token 给出的预测分数
        logits = self.lm_head(self.norm(x))
        # 不提供 targets 时只计算分数，方便后续推理，当前未封装生成循环
        loss = None
        if targets is not None:
            # 调用者传 tokens[:,:-1] 和 tokens[:,1:]，这里不能再次移位
            # 交叉熵 F.cross_entropy 的作用: 1.把 logits 转换成概率意义上的结果;
            #                              2.检查正确类别的预测情况;
            #                              3.正确类别概率越高，loss 越小;正确类别概率越低，loss 越大
            # cross_entropy 内部包含 log_softmax 进行了数值更稳定的计算，不能先手动 softmax
            # reshape 的作用:模型输出"logits:[B,T,V]、targets:[B,T]",但
            #               交叉熵希望看到"预测:[样本数,类别数]、目标:样本数]",所以合并 Batch 和
            #               时间轴 T，[B*T,V] 对 [B*T]，每个位置都可看成一条分类样本,代码中的"-1"
            #               表示"这一维由 PyTorch 根据总元素数量自动推断",例如 logits 原形状[2,3,4096]
            #               总共有 6 个 token 位置,所以 reshape(-1,4096) → [6,4096],targets:[2,3] → [6],
            #               于是可以把 6 个位置一起计算分类损失
            # .float() 的作用:未来可能使用 FP16 或 BF16 训练来节省显存,计算交叉熵时转成 FP32 有助于提高数值稳定性
            #                故此处把 logits 转 FP32 计算损失，默认对有效目标取平均
            loss = F.cross_entropy(logits.float().reshape(-1, self.config.vocab_size), targets.reshape(-1))
        # loss.shape = [], 综合表示当前模型在这一批所有 token 位置上预测得有多差,训练的目标就是不断降低这个损失
        return logits, loss










