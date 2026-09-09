"""
基础层：归一化、旋转位置编码、门控前馈网络
"""
import torch
from torch import nn
from torch.nn import functional as F

class RMSNorm(nn.Module):
    """
    沿最后一维做均方根归一化(输入输出 shape 相同)
    y = x * (mean(x²) + eps)^(-1/2) * weight
    与 LayerNorm 不同，此处不减去均值，没有可学习的偏置
    """
    def __init__(self, dim, eps=1e-6):
        # 必须先初始化 nn.Module,之后才能注册参数和子模块
        super().__init__()
        # Parameter 会被 model.parameters() 找到并交给优化器，初始缩放为 1
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # 转 FP32 计算平方/均值，降低 FP16/BF16 下的数值风险
        xf = x.float()
        # [B.T.C] -> 均方值 [B,T,1]
        # rsqrt(z) = 1/sqrt(z)，每个 token 独立归一化，不混合不同位置
        normalized = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + self.eps)
        # 恢复输入 dtype，再乘可学习的 C 维缩放，转换 dtype 不会切断梯度
        return normalized.to(x.dtype) * self.weight.to(x.dtype)

class RotaryEmbedding(nn.Module):
    """
    RoPE：把位置编码为 Q/K 向量的旋转，输入输出均为 [B,H,T,D]
    本实现用相邻通道配对，无可学习训练参数，角度在每次前向时根据 head_dim 和 theta 计算
    start_pos 为未来增量解码预留，本类自身不保存任何 KV Cache
    """
    def __init__(self, head_dim, theta=10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta

    def forward(self, x, start_pos=0):
        # 对通道 (0,1)、(2,3)...分别做二维旋转，要求 D 为偶数
        if start_pos < 0:
            raise ValueError("start_pos must be non-negative")
        # 实际训练 LLM 使用 FP16、BF16 会降低显存提高速度，但低精度计算可能产生明显的数值误差
        # 故即使外层开启了自动混合精度，这段角度和旋转计算仍使用 FP32 以提高数值稳定性
        with torch.autocast(device_type=x.device.type, enabled=False):
            # arange 生成 [0,2，。。。，D-2],频率为 theta^(-2i/D)，形状 [D/2]
            # 实现公式 freq_i = theta ** (-2i/D),theta 控制旋转频率的尺度
            freq = self.theta ** (-torch.arange(0, self.head_dim, 2, device=x.device,
                                                dtype=torch.float32) / self.head_dim)
            # x.size(-2) 是 T，位置是绝对索引 [start_pos,...,start_pos+T-1]
            pos = torch.arange(start_pos, start_pos + x.size(-2), device=x.device, dtype=torch.float32)
            # [T,1] * [1,D/2] 广播为 [T,D/2]，每个位置 * 每个二维通道对，都有自己的旋转角度
            angles = pos[:, None] * freq[None, :]
            cos, sin = angles.cos(), angles.sin()
            # ... 保留前面的 B/H/T 轴，0::2、1::2 取偶数/奇数通道
            even, odd = x.float()[..., 0::2], x.float()[..., 1::2]
            # cos/sin 自动广播到 B/H 轴，stack 后为 [B,H,T,D/2,2]
            # 使用二维旋转矩阵 [cos,-sin;sin,cos],理论上保持向量长度不变
            rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        # 合并最后两轴，把 [D/2,2] 合并回 [D]，恢复交错顺序 [even0,odd0,even1,odd1,...]
        # to(x.dtype) 原因是原输入可能是 BF16 或 FP16，故要把 FP32 重新转回原来的 dtype
        return rotated.flatten(-2).to(x.dtype)

class SwiGLU(nn.Module):
    """
    逐个 token 的门控前馈层：[B,T,C] -> [B,T,hidden_dim] -> [B,T,C]
    不混合不同 token，跨位置的信息交换由 attention 完成
    SwiGLU(x) = W_down( SiLU(W_gate(x)) * W_up(x))
    普通 FFN 只有一条中间特征流；SwiGLU 多出了一条门控分支，让模型能够动态调节中间特征
    """
    def __init__(self, dim, hidden_dim):
        super().__init__()
        # gate 和 up 是两个独立线性变换，down 把中间特征投影回残差宽度 C
        self.gate = nn.Linear(dim, hidden_dim,bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        # SiLu(z)=z*sigmoid(z) [*是对应元素相乘，不是矩阵乘法]
        # 门控分支调制 up 分支的特征，最后一层输出必须回到 C 维
        return self.down(F.silu(self.gate(x)) * self.up(x))













