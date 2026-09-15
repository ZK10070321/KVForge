"""
集中管理模型超参数，创建模型前就检查不合法的维度组合
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    """
    dataclass 自动生成__init__()构造方法
    frozen=True 防止配置被意外修改
    全项目的形状约定：B=批量、T=序列长度、C=dim、D=head_dim、
    Hq=n_heads、Hkv=n_kv_heads、V=vocab_size
    """
    # 可用 token ID 的数量（合法 ID 为 0 到 V-1）
    vocab_size: int = 4096
    # 每个 token 的隐藏向量宽度 Ｃ，!=词表大小/序列长度
    dim: int = 256
    # 顺序堆叠多少个 TransformerBlock，每层有自己的参数
    n_layers: int = 4
    # Query 头数，把 C 维拆成 Hq 个 D 维子空间
    n_heads: int = 8
    # Key/Value 头数，2 个 KV 头分别供 4 个 Q 头共享，构成 GQA(Grouped Query Attention, 分组查询注意力)
    # 为什么 k/v 头比 Q 头小：未来生成文本时要保存历史 token 的 K/V，也就是 KV Cache
    n_kv_heads: int = 2
    # SwiGLU 两条上投影分支的宽度，704 接近 8*C/3 并向上对齐到 64
    hidden_dim: int = 704
    # 本实现允许的最长输入(是接口限制，不表示提前分配了 KV Cache)
    max_seq_len: int = 512
    # RoPE 的频率基数，控制不同通道对的旋转频率
    rope_theta: float = 10000.0
    # 避免除数为 0，用于数值稳定
    norm_eps: float = 1e-6
    # naive 便于公式理解
    # sdpa 使用 Pytorch 提供的方法把缩放、mask、softmax 和乘 V 等过程封装起来，并可能根据硬件选择更合适的实现
    # 两条路径在数学意义上非常相近
    attention_backend: str = "naive"

    def __post_init__(self):
        """
        对 dataclass 自动__init__()初始化的字段自动调用，尽早给出明确错误
        """
        for name in ("vocab_size", "dim", "n_layers",
                     "n_heads", "n_kv_heads", "hidden_dim", "max_seq_len"):
            # 根据字段名称字符串读取属性
            value = getattr(self, name)
            # 使用 type(...) is int 也会排除 bool(Python 中 bool 是 int 子类)
            if type(value) is not int or value <= 0:
                raise ValueError(f"'{name}' must be a positive integer")
        # 余数不为 0 无法均匀拆头报错
        if self.dim % self.n_heads:
            raise ValueError(f"'dim' must be divisible by 'n_heads' {self.n_heads}")
        # 每个 KV 头必须恰好对应相同数量的 Q 头
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        # 本项目 RoPE 旋转相邻两维度，因此每个头必须有偶数个通道
        if self.head_dim % 2:
            raise ValueError("hidden_dim must be divisible by 2")
        if not self.rope_theta > 0 or not self.norm_eps > 0:
            raise ValueError("rope_theta and norm_eps must be positive")
        if self.attention_backend not in ["naive", "sdpa"]:
            raise ValueError("attention_backend must be naive or sdpa")

    @property
    def head_dim(self):
        """
        property 让这个方法可以通过 config.head_dim 访问
        如果没有 property 要写 config.head_dim() 访问
        默认 256 // 8 = 32
        """
        return self.dim // self.n_heads
