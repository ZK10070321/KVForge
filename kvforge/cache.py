"""Day 3：一次生成请求使用一个缓存；缓存不属于模型权重。"""
import torch


class KVCache:
    """
    同时保存缓存张量、当前有效长度、最大容量、所属模型、batch 大小
    它没有继承 nn.Module，因为它不是需要训练的网络层，不包含要由优化器更新的参数
    """

    def __init__(self, model, batch_size, capacity=None):
        """
        :param model: 给哪个模型创建缓存
        :param batch_size: 同时处理几条序列
        :param capacity: 最多保存多少个位置的缓存(不指定时使用模型的上下文上限)
        """
        # 给模型配置起一个短名称 C 方便访问
        c = model.config
        capacity = c.max_seq_len if capacity is None else capacity
        if batch_size < 1 or not 1 <= capacity <= c.max_seq_len:
            raise ValueError('batch_size 必须为正，capacity 必须在模型上下文范围内')
        # 保存模型身份，防止把模型 A 的历史误用于模型 B。
        self.owner = model
        # 刚创建缓存，所以有效历史长度为 0
        self.length = 0
        self.capacity = capacity
        self.batch_size = batch_size
        weight = model.embedding.weight
        # 缓存张量的形状是 [B, Hkv, capacity, D]
        shape = (batch_size, c.n_kv_heads, capacity, c.head_dim)
        # 每执行一次循环就为一层创建独立的 K 缓存以及与 k 具有相同形状、设备和数据类型的独立的 V 缓存
        self.keys = [torch.empty(shape, device=weight.device, dtype=weight.dtype)
                     for _ in range(c.n_layers)]
        self.values = [torch.empty_like(k) for k in self.keys]

    def reset(self):
        # 旧数据无需清零,只有 [:length] 能被读取,后续写入会覆盖旧位置
        self.length = 0

    @property
    def allocated_bytes(self):
        # 只统计常驻 K/V 张量，不包含模型参数、注意力临时张量或分配器开销
        # self.keys + self.values 是两个 Python 列表的拼接，不是对 K/V 张量做数值相加
        # t.numel() 返回张量的元素总数，t.element_size() 返回每个元素占几个字节
        return sum(t.numel() * t.element_size() for t in self.keys + self.values)

    def validate(self, model, ids):
        """
        防止程序虽然运行，但用了不属于当前计算的历史
        """
        if model is not self.owner:
            raise ValueError('缓存只能交给创建它的模型使用')
        if model.training or torch.is_grad_enabled():
            raise ValueError('缓存推理需要 model.eval() 和 torch.no_grad()')
        if ids.size(0) != self.batch_size:
            raise ValueError('输入 batch 大小与缓存不同')
        weight = model.embedding.weight
        if ids.device != self.keys[0].device or weight.device != self.keys[0].device:
            raise ValueError('模型、输入和缓存必须在同一设备；移动模型后请重建缓存')
        if weight.dtype != self.keys[0].dtype:
            raise ValueError('模型精度已改变，请重建缓存')
        if self.length + ids.size(1) > self.capacity:
            raise ValueError('KV Cache 容量不足；请重建窗口或缩短生成长度')

    def write(self, layer_index, k, v):
        """
        怎样把新 K/V 放进去
        新 K 做一次 RoPE 后进入缓存,历史 K 从缓存直接读取，不重复旋转;
        V 在当前标准 RoPE 实现中不参与位置旋转，它保存用于加权汇总的内容
        :param layer_index: 当前是第几层
        :param k: 这一层新输入产生的 K
        :param v: ...产生的V
        """
        # 先检查精度，新 K/V 与缓存的数据类型应一致，当前简化实现不混用 autocast 和缓存精度，避免隐式转换掩盖误差
        if k.dtype != self.keys[layer_index].dtype or v.dtype != k.dtype:
            raise ValueError('缓存推理请关闭 autocast，或将模型显式转换到目标精度')
        # k 的形状是 [B, Hkv, 新输入长度, D],所以 k.size(2) 取到的是新输入长度
        end = self.length + k.size(2)
        self.keys[layer_index][:, :, self.length:end].copy_(k)
        self.values[layer_index][:, :, self.length:end].copy_(v)
        # 切片是已有存储的视图，只暴露有效历史，不读取尚未写入的空位，是新 Q 做注意力计算时需要的内容
        return self.keys[layer_index][:, :, :end], self.values[layer_index][:, :, :end]
