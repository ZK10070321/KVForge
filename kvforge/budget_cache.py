"""
Day 5: 限制每层K/V最多保留多少个位置，比较两种简单淘汰策略
recent：仅保留最近位置；sink_recent：保留开头若干位置，再保留最近位置
这里只借鉴“起始位置+最近窗口”的选择方式，不是完整StreamingLLM复现
位置编号始终保持窗口内原始编号，不做超出模型max_seq_len的位置外推
"""
import torch
from .cache import KVCache


class BudgetKVCache(KVCache):
    def __init__(self, model, batch_size, capacity, policy='recent', sink_size=0):
        if type(capacity) is not int or type(batch_size) is not int:
            raise ValueError('capacity和batch_size必须是整数')
        if policy not in ('recent', 'sink_recent'):
            raise ValueError('policy必须是recent或sink_recent')
        if type(sink_size) is not int or not 0 <= sink_size < capacity:
            raise ValueError('sink_size必须非负且小于capacity，至少留一个最新位置')
        if policy == 'recent' and sink_size != 0:
            raise ValueError('recent策略不保留固定开头，请设置sink_size=0')
        super().__init__(model, batch_size, capacity)
        self.policy = policy
        self.sink_size = sink_size
        # length：当前有效存储位置数，最多等于capacity
        # seen_tokens：已经处理多少个token，决定下一个token的RoPE位置
        # 例如处理了10个token、预算4时，length=4，seen_tokens=10
        self.seen_tokens = 0
        self.positions = torch.empty(capacity, dtype=torch.long,
                                     device=model.embedding.weight.device)
        self._dirty = False
        self._written_layers = set()
        self._keep = None
        self._pending_positions = None

    @property
    def next_position(self):
        return self.seen_tokens

    def reset(self):
        super().reset()
        self.seen_tokens = 0
        self._dirty = False
        self._written_layers = set()
        self._keep = None
        self._pending_positions = None

    @property
    def metadata_bytes(self):
        # 报告中的allocated_bytes仍只表示K/V；位置表额外占用capacity×8字节
        return self.positions.numel() * self.positions.element_size()

    def validate(self, model, ids):
        if self._dirty:
            raise ValueError('上次缓存写入未完整提交，请reset后从新窗口重新开始')
        if ids.size(1) != 1:
            raise ValueError('预算缓存只支持逐token输入；prefill也应循环输入单个token')
        # 复用Day 3的设备、精度、模型身份和推理模式检查
        # 原validate的容量检查不适用于淘汰策略，因此在这里逐项检查
        if model is not self.owner:
            raise ValueError('缓存属于另一个模型')
        if model.training or torch.is_grad_enabled():
            raise ValueError('预算缓存需要eval和no_grad')
        if ids.size(0) != self.batch_size:
            raise ValueError('batch大小与缓存不符')
        weight = model.embedding.weight
        if ids.device != self.keys[0].device or weight.device != self.keys[0].device:
            raise ValueError('模型、输入与缓存必须在同一设备')
        if weight.dtype != self.keys[0].dtype:
            raise ValueError('模型精度变化后请重建缓存')
        if self.seen_tokens + 1 > model.config.max_seq_len:
            raise ValueError('位置超过模型窗口；本实验不进行长上下文外推')

    def write(self, layer_index, k, v):
        if k.size(2) != 1 or k.dtype != self.keys[layer_index].dtype or v.dtype != k.dtype:
            raise ValueError('需要同精度的单token K/V，不混用autocast')
        if not self._dirty:
            # 预留一个位置给当前token。预算包含当前token，不是“预算个历史+当前”
            if self.length < self.capacity:
                keep = torch.arange(self.length, device=k.device)
            else:
                # 满了时，从capacity个旧位置选capacity-1个，丢弃一个旧位置
                # sink_recent保留开头sink_size个槽；其余保留最近的旧位置
                # capacity=4、sink_size=1时，[0,5,6,7]+新8 → [0,6,7,8]
                sink = self.sink_size if self.policy == 'sink_recent' else 0
                recent = self.capacity - sink - 1
                prefix = torch.arange(sink, device=k.device)
                tail = torch.arange(self.length-recent, self.length, device=k.device)
                keep = torch.cat((prefix, tail))
            self._keep = keep
            self._pending_positions = torch.cat((self.positions[:self.length].index_select(0, keep),
                torch.tensor([self.seen_tokens], device=k.device, dtype=torch.long)))
            self._dirty = True
        if layer_index in self._written_layers:
            raise ValueError('同一模型调用不应重复写同一层；请reset后重试')
        count = self._keep.numel()
        if self.length == self.capacity:
            # index_select先复制到临时张量，避免重叠内存copy_破坏尚未读取的历史
            # 这会产生临时开销：预算保证常驻KV槽数，不保证总峰值显存等于该大小
            self.keys[layer_index][:, :, :count].copy_(
                self.keys[layer_index].index_select(2, self._keep))
            self.values[layer_index][:, :, :count].copy_(
                self.values[layer_index].index_select(2, self._keep))
        self.keys[layer_index][:, :, count:count+1].copy_(k)
        self.values[layer_index][:, :, count:count+1].copy_(v)
        self._written_layers.add(layer_index)
        return self.keys[layer_index][:, :, :count+1], self.values[layer_index][:, :, :count+1]

    def key_positions(self, count, device):
        # attention在commit之前执行，使用本次即将提交的真实位置列表
        return self._pending_positions

    def commit(self, count):
        if count != 1 or len(self._written_layers) != len(self.keys):
            raise ValueError('必须全部层写入成功后才能提交预算缓存')
        self.length = self._pending_positions.numel()
        self.positions[:self.length].copy_(self._pending_positions)
        self.seen_tokens += 1
        self._dirty = False
        self._written_layers.clear()
        self._keep = None
        self._pending_positions = None
