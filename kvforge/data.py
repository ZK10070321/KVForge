"""
Day 2 : 字符词表仅从训练文本建立，验证文本不参与词表学习
"""

import hashlib
import json
from pathlib import Path
import torch

class CharTokenizer:
    """
    一个字符对应一个 token,ID 0 留给未知字符
    """
    def __init__(self, chars):
        self.chars = list(chars)
        # 用字典推导式，建立字符到整数的映射
        # 因为 ID 0 要留给未知字符，所以这里使用 i+1
        self.stoi = {char: i + 1 for i, char in enumerate(self.chars)}

    @classmethod
    def fit(cls, text):
        # 普通实例方法的第一个参数量是 self，此处是 cls 代表类本身，可直接调用而不需要先手动实例化 tokenizer
        # set 集合数据结构对文本字符去重,sorted 进行排序,使同一份文本总是产生相同 ID 映射
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        # 返回词表大小,len(self.chars) 是训练文本中不同字符的数量，还要加上未知字符 ID 0
        return len(self.chars) + 1

    def encode(self, text):
        # 逐字符遍历文本,get(key,default) 表示如果 key 在字典中就返回其 value 值，若不存在就返回 default
        return [self.stoi.get(char, 0) for char in text]

    def decode(self, ids):
        # 解码 token ID，如果 i > 0 则使用 chars[i-1],否则使用 '�'
        # "".join(...) 把所有字符连接起来
        return ''.join(self.chars[i - 1] if i > 0 else '�' for i in ids)


def prepare(source, output, val_fraction=0.1):
    """
    把原始文本变成训练文件并返回元信息
    :param source: 原始文本路径
    :param output: 处理结果保存目录
    :param val_fraction: 验证集比例，默认 0.1，即 10%
    """
    # 检查验证集的比例是否合法
    if not 0 < val_fraction < 1:
        raise ValueError('val_fraction must be between 0 and 1')
    # 以 utf-8 编码读取整个原始文本，text.type = str
    text = Path(source).read_text(encoding='utf-8')
    # 切分训练集与验证集
    split = int(len(text) * (1 - val_fraction))
    train_text, val_text = text[:split], text[split:]
    # 为什么至少两个字符:语言模型至少需要一个输入 token 和一个目标 token
    if min(len(train_text), len(val_text)) < 2:
        raise ValueError('文本太短，训练集和验证集至少各需要两个字符')
    # 用训练文本建立词表 stoi
    tokenizer = CharTokenizer.fit(train_text)
    # 创建输出目录:parents=True 表示父目录不存在时也一起创建,exist_ok=True 表示目录已经存在也不报错
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # JSON 保存可阅读的词表；张量只保存整数，torch.load 可使用 weights_only
    # 生成元信息
    metadata = {
        # 保存字符列表，未来加载数据时可恢复完全相同的 tokenizer
        'chars': tokenizer.chars,
        # 创建 SHA-256 哈希，.hexdigest() 将结果表示成容易保存的十六进制字符串
        'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
        # 记录训练 token 数量
        'train_tokens': len(train_text),
        # 记录验证 token 数量
        'val_tokens': len(val_text),
        # 检查验证文本中有多少个字符在训练词表中
        'val_unknown': sum(char not in tokenizer.stoi for char in val_text)
    }
    # 保存字符表和元信息，以便加载时恢复同一ID映射。
    (output / 'tokenizer.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    # 将训练集和验证集编码为 PyTorch 一维整数张量，保存为 ‘train.pt’ 和 'val.pt'
    torch.save(torch.tensor(tokenizer.encode(train_text), dtype=torch.long), output / 'train.pt')
    torch.save(torch.tensor(tokenizer.encode(val_text), dtype=torch.long), output / 'val.pt')
    # 返回元信息
    return metadata


def load_data(folder):
    """
    重新加载处理结果
    """
    # 接收数据目录
    folder = Path(folder)
    metadata = json.loads((folder / 'tokenizer.json').read_text(encoding='utf-8'))
    # weights_only=True 限制加载器只接收 PyTorch 支持的安全数据类型,而不随意反序列化自定义 Python 对象
    train = torch.load(folder / 'train.pt', weights_only=True)
    val = torch.load(folder / 'val.pt', weights_only=True)
    # 这里先创建一个空的 SHA-256 数据指纹计算器,接下来可以连续向它加入多个文件的内容
    # 内容指纹用于恢复训练时拒绝换成另一份数据,即便文件路径相同
    digest = hashlib.sha256()
    # 计算三个文件的联合指纹
    # 依次读取三个文件的原始字节后交给 digest.update(...)
    # 最终哈希同时取决于 tokenizer、训练 token 和验证 token，即使目录名称没变,只要其中一个文件内容变了,联合指纹通常就会变化
    for name in ('tokenizer.json', 'train.pt', 'val.pt'):
        digest.update((folder / name).read_bytes())
    # 返回训练和验证 token 张量、从 JSON 字符表恢复的 tokenizer、三个数据文件的联合哈希
    return train, val, CharTokenizer(metadata['chars']), digest.hexdigest()


def get_batch(tokens, batch_size, seq_len, generator, device):
    """
    随机抽取训练样本
    :param tokens: 完整的一维 token 序列
    :param batch_size: 依次抽取多少个窗口，记为 B
    :param seq_len: 每个输入窗口长度，记为 T
    :param generator: 随机数生成器
    :param device: 最终放到 CPU 或 CUDA
    """
    # 为了构造长度 T 的输入和目标，需要连续读取 T+1 个 token
    # 例：seq_len = 4 时，必须读取 [a, b, c, d, e] 才能得到
    #    input  = [a, b, c, d]、target = [b, c, d, e]
    #    因此数据至少需要 seq_len + 1 个 token
    if len(tokens) <= seq_len:
        raise ValueError(f'数据需要至少 {seq_len + 1} tokens，当前 {len(tokens)}')
    # 在 [0,len(tokens) - seq_len) 之中随机选取窗口起点(依次抽取 batch_size 个窗口起点)
    # generator=generator 指定使用外部传入的随机数生成器,便于保存和恢复采样状态
    starts = torch.randint(len(tokens) - seq_len, (batch_size,), generator=generator)
    # 对于每个窗口起点取连续 T+1 个连续 token
    # torch.stack() 在前面增加一个 batch 维度: B 个 [T+1]->[B,T+1]
    windows = torch.stack([tokens[i:i + seq_len + 1] for i in starts.tolist()])
    # 错位生成输入与目标
    # windows[:, :-1]:所有 batch 去掉每行最后一个 token
    # windows[:, 1:]:所有 batch 去掉每行第一个 token
    # 最终得到 input shape  = [B,T]、target shape = [B,T]
    return windows[:, :-1].to(device), windows[:, 1:].to(device)
