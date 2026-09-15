"""
提供训练辅助功能：
learning_rate()：计算当前学习率
evaluate()：计算稳定的验证 loss
save_checkpoint()：安全保存训练状态
"""
import math
from pathlib import Path
import torch
from .data import get_batch


def learning_rate(step, total, warmup, peak, minimum):
    """
    学习率太大会使更新幅度太大，loss 可能震荡或爆炸；太小会使训练速度很慢
    :param step: 当前从 0 开始的训练步
    :param total: 计划总训练步数
    :param warmup: 预热步数
    :param peak: 最高学习率
    :param minimum: 最低学习率
    """
    # step 从 0 开始,前 warmup 步逐渐升高
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    # 随后余弦下降到 minimum
    # progress 被映射到 0 到 1(0 表示刚进入衰减阶段，1 表示到达最后一步)
    # 余弦衰减公式中：当 progress=0 时，learning_rate = peak
    #              当 progress=1 时，learning_rate = minimum
    #              中间会平滑下降，而不是突然降低。
    progress = min(1.0, (step - warmup) / max(1, total - warmup - 1))
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, tokens, batch_size, seq_len, batches, device):
    """
    不遍历整个验证集，而是固定随机抽取 batches 个验证 batch 计算平均 loss，这样评估速度较快，但结果是采样估计
    @torch.no_grad() 执行整个 evaluate() 时不记录自动求导图
                    使用原因: 训练需要 forward → 保存计算图 → backward,而验证只需要计算 loss 而
                    不用 backward，因此禁用梯度可以减少内存占用和额外计算、防止误用验证结果反向传播。
    """
    # 记录模型原状态:对于每个 nn.module 的 model.training,如果当前是训练模式为 True，如果当前是评估模式则为 False
    # 先记住原状态是为了函数结束后恢复
    was_training = model.training
    # 设置进入评估模式:eval() 会设置 model.training = False 并递归设置所有子模块
    model.eval()
    # 创建固定验证随机数生成器：每次调用 evaluate() 都创建一个新生成器,并使用相同种子 2026,所以每次验证都会抽到
    #                      同一批窗口,这样是在相同验证样本上比较，降低随机采样造成的波动
    # 它与训练采样器是不同对象，因此不会消耗或改变训练采样顺序
    generator = torch.Generator().manual_seed(2026)
    # 用于保存每个验证 batch 的 loss
    losses = []
    try:
        # 循环评估多个 batch
        for _ in range(batches):
            # 抽取验证 batch，计算验证 loss 并保存到 losses 列表中，最后返回平均 loss
            x, y = get_batch(tokens, batch_size, seq_len, generator, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        return sum(losses) / len(losses)
    finally:
        # 恢复原模式
        model.train(was_training)


def save_checkpoint(path, payload):
    """
    checkpoint 是训练现场的存档，不只保存模型权重，还会保存模型参数、优化器状态、当前步数、
    学习率配置、随机数状态、数据指纹、最佳验证 loss，这样恢复时可以尽量接着原来的训练过程继续。
    :param path:最终的 checkpoint 的路径
    :param payload:需要保存的字典
    """
    path = Path(path)
    # 生成临时文件名(with_suffix() 是替换扩展名,不是在原扩展名后继续追加扩展)
    temporary = path.with_suffix('.tmp')
    # 先保存临时文件，将 checkpoint 字典先完整写到 last.tmp
    # 为什么不直接覆盖 last.pt:假设保存过程中程序崩溃、电脑断电、磁盘写入失败，如果直接覆盖 last.pt，
    #                       上一份完整 checkpoint 可能被破坏，故先写临时文件，上一份 last.pt 会继续保留
    torch.save(payload, temporary)
    # 用临时文件替换正式文件(当它们位于同一文件系统时，这种替换通常比直接持续覆盖正式文件更安全)
    temporary.replace(path)
