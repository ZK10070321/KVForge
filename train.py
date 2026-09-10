"""
Day 2 训练入口,把数据、模型、优化器组织起来,训练并保存模型
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import torch
from kvforge import ModelConfig, MiniLLM
from kvforge.data import load_data, get_batch
from kvforge.training import learning_rate, evaluate, save_checkpoint


def main():
    # 命令行参数:每一行控制什么,下面这些 add_argument() 都是在声明允许从终端传入的选项
    p = argparse.ArgumentParser()
    # 准备好的训练数据目录
    p.add_argument('--data', default='data/prepared')
    # 日志与 checkpoint 输出目录
    p.add_argument('--out', default='runs/day2')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    # 计划训练总步数
    p.add_argument('--steps', type=int, default=500)
    # 在指定全局步数暂停(0 表示不提前暂停)
    p.add_argument('--stop-after', type=int, default=0,
                   help='在指定全局步数暂停；保留原学习率总步数，便于恢复验证')
    # 每步使用多少个文本窗口、每个输入窗口包含多少个 token、模型隐藏维度
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--dim', type=int, default=128)
    # Block 数量、Q 头数、K/V 头数
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--kv-heads', type=int, default=2)
    # SwiGLU 内部宽度、注意力实现
    p.add_argument('--hidden-dim', type=int, default=352)
    p.add_argument('--backend', choices=['naive', 'sdpa'], default='sdpa')
    # 峰值学习率、最低学习率、学习率预热步数、每隔多少步验证并保存、每次验证抽取多少个 batch
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--min-lr', type=float, default=3e-5)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--eval-batches', type=int, default=5)
    # 随机种子、从哪个 checkpoint 恢复、是否反复训练同一个 batch
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', help='恢复自己生成的 last.pt；恢复时使用保存的训练设置')
    p.add_argument('--overfit', action='store_true', help='重复训练一个固定 batch，仅检查可学习性')
    # 正式解析终端参数
    args = p.parse_args()
    # checkpoint 中的超参数是恢复训练的权威来源；steps 不在恢复时悄悄改变
    # 决定从头训练还是恢复训练,若没有传 --resume，从头训练
    # map_location='cpu' 表示先把保存的张量放到 CPU，即使 checkpoint 原来是在 GPU 上保存的，也先在 CPU 上读取
    # weights_only=True 使用受限反序列化方式，适用于这里保存的张量和基础类型字典，它不等于“只读取 model 这一项”
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    settings = checkpoint['settings'] if checkpoint else vars(args).copy()
    # 参数检查
    for name in ('steps', 'batch_size', 'seq_len', 'eval_every', 'eval_batches'):
        if settings[name] <= 0:
            raise ValueError(f'{name} must be positive')
    if not 0 <= settings['warmup'] < settings['steps']:
        raise ValueError('需要 0 <= warmup < steps')
    if not 0 <= settings['min_lr'] <= settings['lr'] or settings['lr'] <= 0:
        raise ValueError('需要 0 <= min_lr <= lr 且 lr > 0')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA 不可用，请先安装对应 PyTorch 构建，或使用 --device cpu')
    if settings['precision'] == 'fp16' and args.device != 'cuda':
        raise ValueError('FP16 训练需要 CUDA；CPU 使用 fp32')
    # 限制 CPU 计算线程数量，小模型测试中线程太多可能增加调度开销
    torch.set_num_threads(2)
    # 设置 PyTorch 随机种子，使模型随机初始化等操作在同一环境下更容易复现
    torch.manual_seed(settings['seed'])
    train, val, tokenizer, fingerprint = load_data(args.data)
    if min(len(train), len(val)) <= settings['seq_len']:
        raise ValueError('训练/验证文本太短，请增加文本或减小 --seq-len')
    # 把训练参数转换成 Day 1 的模型配置
    config = ModelConfig(vocab_size=tokenizer.vocab_size, dim=settings['dim'],
                         n_layers=settings['layers'], n_heads=settings['heads'],
                         n_kv_heads=settings['kv_heads'], hidden_dim=settings['hidden_dim'],
                         max_seq_len=settings['seq_len'], attention_backend=settings['backend'])
    # 先创建模型，再把参数移到目标设备
    model = MiniLLM(config).to(args.device)
    # AdamW 可以先理解成三个动作：1. 参考最近一段时间的梯度方向
    #                         2. 根据梯度大小的历史调整每个参数的更新尺度
    #                         3. 使用权重衰减，让参数适度向 0 收缩
    # 因此 AdamW 会保存历史统计量，恢复训练时只加载模型权重，不加载优化器状态，就会丢掉这些历史
    # weight_decay=0.01 表示权重衰减强度，它对参数施加一个与参数值相关的收缩量，帮助约束参数规模，但不是“每步直接把权重减少 1%”，收缩还与学习率有关
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=0.01)
    # GradScaler 主要解决小梯度下溢问题，梯度变大后更容易被 FP16 表示,如果 enabled=False，scaler 基本退化为直接执行普通训练操作
    scaler = torch.amp.GradScaler('cuda', enabled=settings['precision'] == 'fp16')
    # 正常训练的采样器、专门用于抽取固定 batch 的采样器，它们与模型初始化使用不同随机流，避免互相干扰
    generator = torch.Generator().manual_seed(settings['seed'] + 1)
    fixed_generator = torch.Generator().manual_seed(settings['seed'] + 2)
    # 预先抽取一批数据，如果启用 --overfit 就每一步反复学习这同一批数据，检查模型是否能够记住它
    # 为什么需要这个测试：如果连固定的一小批数据都学不会，往往说明损失、标签、梯度或参数更新有问题，
    #                 但记住一个 batch 不代表模型具有泛化能力
    fixed_batch = get_batch(train, settings['batch_size'], settings['seq_len'], fixed_generator, args.device)
    # 同时初始化两个变量：start=0：从第 0 步开始
    #                 best=正无穷：还没有最佳验证成绩
    # 因为任何正常的有限验证 loss 都小于正无穷，所以第一次验证通常会成为 best
    start, best = 0, float('inf')
    # 恢复 checkpoint
    if checkpoint:
        # 检查当前数据文件是否与原训练一致，避免拿另一份数据继续同一个实验而不自知
        if checkpoint['data_fingerprint'] != fingerprint:
            raise ValueError('数据与 checkpoint 不匹配，拒绝恢复')
        # 恢复模型参数
        model.load_state_dict(checkpoint['model'])
        # 恢复 AdamW 的历史统计量和参数组设置
        optimizer.load_state_dict(checkpoint['optimizer'])
        # 恢复 FP16 的缩放状态
        scaler.load_state_dict(checkpoint['scaler'])
        # 恢复训练采样器的位置
        generator.set_state(checkpoint['sampler_rng'])
        # 恢复 PyTorch CPU 全局随机状态
        torch.set_rng_state(checkpoint['torch_rng'])
        if args.device == 'cuda' and checkpoint['cuda_rng']:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        # 恢复已完成步数和最佳验证 loss
        start, best = checkpoint['step'], checkpoint['best_val']
    # 创建输出目录
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if not checkpoint and (output / 'last.pt').exists():
        raise ValueError('输出目录已有 checkpoint，请 --resume 或选择新的 --out')
    print(f'device={args.device}, parameters={sum(p.numel() for p in model.parameters()):,}, vocab={tokenizer.vocab_size}')
    # 在开始本次训练前计算一次验证 loss，恢复训练时，它是“恢复后的模型在继续训练前”的验证 loss，不是随机初始化模型的 loss
    print('initial_val_loss=', evaluate(model, val, settings['batch_size'], settings['seq_len'], settings['eval_batches'], args.device))
    limit = min(settings['steps'], args.stop_after or settings['steps'])
    # 一次循环对应一次优化尝试；last.pt 记录已完成的全局步数
    for step in range(start, limit):
        # 设置训练模式
        model.train()
        x, y = fixed_batch if settings['overfit'] else get_batch(train, settings['batch_size'], settings['seq_len'], generator, args.device)
        lr = learning_rate(step, settings['steps'], settings['warmup'], settings['lr'], settings['min_lr'])
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=settings['precision'] == 'fp16'):
            _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError('loss 非有限值，停止训练；可从上一份 last.pt 恢复')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        completed = step + 1
        if completed % settings['eval_every'] == 0 or completed == limit:
            val_loss = evaluate(model, val, settings['batch_size'], settings['seq_len'], settings['eval_batches'], args.device)
            improved = val_loss < best
            best = min(best, val_loss)
            record = {'step': completed, 'train_loss': loss.item(), 'val_loss': val_loss, 'lr': lr}
            print(record)
            with (output / 'metrics.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record) + '\n')
            payload = {'model': model.state_dict(), 'model_config': asdict(config),
                       'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                       'settings': settings, 'step': completed, 'best_val': best,
                       'chars': tokenizer.chars, 'data_fingerprint': fingerprint,
                       'sampler_rng': generator.get_state(), 'torch_rng': torch.get_rng_state(),
                       'cuda_rng': torch.cuda.get_rng_state_all() if args.device == 'cuda' else []}
            save_checkpoint(output / 'last.pt', payload)
            if improved:
                save_checkpoint(output / 'best.pt', payload)


if __name__ == '__main__':
    main()
