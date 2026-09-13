"""
自回归生成：保留无缓存基线，并提供 Day 3 增量 KV Cache
"""
import argparse
import torch
from kvforge import ModelConfig, MiniLLM
from kvforge.data import CharTokenizer


@torch.no_grad()
# @torch.no_grad() 让函数内的计算不记录用于反向传播的计算图
# 生成过程中不调用 loss.backward()、optimizer.step()，因此不会更新模型权重
def generate(model, ids, max_new_tokens=100, temperature=0.8, top_k=20, use_cache=False):
    """
    :param model: 已创建并加载权重的模型
    :param ids: 编码后的提示词，形状 [B,T]
    :param max_new_tokens: 在提示词后追加多少个 token
    :param temperature: 采样温度（本项目用 0 表示贪心）
    :param top_k: 采样时保留分数最高的若干选项
    :param use_cache: 是否开启缓存路径
    :return: 原提示词 + 新生成 token
    """
    if ids.ndim != 2 or ids.size(1) == 0:
        raise ValueError('提示词不能为空')
    if max_new_tokens < 0 or temperature < 0 or top_k < 0:
        raise ValueError('生成参数不能为负')
    # 先记录原来的模式，再切到评估模式，末尾再恢复
    was_training = model.training
    model.eval()
    # 每次生成都创建自己的缓存，避免第二次错误地读取第一次的历史
    try:
        cache = model.create_cache(ids.size(0)) if use_cache and max_new_tokens else None
        # 每轮预测并追加一个 token，循环中有三条计算路径
        for _ in range(max_new_tokens):
            # 第一条：无缓存基线
            # 每次取最后一个上下文窗口，完整重算
            if cache is None:
                logits, _ = model(ids[:, -model.config.max_seq_len:])
            # 第二条：缓存为空或缓存已满
            elif cache.length == 0 or cache.length == cache.capacity:
                # 首次调用叫 prefill(一次计算整个提示词，填充各层缓存)
                # 窗口满后重新计算最后一个窗口，以保持 Day 2 的生成语义
                # 不能仅删除最旧 KV,因为后续层的历史表示仍包含被删除 token 的影响
                cache.reset()
                logits, _ = model(ids[:, -cache.capacity:], cache=cache)
            # 缓存已有历史且尚未满
            else:
                # 只输入刚生成的最新 token 执行增量 decode，它通过历史 KV 看见前文
                # 这里 ids[:, -1:] 使用 -1: 而不是 -1 是为了保留序列这一维
                # logits 是 [B,1,V]，不需要重复计算此前 token 的隐藏状态
                logits, _ = model(ids[:, -1:], cache=cache)
            # 为什么只取最后一个位置的 logits：如果是 prefill，logits.shape=[B,提示词长度,V]
            #                             每个位置都有下一 token 的预测分数,例如：输入 [a,b,c]
            #                             a位置的输出：预测a后面是什么
            #                             b位置的输出：预测b后面是什么
            #                             c位置的输出：预测c后面是什么
            #                             现在要继续整个提示词，所以取 c 位置的输出
            scores = logits[:, -1, :].float()
            # tokenizer 把 ID 0 留给未知字符,生成时将它排除，避免生成未知字符占位符
            scores[:, 0] = float('-inf')
            # 贪心分支
            if temperature == 0:
                # 选择每个样本中分数最大的 token
                # 保留最后一维 next_id.shape=[B,1],这样方便接到 ids.shape=[B,T] 后面
                next_id = scores.argmax(-1, keepdim=True)
            # 采样分支
            else:
                # 对正温度来说，温度越低概率更集中，温度越高概率更分散，它不修改模型权重，只改变当前预测分数转成概率时的分布
                scores = scores / temperature
                if top_k:
                    # 找到分数最高的 k 项，取其中最低分作为阈值，排除低于这个阈值的候选
                    threshold = scores.topk(min(top_k, scores.size(-1))).values[:, -1:]
                    scores = scores.masked_fill(scores < threshold, float('-inf'))
                # 根据概率抽取一个 token
                next_id = torch.multinomial(scores.softmax(-1), num_samples=1)
            # 最后追加到序列，这里拼接的是 token ID 序列而不是每层历史 K/V
            ids = torch.cat((ids, next_id), dim=1)
        return ids
    finally:
        # 为什么恢复：例如训练程序中途调用生成函数查看效果：训练->临时生成样例->继续训练
        #           如果生成函数永久把模型留在评估模式，就会影响后续程序的预期行为
        model.train(was_training)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--prompt', default='The ')
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--max-new-tokens', type=int, default=200)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--use-cache', action='store_true', help='启用增量 KV Cache；窗口满后重建')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    tokenizer = CharTokenizer(checkpoint['chars'])
    encoded = tokenizer.encode(args.prompt)
    if 0 in encoded:
        raise ValueError('提示词含词表外字符，请换用训练文本中的字符')
    model = MiniLLM(ModelConfig(**checkpoint['model_config'])).to(args.device)
    model.load_state_dict(checkpoint['model'])
    ids = torch.tensor([encoded], dtype=torch.long, device=args.device)
    output = generate(model, ids, args.max_new_tokens, args.temperature, args.top_k, use_cache=args.use_cache)
    print(tokenizer.decode(output[0].tolist()))


if __name__ == '__main__':
    main()
