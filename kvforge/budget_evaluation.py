"""Day 5质量评测：固定真实文本，比较相同预测位置的NLL和PPL。"""
import math
import torch
from torch.nn import functional as F


def make_windows(tokens, window_length, max_windows=0):
    """取不重叠的预测区间；相邻窗口共享一个边界token作为下一个窗口的输入。

    一个窗口需要T+1个token：前T个输入，后T个作为正确答案。
    最后不足T+1个token的部分丢弃，报告会明确窗口数与实际打分token数。
    """
    if type(window_length) is not int or window_length < 1 or max_windows < 0:
        raise ValueError('window_length必须为正，max_windows不能为负')
    if tokens.ndim != 1 or len(tokens) <= window_length:
        raise ValueError('验证数据必须是一维且至少包含window_length+1个token')
    starts = list(range(0, len(tokens)-window_length, window_length))
    if max_windows:
        starts = starts[:max_windows]
    return torch.stack([tokens[i:i+window_length+1] for i in starts])


@torch.no_grad()
def evaluate_policy(model, windows, policy='full', capacity=None, sink_size=0, score_from=0):
    """逐token喂真实输入，不把模型预测错的token接回去，避免比较不同文本。

    score_from按输入位置从0计数：24表示只打分输入位置24及以后的预测，
    即预测窗口中token索引25及以后。所有策略必须使用同一个score_from。
    """
    if windows.ndim != 2 or windows.size(0) < 1:
        raise ValueError('windows必须为非空[窗口数,T+1]')
    length = windows.size(1)-1
    if not 1 <= length <= model.config.max_seq_len or not 0 <= score_from < length:
        raise ValueError('窗口/评分起点越界')
    if policy not in ('full', 'recent', 'sink_recent'):
        raise ValueError('未知策略')
    if policy == 'full':
        if sink_size != 0 or capacity not in (None, length):
            raise ValueError('full使用完整窗口容量且没有sink')
        capacity = length
    elif type(capacity) is not int or not 1 <= capacity <= length:
        raise ValueError('预算必须在1到窗口长度之间')
    was_training = model.training
    model.eval()
    losses = []
    selected_logits = []
    retained = []
    try:
        for window in windows:
            window = window.to(model.embedding.weight.device)
            cache = (model.create_cache(1, capacity) if policy == 'full' else
                     model.create_budget_cache(1, capacity, policy, sink_size))
            outputs = []
            for position in range(length):
                logits, _ = model(window[position:position+1].view(1, 1), cache=cache)
                outputs.append(logits[0, 0].float())
            # [T,V]对[T]；答案来自原始文本的下一位置，不能再次移动标签。
            all_logits = torch.stack(outputs)
            if policy == 'full':
                # 完整缓存应与完整前向等价；预算策略本来就是近似，不能要求相等。
                reference, _ = model(window[:-1].view(1, -1))
                torch.testing.assert_close(all_logits, reference[0].float(), atol=1e-5, rtol=1e-4)
            scored = all_logits[score_from:]
            targets = window[1+score_from:]
            if not torch.isfinite(scored).all():
                raise ValueError('预测包含NaN/Inf，停止质量评测')
            nll = F.cross_entropy(scored, targets, reduction='none')
            if not torch.isfinite(nll).all():
                raise ValueError('NLL出现非有限值，请检查模型数值范围')
            losses.append(nll.cpu())
            selected_logits.append(scored.cpu())
            retained.append(list(range(cache.length)) if policy == 'full' else
                            cache.positions[:cache.length].tolist())
        token_losses = torch.cat(losses)
        mean_nll = token_losses.double().mean().item()
        return {
            'policy': policy, 'capacity': capacity, 'sink_size': sink_size,
            'windows': windows.size(0), 'scored_tokens': token_losses.numel(),
            'score_from': score_from, 'nll': mean_nll,
            'ppl': math.exp(mean_nll) if mean_nll < 700 else None,
            'kv_allocated_bytes': cache.allocated_bytes,
            'position_metadata_bytes': 0 if policy == 'full' else cache.metadata_bytes,
            'final_retained_positions': retained,
            # 逐窗口结果便于检查差异；小样本不提供虚假的统计显著性结论。
            'window_nll': [x.double().mean().item() for x in losses],
            '_logits': torch.cat(selected_logits),
        }
    finally:
        model.train(was_training)


def compare_with_full(result, reference):
    """补充相对指标；移除张量后返回可直接存成JSON的结果。"""
    full_logits = reference['_logits']
    logits = result['_logits']
    if logits.shape != full_logits.shape or result['scored_tokens'] != reference['scored_tokens']:
        raise ValueError('比较必须使用同一批预测位置')
    clean = {k: v for k, v in result.items() if k != '_logits'}
    clean.update({
        'delta_nll': result['nll'] - reference['nll'],
        'argmax_agreement': (logits.argmax(-1) == full_logits.argmax(-1)).float().mean().item(),
        'kv_saving_fraction': 1 - result['kv_allocated_bytes']/reference['kv_allocated_bytes'],
    })
    return clean
