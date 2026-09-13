"""
Day 4：固定输入的推理微基准，不把文本质量和计算速度混在一起
prefill 处理 P 个提示词；decode 再处理 N 个预先准备好的 token
两条路径看到完全相同的前缀，不在计时区间采样、打印或搬运输入
这里测的是模型调用的墙钟耗时，包含 Python 调度，不是纯 GPU kernel 时间
"""
from statistics import median
from time import perf_counter

import torch


def validate_workload(model, ids, prompt_length, decode_steps, warmup, repeats):
    """
    先拒绝不合法工作量，尤其不能悄悄跨过模型窗口后继续报加速比
    这些检查的意义是 在计算性能数字之前，先确保工作量确实符合实验定义
    """
    # 验证提示词长度为正、decode步数为正、正式重复次数为正、预热次数非负
    for name, value in [('prompt_length', prompt_length), ('decode_steps', decode_steps),
                        ('repeats', repeats)]:
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} 必须是正整数')
    if type(warmup) is not int or warmup < 0:
        raise ValueError('warmup 必须是非负整数')
    # 验证输入是非空[B,T]、输入长度恰好是P+N、总长度不超过窗口
    if ids.ndim != 2 or ids.size(0) < 1:
        raise ValueError('ids 必须是非空的 [B,T] 张量')
    if ids.size(1) != prompt_length + decode_steps:
        raise ValueError('ids 长度必须恰好等于 prompt_length + decode_steps')
    if ids.size(1) > model.config.max_seq_len:
        raise ValueError('工作量超过窗口：请缩短 prompt/decode，本评测不允许窗口重建')
    # 验证输入与模型在同一设备
    if ids.device != model.embedding.weight.device:
        raise ValueError('请在计时前将 ids 和模型移到同一设备')


def synchronize(device):
    # CPU 调用正常返回时计算已经完成,CUDA 可能只提交了任务，需要等待 GPU
    # 用来实现任务同步
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def check_cache_equivalence(model, ids, prompt_length):
    """
    计时前的正确性门槛：所有位置的分数一致才允许输出性能数据
    """
    cache = model.create_cache(ids.size(0), capacity=ids.size(1))
    # 得到所有位置的参考 logits
    expected, _ = model(ids)
    # 然后计算缓存路径
    pieces = [model(ids[:, :prompt_length], cache=cache)[0]]
    # 逐个处理后续位置，将每一步 logits 放入列表
    for position in range(prompt_length, ids.size(1)):
        pieces.append(model(ids[:, position:position+1], cache=cache)[0])
    # 沿序列维度拼回完整输出 prefill的P个位置 + 后续N个位置 = P+N个位置，与 expected 比较
    actual = torch.cat(pieces, dim=1)
    # FP16 的舍入比 FP32 更明显，记录容差，不通过时抛错而不是自动放宽
    low_precision = expected.dtype in (torch.float16, torch.bfloat16)
    atol, rtol = (5e-3, 5e-3) if low_precision else (1e-5, 1e-4)
    # isfinite 检查每个值是否为有限数，.all() 表示所有元素都满足
    if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
        raise ValueError('logits 含 NaN/Inf，不能进行有效性能对比')
    # 不满足容差就抛错，停止评测，不会自动把容差调大
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return {'max_abs_error': (actual.float()-expected.float()).abs().max().item(),
            'atol': atol, 'rtol': rtol}


def run_trial(model, ids, prompt_length, decode_steps, use_cache, track_memory=False):
    """
    一轮包含 prefill 和 decode；每轮使用全新缓存，避免复用上一轮历史
    """
    device = ids.device
    synchronize(device)
    if track_memory and device.type == 'cuda':
        # 此时还没有本轮 cache
        # baseline 是本轮缓存申请之前，当前存活张量已经占用的 GPU 内存
        baseline = torch.cuda.memory_allocated(device)
        # 重置峰值统计的起点(不会释放显存)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline = None
    # 每轮都重新开始空缓存 → prefill → N步decode,如果上一轮结束后直接继续使用上一轮的历史 + 本轮重复输入,
    # 位置、长度和工作量都会改变，甚至越界
    # 新建缓存发生在计时开始前，因此缓存申请不计入 prefill 时间，缓存申请产生的 GPU 张量占用纳入本轮显存峰值统计
    cache = model.create_cache(ids.size(0), ids.size(1)) if use_cache else None
    # prefill 计时
    # 先等此前任务完成，再开始计时，实现开始前同步
    synchronize(device)
    start = perf_counter()
    model(ids[:, :prompt_length], cache=cache)
    # 等这个设备此前提交的工作完成，再继续往下执行，实现结束前同步
    synchronize(device)
    prefill_seconds = perf_counter() - start
    # decode 计时
    start = perf_counter()
    for position in range(prompt_length, prompt_length + decode_steps):
        if use_cache:
            # 只处理最新位置，历史通过 cache 传入，每步有效上下文仍在增长
            model(ids[:, position:position+1], cache=cache)
        else:
            # 基线每步重算相同的完整前缀，不是重复运行固定短输入
            model(ids[:, :position+1])
    synchronize(device)
    decode_seconds = perf_counter() - start
    # 取得这一统计区间内 PyTorch 张量分配量达到的峰值
    peak = torch.cuda.max_memory_allocated(device) if baseline is not None else None
    return {
        'prefill_seconds': prefill_seconds,
        'decode_seconds': decode_seconds,
        'cache_allocated_bytes': cache.allocated_bytes if cache is not None else 0,
        # CPU 返回 None，在 JSON 中是 null，绝不把“没有测量”写成显存为 0
        'cuda_baseline_allocated_bytes': baseline,
        'cuda_peak_allocated_bytes': peak,
        # peak - baseline 可能包含 本轮缓存、注意力临时张量、其他本轮新增张量，它不一定等于 KV 缓存大小
        # 而且这里测的是 PyTorch 的 allocated 张量内存，不是 nvidia-smi 显示的整个进程占用，也不是分配器的 reserved 内存
        'cuda_peak_extra_bytes': peak - baseline if peak is not None else None,
    }


def summarize(samples, batch_size, decode_steps):
    """
    保留原始样本，用中位数减弱偶发抖动；不会取最快一次充当典型结果
    """
    # 为什么使用中位数，不挑最快一次: 假设五次耗时 10、11、10、12、50 毫秒，排序后 10、10、11、12、50，
    #                           中位数 = 11毫秒，平均数为 (10+11+10+12+50)÷5 = 18.6毫秒，一次
    #                           较大的偶发波动把平均数拉高了。项目使用 median(...) 帮助描述这些样本的典型水平
    # 为什么不只选最小值 10: 因为只报告最快一次容易显得过于乐观，选择中位数并保留所有样本，让结果可以被检查，
    #                     但中位数也不能取代原始数据，比如 [10,10,10,10,1000] 虽然中位数为 10，仍存在很大的
    #                     尾部波动。因此项目同时保存汇总指标和每轮原始计时
    prefill = median(s['prefill_seconds'] for s in samples)
    decode = median(s['decode_seconds'] for s in samples)
    peaks = [s['cuda_peak_allocated_bytes'] for s in samples
             if s['cuda_peak_allocated_bytes'] is not None]
    extras = [s['cuda_peak_extra_bytes'] for s in samples
              if s['cuda_peak_extra_bytes'] is not None]
    return {
        'prefill_ms_median': prefill * 1000,
        'decode_ms_median': decode * 1000,
        # 一步同时处理整个 batch：这是每批一步的平均延迟，不是除以 batch 的延迟
        'decode_step_ms': decode * 1000 / decode_steps,
        'decode_tokens_per_second': batch_size * decode_steps / decode,
        'cache_allocated_bytes': samples[0]['cache_allocated_bytes'],
        'cuda_peak_allocated_bytes': max(peaks) if peaks else None,
        'cuda_peak_extra_bytes': max(extras) if extras else None,
        'samples': samples,
    }


@torch.no_grad()
def benchmark_pair(model, ids, prompt_length, decode_steps, warmup=2, repeats=5):
    """
    同一个模型比较 cache/no-cache，返回 JSON 可保存的普通 Python 数据
    """
    validate_workload(model, ids, prompt_length, decode_steps, warmup, repeats)
    was_training = model.training
    model.eval()
    try:
        correctness = check_cache_equivalence(model, ids, prompt_length)
        # 为什么需要预热：程序第一次运行时可能包含懒初始化、首次建立执行所需的资源、分配器和其他缓存状态变化、
        #              与后续运行不同的启动开销，如果只测第一次，就可能把这些启动因素当成稳定运行成本。
        #              预热就是先运行几轮，让常见初始化发生，这些轮次不计入最终成绩。两条路径都预热，避免
        #              只让一条路径获得这个条件。
        # 但是预热不能保证完全消除波动，也不代表第一次运行的成本不存在
        # 本项目测的是预热后的表现，而非冷启动表现
        # 此外，正确性验证本身已经执行过模型，即使设置 warmup=0 也不能声称测到纯冷启动
        for _ in range(warmup):
            for enabled in (False, True):
                run_trial(model, ids, prompt_length, decode_steps, enabled)
        samples = {False: [], True: []}
        for repeat in range(repeats):
            # 交替顺序，降低某条路径总是先运行导致的系统性偏差
            # repeat % 2 是取除以 2 的余数用来判断奇偶,执行顺序：第0轮：无缓存 → 有缓存
            #                                              第1轮：有缓存 → 无缓存
            #                                              第2轮：无缓存 → 有缓存...
            #为什么不永远先跑无缓存:因为运行顺序可能与设备温度、系统负载等因素发生关联,交替顺序可以降低某条路径永远先跑的偏差,
            #                   它不能完全消除噪声，但比固定顺序更谨慎。
            order = (False, True) if repeat % 2 == 0 else (True, False)
            for enabled in order:
                samples[enabled].append(run_trial(
                    model, ids, prompt_length, decode_steps, enabled, track_memory=True))
        baseline = summarize(samples[False], ids.size(0), decode_steps)
        cached = summarize(samples[True], ids.size(0), decode_steps)
        return {
            'prompt_length': prompt_length, 'decode_steps': decode_steps,
            'batch_size': ids.size(0), 'cache_capacity': ids.size(1),
            'correctness': correctness, 'no_cache': baseline, 'cache': cached,
            # 大于1表示缓存更快，小于1表示本次配置下缓存更慢；原样报告。
            'decode_speedup': baseline['decode_ms_median'] / cached['decode_ms_median'],
        }
    finally:
        model.train(was_training)
