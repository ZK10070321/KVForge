"""
最小训练链路检查：随机 token -> 前向 -> 损失 -> 反向 -> 一次参数更新
在项目根目录运行 python smoke.py；不下载数据，不衡量语言生成质量
"""

import argparse
import torch
from kvforge import ModelConfig, MiniLLM


def main():
    """
    解析设备/后端选项，验证默认模型能执行一次训练步骤
    """
    # argparse 从终端读取 --device 和 --backend，限制为已支持的选项
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--backend", choices=("naive", "sdpa"), default="naive")
    args = parser.parse_args()
    # 固定随机种子，便于同一环境复现;不保证跨设备/版本逐位一致
    torch.manual_seed(42)
    # 小 CPU 测试限制线程，避免线程调度开销超过实际计算量
    torch.set_num_threads(2)
    c = ModelConfig(attention_backend=args.backend)
    # 参数和输入必须在同一设备;cuda 需要可用的 CUDA 版 PyTorch
    model = MiniLLM(c).to(args.device)
    # 每条取 33 个 token，错位切片后得到长度 32 的输入和标签
    tokens = torch.randint(c.vocab_size, (2, 33), device=args.device)
    # AdamW 保存并更新模型参数；这里只验证链路，不是正式训练超参数方案
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # 例如 [a,b,c,d] -> 输入 [a,b,c]，目标 [b,c,d]，每个位置预测下一项
    logits, loss = model(tokens[:, :-1], tokens[:, 1:])
    # 沿计算图求梯度并写入每个参数的 .grad，此时参数值尚未更新
    loss.backward()
    # 检查每个参数都参与计算，且梯度不含 NaN/Inf；这不是梯度正确性的证明
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    # 根据梯度更新参数。本脚本只有一步，无旧梯度；多步训练需先 zero_grad
    optimizer.step()
    print(f"torch={torch.__version__}, device={args.device}, backend={args.backend}")
    # parameters() 会去重共享参数，因此嵌入/输出头只统计一次
    print(f"parameters={sum(p.numel() for p in model.parameters()):,}")
    # loss 是更新之前前向计算得到的值；不能将其当作更新之后的训练成绩
    print(f"logits={tuple(logits.shape)}, loss={loss.item():.4f}; backward + optimizer OK")


if __name__ == "__main__":
    # 直接运行脚本才进入 main,被其他模块导入时不会自动执行训练
    main()
