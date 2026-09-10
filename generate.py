"""无 KV Cache 的生成基线：每生成一个 token，重新计算当前上下文。"""
import argparse
import torch
from kvforge import ModelConfig, MiniLLM
from kvforge.data import CharTokenizer


@torch.no_grad()
def generate(model, ids, max_new_tokens=100, temperature=0.8, top_k=20):
    if ids.ndim != 2 or ids.size(1) == 0:
        raise ValueError('提示词不能为空')
    if max_new_tokens < 0 or temperature < 0 or top_k < 0:
        raise ValueError('生成参数不能为负')
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            # 超长时仅使用最后 max_seq_len 个 token；此基线在窗口内重置位置。
            logits, _ = model(ids[:, -model.config.max_seq_len:])
            scores = logits[:, -1, :].float()
            scores[:, 0] = float('-inf')  # 不生成未知字符占位符。
            if temperature == 0:
                next_id = scores.argmax(-1, keepdim=True)
            else:
                scores = scores / temperature
                if top_k:
                    threshold = scores.topk(min(top_k, scores.size(-1))).values[:, -1:]
                    scores = scores.masked_fill(scores < threshold, float('-inf'))
                next_id = torch.multinomial(scores.softmax(-1), num_samples=1)
            ids = torch.cat((ids, next_id), dim=1)
        return ids
    finally:
        model.train(was_training)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--prompt', default='The ')
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--max-new-tokens', type=int, default=200)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top-k', type=int, default=20)
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
    output = generate(model, ids, args.max_new_tokens, args.temperature, args.top_k)
    print(tokenizer.decode(output[0].tolist()))


if __name__ == '__main__':
    main()
