"""
准备本地 UTF-8 文本：python prepare_data.py --input your_text.txt
"""
import argparse
from kvforge.data import prepare


def main():
    # 创建参数解析器,parser 负责定义允许哪些参数、读取终端输入、生成 --help、检查参数类型
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='你有权使用的 UTF-8 文本文件')
    # 定义输出目录,这是可选参数，如果没有提供 python prepare_data.py --input text.txt,默认输出到 data/prepared
    # 也可以覆盖：python prepare_data.py --input text.txt --output data/my_corpus
    parser.add_argument('--output', default='data/prepared')
    # 将终端输入的验证集比例由 str 转换成 float，默认值为 10%
    parser.add_argument('--val-fraction', type=float, default=0.1)
    # 正式解析参数
    args = parser.parse_args()
    # 执行 prepare():读取文本->切分->建词表->编码->保存文件->返回 metadata;然后打印出 metadata
    print(prepare(args.input, args.output, args.val_fraction))


if __name__ == '__main__':
    main()
