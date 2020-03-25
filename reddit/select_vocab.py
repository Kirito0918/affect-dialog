from collections import defaultdict
import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', dest='file_path', default='data/trainset.txt', type=str, help='输入需要统计词频的数据集')
parser.add_argument('--num_vocab', dest='num_vocab', default=25000, type=int, help='截取词汇表')
parser.add_argument('--save_path', dest='save_path', default='data/vocab.txt', type=str, help='词汇表路径')
args = parser.parse_args()


def select_vocab(args):
    vocab = defaultdict(int)
    with open(args.file_path, 'r', encoding='utf8') as fr:
        data_num = 0  # 统计样本总数
        post_len = 0  # 用于统计post平均长度
        response_len = 0  # 用于统计response平均长度

        for line in fr:
            data_num += 1
            data = json.loads(line)

            post = data['post']
            response = data['response']

            post_len += len(post)
            response_len += len(response)

            for word in post:
                vocab[word] += 1
            for word in response:
                vocab[word] += 1

    vocab = dict(sorted(vocab.items(), key=lambda x: x[1], reverse=True))  # 词频降序排列

    print(f'数据集位置: {os.path.abspath(args.file_path)}')
    print(f'样本总数: {data_num}', end=', ')
    print('post平均长度: {:.2f}'.format(post_len / data_num), end=', ')
    print('response平均长度: {:.2f}'.format(response_len / data_num), end=', ')
    print(f'包含词汇总数: {len(vocab)}')

    vocab = list(vocab.keys())[:args.num_vocab]
    with open(args.save_path, 'w', encoding='utf8') as fw:
        for v in vocab:
            fw.write(v+'\n')


if __name__ == '__main__':
    select_vocab(args)
