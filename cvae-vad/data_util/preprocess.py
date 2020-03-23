import regex as re
import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', dest='train_path', default='../data/raw/trainset.txt', type=str, help='训练集位置')
parser.add_argument('--valid_path', dest='valid_path', default='../data/raw/validset.txt', type=str, help='验证集位置')
parser.add_argument('--test_path', dest='test_path', default='../data/raw/testset.txt', type=str, help='测试集位置')
parser.add_argument('--write_train_path', dest='write_train_path', default='../data/trainset_a.txt', type=str, help='训练集位置')
parser.add_argument('--write_valid_path', dest='write_valid_path', default='../data/validset_a.txt', type=str, help='验证集位置')
parser.add_argument('--write_test_path', dest='write_test_path', default='../data/testset_a.txt', type=str, help='测试集位置')
args = parser.parse_args()


def preprocess(read_path, write_path):
    r""" 将一些数据集中一些数字转化成<num>，用于删减词汇表 """
    dataset = []
    with open(read_path, 'r', encoding='utf8') as fr:
        for line in fr:
            line = line.strip()
            dataset.append(json.loads(line))
    regex = re.compile(r'[0-9]+')
    for data in tqdm(dataset):
        post = data['post']
        response = data['response']
        for idx, word in enumerate(post):
            if regex.search(word):
                data['post'][idx] = '<num>'
        for idx, word in enumerate(response):
            if regex.search(word):
                data['response'][idx] = '<num>'
    with open(write_path, 'w', encoding='utf8') as fw:
        for data in dataset:
            fw.write(json.dumps(data)+'\n')


if __name__ == '__main__':
    preprocess(args.train_path, args.write_train_path)
    preprocess(args.valid_path, args.write_valid_path)
    preprocess(args.test_path, args.write_test_path)
