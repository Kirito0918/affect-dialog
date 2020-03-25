import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', dest='train_path', default='data/e_trainset.txt', type=str, help='训练集')
parser.add_argument('--valid_path', dest='valid_path', default='data/e_validset.txt', type=str, help='验证集')
parser.add_argument('--test_path', dest='test_path', default='data/e_testset.txt', type=str, help='测试集')
parser.add_argument('--save_path', dest='save_path', default='data', type=str, help='保存结果')
args = parser.parse_args()


def get_new_dataset(args):
    validset = []
    with open(args.valid_path, 'r', encoding='utf8') as fr:
        for line in fr:
            validset.append(json.loads(line.strip()))
    testset = []
    with open(args.test_path, 'r', encoding='utf8') as fr:
        for line in fr:
            testset.append(json.loads(line.strip()))
    dataset = validset + testset
    new_testset = dataset[-5000:]
    new_validset = dataset[-10000: -5000]
    left = dataset[:-10000]
    with open(os.path.join(args.save_path, 'newtestset.txt'), 'w', encoding='utf8') as fw:
        for data in new_testset:
            new_data = {'post': data['post'], 'response': data['response']}
            fw.write(json.dumps(new_data, ensure_ascii=False)+'\n')
    with open(os.path.join(args.save_path, 'newvalidset.txt'), 'w', encoding='utf8') as fw:
        for data in new_validset:
            new_data = {'post': data['post'], 'response': data['response']}
            fw.write(json.dumps(new_data, ensure_ascii=False)+'\n')
    print(f'写入测试集{len(new_testset)}条')
    print(f'写入验证集{len(new_validset)}条')
    num_train = 0
    with open(args.train_path, 'r', encoding='utf8') as fr:
        with open(os.path.join(args.save_path, 'newtrainset.txt'), 'w', encoding='utf8') as fw:
            for line in fr:
                data = json.loads(line.strip())
                new_data = {'post': data['post'], 'response': data['response']}
                fw.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                num_train += 1
            for data in left:
                new_data = {'post': data['post'], 'response': data['response']}
                fw.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                num_train += 1
    print(f'写入训练集{num_train}条')


if __name__ == '__main__':
    get_new_dataset(args)
