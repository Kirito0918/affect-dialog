import argparse
import os
import json
import random


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', dest='data_path', default='result/SentimentClassification.txt', type=str, help='数据路径')
parser.add_argument('--save_path', dest='save_path', default='./result', type=str, help='保存路径')
args = parser.parse_args()


def filter_emotion_dataset(data_path, save_path):
    if not os.path.isfile(data_path):
        print('请输入正确的数据路径！')
        return
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    train_path = os.path.join(save_path, 'sc_train.txt')
    valid_path = os.path.join(save_path, 'sc_valid.txt')
    test_path = os.path.join(save_path, 'sc_test.txt')

    dataset = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    with open(data_path, 'r', encoding='utf8') as fr:
        for line in fr:
            data = json.loads(line.strip())
            dataset[data['emotion']].append(data)
    print('情感分布:', [len(_) for _ in dataset.values()])
    random.shuffle(dataset[0])
    dataset[0] = dataset[0][: 12000]
    total_num = 0
    for emo in dataset.values():
        total_num += len(emo)
    print(f'筛选后样本总数:{total_num}')
    print(f'筛选后情感分布:', [len(emo) for emo in dataset.values()])
    print(f'筛选后情感占比:', ['{:.2%}'.format(len(emo)/total_num) for emo in dataset.values()])

    num_test = num_valid = int(total_num * 0.1)
    num_train = total_num - num_valid - num_test
    new_dataset = []
    for data in dataset.values():
        new_dataset.extend(data)
    random.shuffle(new_dataset)
    trainset = new_dataset[: num_train]
    validset = new_dataset[num_train: num_train+num_test]
    testset = new_dataset[-num_test:]
    print(f'训练集大小:{len(trainset)}, 验证集大小:{len(validset)}, 测试集大小:{len(testset)}')
    with open(train_path, 'w', encoding='utf8') as fw:
        for data in trainset:
            fw.write(json.dumps(data, ensure_ascii=False)+'\n')
    with open(valid_path, 'w', encoding='utf8') as fw:
        for data in validset:
            fw.write(json.dumps(data, ensure_ascii=False)+'\n')
    with open(test_path, 'w', encoding='utf8') as fw:
        for data in testset:
            fw.write(json.dumps(data, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    filter_emotion_dataset(args.data_path, args.save_path)
