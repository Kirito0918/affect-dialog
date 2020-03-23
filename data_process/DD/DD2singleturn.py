import argparse
import os
import json


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', dest='data_path', default='../../ijcnlp_dailydialog/test/dialogues_test.txt', type=str, help='数据路径')
parser.add_argument('--save_path', dest='save_path', default='./result', type=str, help='保存路径')
args = parser.parse_args()


def DD2SingleTurn(data_path, save_path):
    if not os.path.isfile(data_path):
        print('请输入正确的数据路径！')
        return
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    data_name = os.path.basename(data_path).split('.')
    save_name = data_name[0] + '_singleturn.' + data_name[1]
    save_path = os.path.join(save_path, save_name)
    num = 0

    with open(data_path, 'r', encoding='utf8') as fr:
        with open(save_path, 'w', encoding='utf8') as fw:
            for line in fr:
                utterances = line.strip().split('__eou__')[:-1]
                for idx in range(len(utterances)-1):
                    data = {'post': utterances[idx].strip().split(), 'response': utterances[idx+1].strip().split()}
                    fw.write(json.dumps(data, ensure_ascii=False)+'\n')
                    num += 1
    print(f'共处理了{num}个数据')


if __name__ == '__main__':
    DD2SingleTurn(args.data_path, args.save_path)
