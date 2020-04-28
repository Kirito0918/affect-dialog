import argparse
import os
import json


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', dest='data_path', default='../../ijcnlp_dailydialog/test/dialogues_test.txt', type=str, help='数据路径')
parser.add_argument('--save_path', dest='save_path', default='./result', type=str, help='保存路径')
args = parser.parse_args()


def dd4_many_turn(data_path, save_path):
    if not os.path.isfile(data_path):
        print('请输入正确的数据路径！')
        return
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    data_name = os.path.basename(data_path).split('.')
    save_name = data_name[0] + '_manyturns.' + data_name[1]
    save_path = os.path.join(save_path, save_name)

    fw = open(save_path, 'w', encoding='utf8')
    total_num = 0

    with open(data_path, 'r', encoding='utf8') as fr:
        for line in fr:
            utterances = line.strip().split('__eou__')[:-1]
            for ed in range(1, len(utterances)):
                posts = []
                for idx in range(ed):
                    posts.append(utterances[idx].strip().split())
                response = utterances[idx+1].strip().split()
                data = {'posts': posts, 'response': response}
                fw.write(json.dumps(data, ensure_ascii=False) + '\n')
                total_num += 1
    print(f'共{total_num}个数据')
    fw.close()


if __name__ == '__main__':
    dd4_many_turn(args.data_path, args.save_path)
