import argparse
import os
import json


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', dest='data_path', default='../../ijcnlp_dailydialog/test/dialogues_test.txt', type=str, help='数据路径')
parser.add_argument('--save_path', dest='save_path', default='./result', type=str, help='保存路径')
parser.add_argument('--max_turn', dest='max_turn', default=8, type=int, help='历史最大轮数')
args = parser.parse_args()


def DD2ManyTurn(data_path, save_path, max_turn):
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

    for turn in range(1, max_turn+1):
        num = 0
        fr = open(data_path, 'r', encoding='utf8')
        for line in fr:
            utterances = line.strip().split('__eou__')[:-1]
            if len(utterances) > turn:
                for start_idx in range(len(utterances)-turn):
                    posts = []
                    for uidx in range(start_idx, start_idx+turn):
                        posts.append(utterances[uidx].strip().split())
                    response = utterances[start_idx+turn].strip().split()
                    data = {'posts': posts, 'response': response}
                    fw.write(json.dumps(data, ensure_ascii=False)+'\n')
                    num += 1
        fr.close()
        total_num += num
        print(f'{turn}轮对话共{num}个数据')
    print(f'共{total_num}个数据')
    fw.close()


if __name__ == '__main__':
    DD2ManyTurn(args.data_path, args.save_path, args.max_turn)
