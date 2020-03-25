import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', dest='data_path', default='data/testset.txt', type=str, help='数据集')
parser.add_argument('--vocab_path', dest='vocab_path', default='data/vocab.txt', type=str, help='词汇表')
parser.add_argument('--save_path', dest='save_path', default='data/v_testset.txt', type=str, help='保存结果')
args = parser.parse_args()


def filter_by_vocab(args):
    vocab = []
    with open(args.vocab_path, 'r', encoding='utf8') as fr:
        for line in fr:
            vocab.append(line.strip())
    print(f'词汇表{len(vocab)}个')
    num = 0
    with open(args.data_path, 'r', encoding='utf8') as fr:
        with open(args.save_path, 'w', encoding='utf8') as fw:
            for line in fr:
                data = json.loads(line.strip())
                fp = True
                fr = True
                for w in data['post']:
                    if w not in vocab:
                        fp = False
                        break
                for w in data['response']:
                    if w not in vocab:
                        fr = False
                        break
                if fp and fr:
                    new_data = {'post': data['post'], 'response': data['response']}
                    fw.write(json.dumps(new_data, ensure_ascii=False)+'\n')
                    num += 1
    print(f'剩余{num}条数据')


if __name__ == '__main__':
    filter_by_vocab(args)
