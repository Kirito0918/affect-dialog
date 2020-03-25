import argparse
import json
import torch
from model.util.config import Config
from model.util.sentence_processor import SentenceProcessor
from model.model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', dest='data_path', default='data/v_testset.txt', type=str, help='数据集')
parser.add_argument('--model_path', dest='model_path', default='model/log/006000000008826.model', type=str, help='模型')
parser.add_argument('--save_path', dest='save_path', default='data/e_testset.txt', type=str, help='保存结果')
parser.add_argument('--embed_path', dest='embed_path', default='data/embed.txt', type=str, help='词向量位置')
parser.add_argument('--vad_path', dest='vad_path', default='data/vad.txt', type=str, help='vad位置')
parser.add_argument('--gpu', dest='gpu', default=True, type=bool, help='是否使用gpu')
args = parser.parse_args()
config = Config()


def filter_by_emotion(args):
    vocab, embeds = [], []
    with open(args.embed_path, 'r', encoding='utf8') as fr:
        for line in fr:
            line = line.strip()
            word = line[: line.find(' ')]
            vec = line[line.find(' ') + 1:].split()
            embed = [float(v) for v in vec]
            assert len(embed) == config.embedding_size  # 检测词向量维度
            vocab.append(word)
            embeds.append(embed)
    print(f'载入词汇表: {len(vocab)}个')
    print(f'词向量维度: {config.embedding_size}')
    vads = []
    with open(args.vad_path, 'r', encoding='utf8') as fr:
        for line in fr:
            line = line.strip()
            vad = line[line.find(' ') + 1:].split()
            vad = [float(item) for item in vad]
            assert len(vad) == config.affect_embedding_size
            vads.append(vad)
    print(f'载入vad字典: {len(vads)}个')
    print(f'vad维度: {config.affect_embedding_size}')
    sentence_processor = SentenceProcessor(vocab, config.pad_id, config.start_id, config.end_id, config.unk_id)
    model = Model(config)
    if args.gpu:
        model.cuda()
    num = 0
    with open(args.data_path, 'r', encoding='utf8') as fr:
        with open(args.save_path, 'w', encoding='utf8') as fw:
            for line in fr:
                data = json.loads(line.strip())
                post = data['post']
                response = data['response']
                id_post, len_post = sentence_processor.word2index(post)
                id_response, len_response = sentence_processor.word2index(response)
                max_len = max(len_post, len_response) + 2
                id_post = sentence_processor.pad_sentence(id_post, max_len)
                id_response = sentence_processor.pad_sentence(id_response, max_len)
                texts = [id_post, id_response]
                lengths = [len_post+2, len_response+2]
                feed_data = {'x': torch.tensor(texts).long(),
                             'len_x': torch.tensor(lengths).long()}
                if args.gpu:
                    for key, value in feed_data.items():
                        feed_data[key] = value.cuda()
                result = model(feed_data).argmax(1).detach().tolist()
                if sum(result) != 0:
                    new_data = {'post': post, 'post_e': result[0],
                                'response': response, 'response_e': result[1]}
                    fw.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                    num += 1
    print(f'剩余{num}条数据')


if __name__ == '__main__':
    filter_by_emotion(args)
