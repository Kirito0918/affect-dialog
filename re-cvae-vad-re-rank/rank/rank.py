import argparse
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from seq2seq.model import Model
from seq2seq.util.config import Config
from seq2seq.util.sentence_processor import SentenceProcessor
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--samples_path', dest='samples_path', default='../result', type=str, help='采样的位置')
parser.add_argument('--vad_path', dest='vad_path', default='../data/vad.txt', type=str, help='vad字典的位置')
parser.add_argument('--seq2seq_path', dest='seq2seq_path', default='./log/020000001400100.model', type=str, help='seq2seq位置')
parser.add_argument('--result_path', dest='result_path', default='./result', type=str, help='结果位置')
parser.add_argument('--gpu', dest='gpu', default=True, type=bool, help='是否使用gpu')
args = parser.parse_args()


def main():
    config = Config()
    seq2seq = Model(config)
    seq2seq.eval()
    seq2seq.print_parameters()
    if os.path.isfile(args.seq2seq_path):
        _, _ = seq2seq.load_model(args.seq2seq_path)
        print('载入seq2seq模型完成!')
    else:
        print('请载入一个seq2seq模型！')
        return
    if args.gpu:
        seq2seq.to('cuda')

    file_readers = []  # 读取所有结果文件
    if os.path.isdir(args.samples_path):
        for root, dirs, files in os.walk(args.samples_path):
            for idx, file in enumerate(files):
                print(f'打开第{idx}个采样文件：{file}')
                file_readers.append(open(os.path.join(args.samples_path, file), 'r', encoding='utf8'))
        print(f'所有采样文件打开完毕，共打开{len(file_readers)}个文件！')
    else:
        print(f'{os.path.abspath(args.samples_path)}路径错误！')
        return

    results = []  # 将所有文件结果合并
    for fid, fr in enumerate(file_readers):
        for lid, line in enumerate(fr):
            data = json.loads(line.strip())
            if fid == 0:
                result = {'post': data['post'], 'response': data['response'], 'result': [data['result']]}
                results.append(result)
            else:
                results[lid]['result'].append(data['result'])
    print(f'共读取{len(results)}条数据！')
    for fr in file_readers:
        fr.close()

    vocab, vads = [], []  # 读取vad字典
    with open(args.vad_path, 'r', encoding='utf8') as fr:
        for line in fr:
            line = line.strip()
            word = line[: line.find(' ')]
            vad = line[line.find(' ') + 1:].split()
            vad = [float(item) for item in vad]
            vocab.append(word)
            vads.append(vad)
    print(f'载入词汇表: {len(vocab)}个')
    print(f'载入vad字典: {len(vads)}个')

    sp = SentenceProcessor(vocab, vads, config.pad_id, config.start_id, config.end_id, config.unk_id)

    if not os.path.exists(args.result_path):  # 创建结果文件夹
        os.makedirs(args.result_path)
    fw = open(os.path.join(args.result_path, 'result.txt'), 'w', encoding='utf8')
    fwd = open(os.path.join(args.result_path, 'detail.txt'), 'w', encoding='utf8')

    for result in tqdm(results):  # 对每个post的回复进行重排序
        str_post = result['post']  # [len]
        str_response = result['response']  # [len]
        str_results = result['result']  # [sample, len]
        sample_times = len(str_results)

        # 1. seq2seq给出语法流利的分数
        id_post, len_post = sp.word2index(str_post)
        id_post = [sp.start_id] + id_post + [sp.end_id]
        id_posts = [id_post for _ in range(sample_times)]  # [sample, len]
        len_posts = [len_post for _ in range(sample_times)]  # [sample]

        id_results, len_results = [], []
        for str_result in str_results:
            id_result, len_result = sp.word2index(str_result)
            id_results.append(id_result)
            len_results.append(len_result)

        len_posts = [l+2 for l in len_posts]  # 加上start和end
        len_results = [l+2 for l in len_results]

        max_len_results = max(len_results)
        id_results = [sp.pad_sentence(id_result, max_len_results) for id_result in id_results]  # 补齐

        feed_data = {'posts': id_posts, 'responses': id_results, 'len_posts': len_posts, 'len_responses': len_results}
        feed_data = prepare_feed_data(feed_data)
        output_vocab = seq2seq(feed_data, gpu=args.gpu)  # [sample, len_decoder, num_vocab]

        masks = feed_data['masks']  # [sample, len_decoder]
        labels = feed_data['responses'][:, 1:]  # [sample, len_decoder]
        token_per_batch = masks.sum(1)
        nll_loss = F.nll_loss(output_vocab.reshape(-1, config.num_vocab).clamp_min(1e-12).log(),
                              labels.reshape(-1), reduction='none') * masks.reshape(-1)
        nll_loss = nll_loss.reshape(sample_times, -1).sum(1)  # [sample]
        ppl = (nll_loss / token_per_batch.clamp_min(1e-12)).exp().cpu().detach().numpy()  # [sample]
        score_ppl = (ppl - ppl.min()) / (ppl.max() - ppl.min())  # [sample]

        # 语义
        embed_posts = torch.cat([seq2seq.embedding(feed_data['posts']), seq2seq.affect_embedding(feed_data['posts'])], 2)
        embed_responses = torch.cat([seq2seq.embedding(feed_data['responses']), seq2seq.affect_embedding(feed_data['responses'])], 2)
        embed_posts = embed_posts.sum(1) / feed_data['len_posts'].float().unsqueeze(1).clamp_min(1e-12)  # [sample, 303]
        embed_responses = embed_responses.sum(1) / feed_data['len_responses'].float().unsqueeze(1).clamp_min(1e-12)
        score_cos = torch.cosine_similarity(embed_posts, embed_responses, 1).cpu().detach().numpy()  # [sample]
        score_cos = (score_cos - score_cos.min()) / (score_cos.max() - score_cos.min())

        # 2. vad奖励分数
        vad_posts = np.array([sp.index2vad(id_post) for id_post in id_posts])[:, 1:]  # [sample, len, 3]
        vad_results = np.array([sp.index2vad(id_result) for id_result in id_results])[:, 1:]

        neutral_posts = np.tile(np.array([0.5, 0.5, 0.5]).reshape(1, 1, -1), (sample_times, vad_posts.shape[1], 1))
        neutral_results = np.tile(np.array([0.5, 0.5, 0.5]).reshape(1, 1, -1), (sample_times, vad_results.shape[1], 1))

        posts_mask = 1 - (vad_posts == neutral_posts).astype(np.float).prod(2)  # [sample, len]
        affect_posts = (vad_posts * np.expand_dims(posts_mask, 2)).sum(1) / posts_mask.sum(1).clip(1e-12).reshape(sample_times, 1)
        results_mask = 1 - (vad_results == neutral_results).astype(np.float).prod(2)  # [sample, len]
        affect_results = (vad_results * np.expand_dims(results_mask, 2)).sum(1) / results_mask.sum(1).clip(1e-12).reshape(sample_times, 1)

        post_v = affect_posts[:, 0]  # batch
        post_a = affect_posts[:, 1]
        post_d = affect_posts[:, 2]
        result_v = affect_results[:, 0]
        result_a = affect_results[:, 1]
        result_d = affect_results[:, 2]

        score_v = 1 - np.abs(post_v - result_v)  # [0, 1]
        score_a = np.abs(post_a - result_a)
        score_d = np.abs(post_d - result_d)
        score_vad = score_v + score_a + score_d
        baseline_score_vad = score_vad.mean()
        score_vad = score_vad - baseline_score_vad
        score_vad = (score_vad - score_vad.min()) / (score_vad.max() - score_vad.min())

        # 3. 情感分数
        # score_af = ((vad_results - neutral_results) ** 2).sum(2) ** 0.5  # [sample, len]
        # token_per_batch = token_per_batch.cpu().detach().numpy() - 1  # [sample]
        # score_af = score_af.sum(1) / token_per_batch.clip(1e-12)
        # score_af = (score_af - score_af.min()) / (score_af.max() - score_af.min())

        # 4. 句子长度
        # score_len = np.array([len(str_result) for str_result in str_results])  # [sample]
        # score_len = (score_len - score_len.min()) / (score_len.max() - score_len.min())

        score = 0.1*score_ppl + 0.4*score_vad + 0.5*score_cos
        output_id = score.argmax()

        output = {'post': str_post, 'response': str_response, 'result': str_results[output_id]}
        fw.write(json.dumps(output, ensure_ascii=False) + '\n')

        fwd.write('post: {}\n'.format(' '.join(str_post)))
        fwd.write('chosen response: {}\n'.format(' '.join(str_results[output_id])))
        fwd.write('response: {}\n'.format(' '.join(str_response)))
        for idx, str_result in enumerate(str_results):
            fwd.write('candidate{}: {} (t:{:.2f} p:{:.2f} v:{:.2f} c:{:.2f})\n'
                      .format(idx, ' '.join(str_result), score[idx], 0.1*score_ppl[idx], 0.4*score_vad[idx],
                              0.5*score_cos[idx]))
        fwd.write('\n')
    fw.close()
    fwd.close()


def prepare_feed_data(data):
    len_labels = torch.tensor([l - 1 for l in data['len_responses']]).long()  # [batch] 标签没有start_id，长度-1
    masks = (1 - F.one_hot(len_labels, len_labels.max() + 1).cumsum(1))[:, :-1]  # [batch, len_decoder]

    feed_data = {'posts': torch.tensor(data['posts']).long(),  # [batch, len_encoder]
                 'len_posts': torch.tensor(data['len_posts']).long(),  # [batch]
                 'responses': torch.tensor(data['responses']).long(),  # [batch, len_decoder]
                 'len_responses': torch.tensor(data['len_responses']).long(),  # [batch]
                 'masks': masks.float()}  # [batch, len_decoder]

    if args.gpu:
        for key, value in feed_data.items():
            feed_data[key] = value.cuda()

    return feed_data


if __name__ == '__main__':
    main()
