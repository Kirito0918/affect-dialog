from model.util.config import Config
from model.model import Model
from model.Optim import Optim
from model.util.sentence_processor import SentenceProcessor
from model.util.data_processor import DataProcessor
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
import argparse
import json
import os
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--trainset_path', dest='trainset_path', default='data/raw/newtrainset.txt', type=str, help='训练集位置')
parser.add_argument('--validset_path', dest='validset_path', default='data/raw/newvalidset.txt', type=str, help='验证集位置')
parser.add_argument('--testset_path', dest='testset_path', default='data/raw/newtestset.txt', type=str, help='测试集位置')
parser.add_argument('--embed_path', dest='embed_path', default='data/embed.txt', type=str, help='词向量位置')
parser.add_argument('--vad_path', dest='vad_path', default='data/vad.txt', type=str, help='vad位置')
parser.add_argument('--result_path', dest='result_path', default='result', type=str, help='测试结果位置')
parser.add_argument('--print_per_step', dest='print_per_step', default=100, type=int, help='每更新多少次参数summary学习情况')
parser.add_argument('--log_per_step', dest='log_per_step', default=20000, type=int, help='每更新多少次参数保存模型')
parser.add_argument('--log_path', dest='log_path', default='log', type=str, help='记录模型位置')
parser.add_argument('--inference', dest='inference', default=True, type=bool, help='是否测试')  #
parser.add_argument('--max_len', dest='max_len', default=60, type=int, help='测试时最大解码步数')
parser.add_argument('--model_path', dest='model_path', default='log/run1587815462/015000001115000.model', type=str, help='载入模型位置')  #
parser.add_argument('--seed', dest='seed', default=666, type=int, help='随机种子')  #
parser.add_argument('--gpu', dest='gpu', default=True, type=bool, help='是否使用gpu')  #
parser.add_argument('--max_epoch', dest='max_epoch', default=16, type=int, help='最大训练epoch')

args = parser.parse_args()  # 程序运行参数

config = Config()  # 模型配置

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)


def main():
    trainset, validset, testset = [], [], []
    if args.inference:  # 测试时只载入测试集
        with open(args.testset_path, 'r', encoding='utf8') as fr:
            for line in fr:
                testset.append(json.loads(line))
        print(f'载入测试集{len(testset)}条')
    else:  # 训练时载入训练集和验证集
        with open(args.trainset_path, 'r', encoding='utf8') as fr:
            for line in fr:
                trainset.append(json.loads(line))
        print(f'载入训练集{len(trainset)}条')
        with open(args.validset_path, 'r', encoding='utf8') as fr:
            for line in fr:
                validset.append(json.loads(line))
        print(f'载入验证集{len(validset)}条')

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

    # 通过词汇表构建一个word2index和index2word的工具
    sentence_processor = SentenceProcessor(vocab, config.pad_id, config.start_id, config.end_id, config.unk_id)

    model = Model(config)
    model.print_parameters()  # 输出模型参数个数
    epoch = 0  # 训练集迭代次数
    global_step = 0  # 参数更新次数

    # 载入模型
    if os.path.isfile(args.model_path):  # 如果载入模型的位置存在则载入模型
        epoch, global_step = model.load_model(args.model_path)
        model.affect_embedding.embedding.weight.requires_grad = False
        print('载入模型完成')
        log_dir = os.path.split(args.model_path)[0]
    elif args.inference:  # 如果载入模型的位置不存在，但是又要测试，这是没有意义的
        print('请测试一个训练过的模型!')
        return
    else:  # 如果载入模型的位置不存在，重新开始训练，则载入预训练的词向量
        model.embedding.embedding.weight = torch.nn.Parameter(torch.FloatTensor(embeds))
        model.affect_embedding.embedding.weight = torch.nn.Parameter(torch.FloatTensor(vads))
        model.affect_embedding.embedding.weight.requires_grad = False
        print('初始化模型完成')
        log_dir = os.path.join(args.log_path, 'run' + str(int(time.time())))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    if args.gpu:
        model.to('cuda')  # 将模型参数转到gpu

    # 定义优化器参数
    optim = Optim(config.method, config.lr, config.lr_decay, config.weight_decay, config.eps, config.max_grad_norm)
    optim.set_parameters(model.parameters())  # 给优化器设置参数
    optim.update_lr(epoch)  # 每个epoch更新学习率

    # 训练
    if not args.inference:
        summary_writer = SummaryWriter(os.path.join(log_dir, 'summary'))  # 创建tensorboard记录的文件夹
        dp_train = DataProcessor(trainset, config.batch_size, sentence_processor)  # 数据的迭代器
        dp_valid = DataProcessor(validset, config.batch_size, sentence_processor, shuffle=False)

        while epoch < args.max_epoch:  # 最大训练轮数
            model.train()  # 切换到训练模式
            for data in dp_train.get_batch_data():
                start_time = time.time()
                feed_data = prepare_feed_data(data)
                loss, nll_loss, kld_loss, kld_weight, bow_loss, ppl = train(model, feed_data, global_step)
                loss.mean().backward()  # 反向传播
                optim.step()  # 更新参数
                optim.optimizer.zero_grad()  # 清空梯度
                use_time = time.time() - start_time

                global_step += 1  # 参数更新次数+1
                if global_step % args.print_per_step == 0:
                    print('epoch: {:d}, global_step: {:d}, lr: {:g}, nll_loss: {:.2f}, kld_loss: {:.2f}, '
                          'kld_weight: {:g}, bow_loss: {:.2f}, ppl: {:.2f}, time: {:.2f}s/step'
                          .format(epoch, global_step, optim.lr, nll_loss.mean().item(), kld_loss.mean().item(),
                                  kld_weight, bow_loss.mean().item(), ppl.mean().exp().item(), use_time))
                    summary_writer.add_scalar('train_nll', nll_loss.mean().item(), global_step)
                    summary_writer.add_scalar('train_kld', kld_loss.mean().item(), global_step)
                    summary_writer.add_scalar('train_weight', kld_weight, global_step)
                    summary_writer.add_scalar('train_bow', bow_loss.mean().item(), global_step)
                    summary_writer.add_scalar('train_ppl', ppl.mean().exp().item(), global_step)
                    summary_writer.flush()  # 将缓冲区写入文件

                if global_step % args.log_per_step == 0 \
                        or (global_step % (2*config.kl_step) - config.kl_step) == config.kl_step // 2:
                    log_file = os.path.join(log_dir, '{:03d}{:012d}.model'.format(epoch, global_step))
                    model.save_model(epoch, global_step, log_file)

                    model.eval()
                    nll_loss, kld_loss, bow_loss, ppl = valid(model, dp_valid, global_step-1)
                    model.train()
                    print('在验证集上的NLL损失为: {:g}, KL损失为: {:g}, BOW损失为: {:g}, PPL为: {:g}'
                          .format(nll_loss, kld_loss, bow_loss, np.exp(ppl)))
                    summary_writer.add_scalar('valid_nll', nll_loss, global_step)
                    summary_writer.add_scalar('valid_kld', kld_loss, global_step)
                    summary_writer.add_scalar('valid_bow', bow_loss, global_step)
                    summary_writer.add_scalar('valid_ppl', np.exp(ppl), global_step)
                    summary_writer.flush()  # 将缓冲区写入文件

            epoch += 1  # 数据集迭代次数+1
            optim.update_lr(epoch)  # 调整学习率

            log_file = os.path.join(log_dir, '{:03d}{:012d}.model'.format(epoch, global_step))
            model.save_model(epoch, global_step, log_file)

            model.eval()
            nll_loss, kld_loss, bow_loss, ppl = valid(model, dp_valid, global_step-1)
            print('在验证集上的NLL损失为: {:g}, KL损失为: {:g}, BOW损失为: {:g}, PPL为: {:g}'
                  .format(nll_loss, kld_loss, bow_loss, np.exp(ppl)))
            summary_writer.add_scalar('valid_nll', nll_loss, global_step)
            summary_writer.add_scalar('valid_kld', kld_loss, global_step)
            summary_writer.add_scalar('valid_bow', bow_loss, global_step)
            summary_writer.add_scalar('valid_ppl', np.exp(ppl), global_step)
            summary_writer.flush()  # 将缓冲区写入文件

        summary_writer.close()
    else:  # 测试
        if not os.path.exists(args.result_path):  # 创建结果文件夹
            os.makedirs(args.result_path)

        result_file = os.path.join(args.result_path, '{:03d}{:012d}.txt'.format(epoch, global_step))  # 命名结果文件
        fw = open(result_file, 'w', encoding='utf8')
        dp_test = DataProcessor(testset, config.batch_size, sentence_processor, shuffle=False)

        model.eval()  # 切换到测试模式，会停用dropout等等
        nll_loss, kld_loss, bow_loss, ppl = valid(model, dp_test, global_step-1)
        print('在测试集上的NLL损失为: {:g}, KL损失为: {:g}, BOW损失为: {:g}, PPL为: {:g}'
              .format(nll_loss, kld_loss, bow_loss, np.exp(ppl)))

        len_results = []  # 统计生成结果的总长度
        for data in dp_test.get_batch_data():
            posts = data['str_posts']
            responses = data['str_responses']
            feed_data = prepare_feed_data(data, inference=True)
            results = test(model, feed_data)  # 使用模型计算结果 [batch, len_decoder]

            for idx, result in enumerate(results):
                new_data = dict()
                new_data['post'] = posts[idx]
                new_data['response'] = responses[idx]
                new_data['result'] = sentence_processor.index2word(result)  # 将输出的句子转回单词的形式
                len_results.append(len(new_data['result']))
                fw.write(json.dumps(new_data, ensure_ascii=False) + '\n')

        fw.close()
        print(f'生成句子平均长度: {1.0 * sum(len_results) / len(len_results)}')


def prepare_feed_data(data, inference=False):
    len_labels = torch.tensor([l - 1 for l in data['len_responses']]).long()  # [batch] 标签没有start_id，长度-1
    masks = (1 - F.one_hot(len_labels, len_labels.max() + 1).cumsum(1))[:, :-1]  # [batch, len_decoder]
    batch_size = masks.size(0)

    if not inference:  # 训练时的输入
        feed_data = {'posts': torch.tensor(data['posts']).long(),  # [batch, len_encoder]
                     'len_posts': torch.tensor(data['len_posts']).long(),  # [batch]
                     'responses': torch.tensor(data['responses']).long(),  # [batch, len_decoder]
                     'len_responses': torch.tensor(data['len_responses']).long(),  # [batch]
                     'sampled_latents': torch.randn((batch_size, config.latent_size)),  # [batch, latent_size]
                     'masks': masks.float()}  # [batch, len_decoder]
    else:  # 测试时的输入
        feed_data = {'posts': torch.tensor(data['posts']).long(),
                     'len_posts': torch.tensor(data['len_posts']).long(),
                     'sampled_latents': torch.randn((batch_size, config.latent_size))}

    if args.gpu:  # 将数据转移到gpu上
        for key, value in feed_data.items():
            feed_data[key] = value.cuda()

    return feed_data


def compute_loss(outputs, labels, masks, global_step):
    def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):  # [batch, latent]
        """ 两个高斯分布之间的kl散度公式 """
        kld = 0.5 * torch.sum(prior_logvar - recog_logvar - 1
                              + recog_logvar.exp() / prior_logvar.exp()
                              + (prior_mu - recog_mu).pow(2) / prior_logvar.exp(), 1)
        return kld  # [batch]

    # output_vocab: [batch, len_decoder, num_vocab] 对每个单词的softmax概率
    output_vocab, bow_predict, output_affect, _mu, _logvar, mu, logvar = outputs
    labels_word, labels_affect = labels

    token_per_batch = masks.sum(1)  # 每个样本要计算损失的token数 [batch]
    batch_size, len_decoder = masks.size()

    output_vocab = output_vocab.reshape(-1, config.num_vocab)  # [batch*len_decoder, num_vocab]
    labels_word = labels_word.reshape(-1)  # [batch*len_decoder]
    masks = masks.reshape(-1)  # [batch*len_decoder]

    # nll_loss需要自己求log，它只是把label指定下标的损失取负并拿出来，reduction='none'代表只是拿出来，而不需要求和或者求均值
    _nll_loss = F.nll_loss(output_vocab.clamp_min(1e-12).log(), labels_word, reduction='none')  # [batch*len_decoder]
    _nll_loss = _nll_loss * masks  # 忽略掉不需要计算损失的token [batch*len_decoder]

    nll_loss = _nll_loss.reshape(-1, len_decoder).sum(1)  # 每个batch的nll损失 [batch]

    ppl = nll_loss / token_per_batch.clamp_min(1e-12)  # ppl的计算需要平均到每个有效的token上 [batch]

    # kl散度损失 [batch]
    kld_loss = gaussian_kld(mu, logvar, _mu, _logvar)

    # kl退火
    kld_weight = min(1.0 * (global_step % (2*config.kl_step)) / config.kl_step, 1)  # 周期性退火

    # bow损失
    labels_word = labels_word.reshape([-1, len_decoder])  # [batch, len_decoder]
    masks = masks.reshape([-1, len_decoder])
    bow_loss = -bow_predict.gather(1, labels_word).clamp_min(1e-12).log()  # [batch, len_decoder]
    bow_loss = (bow_loss * masks).sum(1)  # [batch]

    # 损失
    loss = nll_loss + kld_weight * kld_loss + bow_loss

    return loss, nll_loss, kld_loss, kld_weight, bow_loss, ppl


def train(model, feed_data, global_step):
    output_vocab, bow_predict, _mu, _logvar, mu, logvar = model(feed_data, gpu=args.gpu)
    with torch.no_grad():
        output_affect = model.affect_embedding(output_vocab.argmax(2))
        labels_affect = model.affect_embedding(feed_data['posts'][:, 1:])  # 去掉start id
    outputs = (output_vocab, bow_predict, output_affect, _mu, _logvar, mu, logvar)
    labels_word = feed_data['responses'][:, 1:]  # 去掉start_id
    labels = (labels_word, labels_affect)
    masks = feed_data['masks']
    loss, nll_loss, kld_loss, kld_weight, bow_loss, ppl = compute_loss(outputs, labels, masks, global_step)
    return loss, nll_loss, kld_loss, kld_weight, bow_loss, ppl


def valid(model, data_processor, global_step):
    nll_losses, kld_losses, bow_losses, ppls = [], [], [], []
    for data in data_processor.get_batch_data():
        feed_data = prepare_feed_data(data)
        output_vocab, bow_predict, _mu, _logvar, mu, logvar = model(feed_data, gpu=args.gpu)
        with torch.no_grad():
            output_affect = model.affect_embedding(output_vocab.argmax(2))
            labels_affect = model.affect_embedding(feed_data['posts'][:, 1:])
        outputs = (output_vocab, bow_predict, output_affect, _mu, _logvar, mu, logvar)
        labels_word = feed_data['responses'][:, 1:]  # 去掉start_id
        labels = (labels_word, labels_affect)
        masks = feed_data['masks']

        _, nll_loss, kld_loss, kld_weight, bow_loss, ppl = compute_loss(outputs, labels, masks, global_step)
        nll_losses.extend(nll_loss.detach().tolist())
        kld_losses.extend(kld_loss.detach().tolist())
        bow_losses.extend(bow_loss.detach().tolist())
        ppls.extend(ppl.detach().tolist())

    nll_losses = np.array(nll_losses)
    kld_losses = np.array(kld_losses) * kld_weight
    bow_losses = np.array(bow_losses)
    ppls = np.array(ppls)

    return nll_losses.mean(), kld_losses.mean(), bow_losses.mean(), ppls.mean()


def test(model, feed_data):
    output_vocab, _, _, _, _, _ = model(feed_data, inference=True, max_len=args.max_len, gpu=args.gpu)
    return output_vocab.argmax(2).detach().tolist()


if __name__ == '__main__':
    main()
