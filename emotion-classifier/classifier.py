import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import argparse


parser = argparse.ArgumentParser()
pad_id = 0
    start_id = 1
    end_id = 2
    unk_id = 3
affect_embedding_size = 3
parser.add_argument('--num_vocab', dest='num_vocab', default=25000, type=int, help='词汇表大小')
parser.add_argument('--embedding_size', dest='embedding_size', default=128, type=int, help='词嵌入维度')
parser.add_argument('--cell_type', dest='cell_type', default='GRU', type=str, help='编码器类型')
parser.add_argument('--output_size', dest='output_size', default=128, type=int, help='编码器输出维度')
parser.add_argument('--num_layers', dest='num_layers', default=1, type=int, help='编码器层数')
parser.add_argument('--bidirectional', dest='bidirectional', default=True, type=bool, help='编码器是否双向')
parser.add_argument('--dropout', dest='dropout', default=0.1, type=float, help='dropout')
parser.add_argument('--gpu', dest='gpu', default=False, type=bool, help='是否使用gpu')
args = parser.parse_args()


class Encoder(nn.Module):
    r""" 编码器 """
    def __init__(self, cell_type,  # rnn类型
                 input_size,  # 输入维度
                 output_size,  # 输出维度
                 num_layers,  # rnn层数
                 bidirectional=False,  # 是否双向
                 dropout=0.1):  # dropout
        super(Encoder, self).__init__()
        assert cell_type in ['GRU', 'LSTM']  # 限定rnn类型

        if bidirectional:  # 如果双向
            assert output_size % 2 == 0
            cell_size = output_size // 2  # rnn维度
        else:
            cell_size = output_size

        self.bidirectional = bidirectional
        self.cell_type = cell_type
        self.rnn_cell = getattr(nn, cell_type)(input_size=input_size,
                                               hidden_size=cell_size,
                                               num_layers=num_layers,
                                               bidirectional=bidirectional,
                                               dropout=dropout)

    def forward(self, x,  # [seq, batch, dim]
                length):  # [batch]
        x = pack_padded_sequence(x, length, enforce_sorted=False)

        # output: [seq, batch, dim*directions] 每个时间步的输出
        # final_state = [layers*directions, batch, dim] 每一层的最终状态
        output, final_state = self.rnn_cell(x)
        output = pad_packed_sequence(output)[0]

        if self.bidirectional:  # 如果是双向的，对双向进行拼接作为每层的最终状态
            if self.cell_type == 'GRU':
                final_state_forward = final_state[0::2, :, :]  # [layers, batch, dim]
                final_state_back = final_state[1::2, :, :]  # [layers, batch, dim]
                final_state = torch.cat([final_state_forward, final_state_back], 2)  # [layers, batch, dim*2]
            else:
                final_state_h, final_state_c = final_state
                final_state_h = torch.cat([final_state_h[0::2, :, :], final_state_h[1::2, :, :]], 2)
                final_state_c = torch.cat([final_state_c[0::2, :, :], final_state_c[1::2, :, :]], 2)
                final_state = (final_state_h, final_state_c)

        # output = [seq, batch, dim]
        # final_state = [layers, batch, dim]
        return output, final_state


class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        assert config.cell_type in ['GRU', 'LSTM']  # 限定rnn类型
        self.embedding = nn.Embedding(config.num_vocab, config.embedding_size, 0)
        self.encoder = Encoder(config.cell_type, config.embedding_size, config.output_size,
                               config.num_layers, config.bidirectional, config.dropout)
        self.classifier = nn.Sequential(nn.Linear(config.output_size, config.output_size//2),
                                        nn.Linear(config.output_size//2, config.num_classification),
                                        nn.Softmax(-1))

    def forward(self, inputs):
        x = inputs['x']  # [batch, len]
        len_x = inputs['len_x']  # [batch]

        embed_x = self.embedding(x)  # [batch, len, embed]
        # state: [layers, batch, dim]
        _, state = self.encoder(embed_x.transpose(0, 1), len_x)
        if isinstance(state, tuple):
            state = state[0]
        context = state[-1, :, :]  # [batch, dim]

        output = self.classifier(context)  # [batch, 7]
        return output


