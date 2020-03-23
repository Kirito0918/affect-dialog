import torch
import torch.nn as nn
import torch.nn.functional as F
from Embedding import Embedding
from Encoder import Encoder
from Decoder import Decoder
from Attention import Attention


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        assert config.attention_size == config.encoder_decoder_output_size
        self.config = config

        # 定义嵌入层
        self.embedding = Embedding(config.num_vocab,  # 词汇表大小
                                   config.embedding_size,  # 嵌入层维度
                                   config.pad_id,
                                   config.dropout)

        # 编码器
        self.encoder = Encoder(config.encoder_decoder_cell_type,  # rnn类型
                               config.embedding_size,  # 输入维度
                               config.encoder_decoder_output_size,  # 输出维度
                               config.encoder_decoder_num_layers,  # rnn层数
                               config.encoder_bidirectional,  # 是否双向
                               config.dropout)  # dropout概率

        # 解码器
        self.decoder = Decoder(config.encoder_decoder_cell_type,  # rnn类型
                               config.embedding_size+config.attention_size+config.encoder_decoder_output_size,
                               config.encoder_decoder_output_size,  # 输出维度
                               config.encoder_decoder_num_layers,  # rnn层数
                               config.dropout)  # dropout概率

        # 注意力
        self.attention = Attention(config.encoder_decoder_output_size,
                                   config.encoder_decoder_output_size,
                                   config.attention_type,
                                   config.attention_size,
                                   config.dropout)

        # 输出层
        self.projector = nn.Sequential(
            nn.Linear(config.encoder_decoder_output_size, config.num_vocab),
            nn.Softmax(-1)
        )

    def forward(self, inputs, inference=False, max_len=60, gpu=True):
        if not inference:  # 训练
            id_posts = inputs['posts']  # [batch, seq]
            len_posts = inputs['len_posts']  # [batch]
            id_responses = inputs['responses']  # [batch, seq]
            batch_size = id_posts.size(0)
            len_decoder = id_responses.size(1) - 1

            embed_posts = self.embedding(id_posts)  # [batch, seq, embed_size]
            embed_responses = self.embedding(id_responses)  # [batch, seq, embed_size]

            # encoder_output: [seq, batch, dim]
            # state: [layers, batch, dim]
            encoder_output, state_encoder = self.encoder(embed_posts.transpose(0, 1), len_posts)
            encoder_output = encoder_output.transpose(0, 1)
            attn_masks = (1 - F.one_hot(len_posts, len_posts.max() + 1).cumsum(1))[:, :-1].bool()  # [batch, len]
            if isinstance(state_encoder, tuple):
                context = state_encoder[0]
            context = context[-1, :, :].unsqueeze(0)  # [1, batch, dim]

            # 解码器的输入为回复去掉end_id
            decoder_inputs = embed_responses[:, :-1, :].transpose(0, 1)  # [seq-1, batch, embed_size]
            decoder_inputs = decoder_inputs.split([1] * len_decoder, 0)
            init_attn = torch.zeros([1, batch_size, self.config.attention_size])
            if gpu:
                init_attn = init_attn.cuda()

            outputs = []
            for idx in range(len_decoder):
                if idx == 0:
                    state = state_encoder  # 解码器初始状态
                    decoder_input = torch.cat([decoder_inputs[idx], init_attn, context], 2)
                else:
                    decoder_input = torch.cat([decoder_inputs[idx], attn, context], 2)
                # output: [1, batch, dim_out]
                # state: [num_layer, batch, dim_out]
                output, state = self.decoder(decoder_input, state)
                # attn: [batch, 1, attention_size]
                attn, _ = self.attention(output.transpose(0, 1), encoder_output, attn_masks)
                attn = attn.transpose(0, 1)
                output = output + attn
                outputs.append(output)

            outputs = torch.cat(outputs, 0).transpose(0, 1)  # [batch, seq-1, dim_out]
            output_vocab = self.projector(outputs)  # [batch, seq-1, num_vocab]

            return output_vocab
        else:  # 测试
            id_posts = inputs['posts']  # [batch, seq]
            len_posts = inputs['len_posts']  # [batch]
            batch_size = id_posts.size(0)

            embed_posts = self.embedding(id_posts)  # [batch, seq, embed_size]

            # state: [layers, batch, dim]
            encoder_output, state_encoder = self.encoder(embed_posts.transpose(0, 1), len_posts)
            encoder_output = encoder_output.transpose(0, 1)
            attn_masks = (1 - F.one_hot(len_posts, len_posts.max() + 1).cumsum(1))[:, :-1].bool()  # [batch, len]
            if isinstance(state_encoder, tuple):
                context = state_encoder[0]
            context = context[-1, :, :].unsqueeze(0)  # [1, batch, dim]

            done = torch.tensor([0] * batch_size).bool()
            first_input_id = (torch.ones((1, batch_size)) * self.config.start_id).long()
            init_attn = torch.zeros([1, batch_size, self.config.attention_size])
            if gpu:
                done = done.cuda()
                first_input_id = first_input_id.cuda()
                init_attn = init_attn.cuda()

            outputs = []
            for idx in range(max_len):
                if idx == 0:  # 第一个时间步
                    state = state_encoder  # 解码器初始状态
                    decoder_input = torch.cat([self.embedding(first_input_id), init_attn, context], 2)
                else:
                    decoder_input = torch.cat([self.embedding(next_input_id), attn, context], 2)

                # output: [1, batch, dim_out]
                # state: [num_layers, batch, dim_out]
                output, state = self.decoder(decoder_input, state)
                attn, _ = self.attention(output.transpose(0, 1), encoder_output, attn_masks)
                attn = attn.transpose(0, 1)
                output = output + attn
                outputs.append(output)

                vocab_prob = self.projector(output)  # [1, batch, num_vocab]
                next_input_id = torch.argmax(vocab_prob, 2)  # 选择概率最大的词作为下个时间步的输入 [1, batch]
                _done = next_input_id.squeeze(0) == self.config.end_id  # 当前时间步完成解码的 [batch]
                done = done | _done  # 所有完成解码的
                if done.sum() == batch_size:  # 如果全部解码完成则提前停止
                    break

            outputs = torch.cat(outputs, 0).transpose(0, 1)  # [batch, seq, dim_out]
            output_vocab = self.projector(outputs)  # [batch, seq, num_vocab]

            return output_vocab

    # 统计参数
    def print_parameters(self):
        r""" 统计参数 """
        total_num = 0  # 参数总数
        for param in self.parameters():
            num = 1
            if param.requires_grad:
                size = param.size()
                for dim in size:
                    num *= dim
            total_num += num
        print(f"参数总数: {total_num}")

    def save_model(self, epoch, global_step, path):
        r""" 保存模型 """
        torch.save({'embedding': self.embedding.state_dict(),
                    'encoder': self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                    'attention': self.attention.state_dict(),
                    'projector': self.projector.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step}, path)

    def load_model(self, path):
        r""" 载入模型 """
        checkpoint = torch.load(path)
        self.embedding.load_state_dict(checkpoint['embedding'])
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.attention.load_state_dict(checkpoint['attention'])
        self.projector.load_state_dict(checkpoint['projector'])
        epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        return epoch, global_step
