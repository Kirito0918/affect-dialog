import torch
import torch.nn as nn
from Embedding import Embedding
from Encoder import Encoder
from PriorNet import PriorNet
from RecognizeNet import RecognizeNet
from Decoder import Decoder
from PrepareState import PrepareState


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config

        self.embedding = Embedding(config.num_vocab,
                                   config.embedding_size,
                                   config.pad_id,
                                   config.dropout)

        self.affect_embedding = Embedding(config.num_vocab,
                                          config.affect_embedding_size,
                                          config.pad_id,
                                          config.dropout)
        self.affect_embedding.embedding.weight.requires_grad = False

        self.post_encoder = Encoder(config.encoder_cell_type,
                                    config.embedding_size + config.affect_embedding_size,
                                    config.encoder_output_size,
                                    config.encoder_num_layers,
                                    config.encoder_bidirectional,
                                    config.dropout)

        self.response_encoder = Encoder(config.encoder_cell_type,
                                        config.embedding_size + config.affect_embedding_size,
                                        config.encoder_output_size,
                                        config.decoder_num_layers,
                                        config.encoder_bidirectional,
                                        config.dropout)

        self.prior_net = PriorNet(config.encoder_output_size,
                                  config.latent_size,
                                  config.dims_prior)

        self.recognize_net = RecognizeNet(config.encoder_output_size,
                                          config.encoder_output_size,
                                          config.latent_size,
                                          config.dims_recognize)

        self.prepare_state = PrepareState(config.encoder_output_size + config.latent_size,
                                          config.decoder_cell_type,
                                          config.decoder_output_size,
                                          config.decoder_num_layers)

        self.decoder = Decoder(config.decoder_cell_type,
                               config.embedding_size + config.affect_embedding_size + config.encoder_output_size,
                               config.decoder_output_size,
                               config.decoder_num_layers,
                               config.dropout)

        self.projector = nn.Sequential(nn.Linear(config.decoder_output_size, config.num_vocab), nn.Softmax(-1))

    def forward(self, inputs, inference=False, max_len=60, gpu=True):
        if not inference:  # 训练
            id_posts = inputs['posts']  # [batch, seq]
            len_posts = inputs['len_posts']  # [batch]
            id_responses = inputs['responses']  # [batch, seq]
            len_responses = inputs['len_responses']  # [batch, seq]
            sampled_affect_latents = inputs['sampled_latents']  # [batch, latent_size]
            len_decoder = id_responses.size(1) - 1

            embed_posts = torch.cat([self.embedding(id_posts), self.affect_embedding(id_posts)], 2)
            embed_responses = torch.cat([self.embedding(id_responses), self.affect_embedding(id_responses)], 2)
            _, state_posts = self.post_encoder(embed_posts.transpose(0, 1), len_posts)
            _, state_responses = self.response_encoder(embed_responses.transpose(0, 1), len_responses)
            if isinstance(state_posts, tuple):
                state_posts = state_posts[0]
            if isinstance(state_responses, tuple):
                state_responses = state_responses[0]
            x = state_posts[-1, :, :]  # [batch, dim]
            y = state_responses[-1, :, :]  # [batch, dim]

            # 采样并重参数化
            _mu, _logvar = self.prior_net(x)  # [batch, latent]
            mu, logvar = self.recognize_net(x, y)  # [batch, latent]
            z = mu + (0.5 * logvar).exp() * sampled_affect_latents

            first_state = self.prepare_state(torch.cat([z, x], 1))  # [num_layer, batch, dim_out]
            decoder_inputs = embed_responses[:, :-1, :].transpose(0, 1)  # [len_decoder, batch, embed_size]
            decoder_inputs = decoder_inputs.split([1] * len_decoder, 0)

            outputs = []
            for idx in range(len_decoder):
                if idx == 0:
                    state = first_state  # 解码器初始状态
                decoder_input = torch.cat([decoder_inputs[idx], x.unsqueeze(0)], 2)  # 当前时间步输入 [1, batch, embed_size+context_size]
                # output: [1, batch, dim_out]
                # state: [num_layer, batch, dim_out]
                output, state = self.decoder(decoder_input, state)
                outputs.append(output)  # [len_decoder*[1, batch, dim_out]]

            outputs = torch.cat(outputs, 0).transpose(0, 1)  # [batch, len_decoder, dim_out]
            output_vocab = self.projector(outputs)
            return output_vocab, _mu, _logvar, mu, logvar
        else:  # 测试
            id_posts = inputs['posts']  # [batch, seq]
            len_posts = inputs['len_posts']  # [batch]
            sampled_latents = inputs['sampled_latents']  # [batch, latent_size]
            batch_size = id_posts.size(0)

            embed_posts = torch.cat([self.embedding(id_posts), self.affect_embedding(id_posts)], 2)
            _, state_posts = self.post_encoder(embed_posts.transpose(0, 1), len_posts)
            if isinstance(state_posts, tuple):
                state_posts = state_posts[0]  # [layers, batch, dim]
            x = state_posts[-1, :, :]  # [batch, dim]

            _mu, _logvar = self.prior_net(x)  # [batch, latent]
            z = _mu + (0.5 * _logvar).exp() * sampled_latents

            first_state = self.prepare_state(torch.cat([z, x], 1))  # [num_layer, batch, dim_out]
            first_input_id = (torch.ones((1, batch_size)) * self.config.start_id).long()  # [1, batch_size]
            done = torch.tensor([0] * batch_size).bool()
            if gpu:
                first_input_id = first_input_id.cuda()
                done = done.cuda()

            outputs = []
            for idx in range(max_len):
                if idx == 0:
                    state = first_state  # 解码器初始状态
                    decoder_input = torch.cat([self.embedding(first_input_id),
                                               self.affect_embedding(first_input_id),
                                               x.unsqueeze(0)], 2)
                else:
                    decoder_input = torch.cat([self.embedding(next_input_id),
                                               self.affect_embedding(next_input_id),
                                               x.unsqueeze(0)], 2)
                # output: [1, batch, dim_out]
                # state: [num_layer, batch, dim_out]
                output, state = self.decoder(decoder_input, state)
                outputs.append(output)  # [len_decoder*[1, batch, 3]]
                vocab_prob = self.projector(output)  # [1, batch, num_vocab]
                next_input_id = torch.argmax(vocab_prob, 2)  # 选择概率最大的词作为下个时间步的输入 [1, batch]

                _done = next_input_id.squeeze(0) == self.config.end_id  # 当前时间步完成解码的 [batch]
                done = done | _done  # 所有完成解码的
                if done.sum() == batch_size:  # 如果全部解码完成则提前停止
                    break

            outputs = torch.cat(outputs, 0).transpose(0, 1)  # [batch, max_len, 3]
            output_vocab = self.projector(outputs)
            return output_vocab, _mu, _logvar, None, None

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
                    'affect_embedding': self.affect_embedding.state_dict(),
                    'post_encoder': self.post_encoder.state_dict(),
                    'response_encoder': self.response_encoder.state_dict(),
                    'prior_net': self.prior_net.state_dict(),
                    'recognize_net': self.recognize_net.state_dict(),
                    'prepare_state': self.prepare_state.state_dict(),
                    'decoder': self.decoder.state_dict(),
                    'projector': self.projector.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step}, path)

    def load_model(self, path):
        r""" 载入模型 """
        checkpoint = torch.load(path)
        self.embedding.load_state_dict(checkpoint['embedding'])
        self.affect_embedding.load_state_dict(checkpoint['affect_embedding'])
        self.post_encoder.load_state_dict(checkpoint['post_encoder'])
        self.response_encoder.load_state_dict(checkpoint['response_encoder'])
        self.prior_net.load_state_dict(checkpoint['prior_net'])
        self.recognize_net.load_state_dict(checkpoint['recognize_net'])
        self.prepare_state.load_state_dict(checkpoint['prepare_state'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.projector.load_state_dict(checkpoint['projector'])
        epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']

        return epoch, global_step
