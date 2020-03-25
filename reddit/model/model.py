import torch
import torch.nn as nn
from Embedding import Embedding
from Encoder import Encoder


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config

        # 定义嵌入层
        self.embedding = Embedding(config.num_vocab,  # 词汇表大小
                                   config.embedding_size,  # 嵌入层维度
                                   config.pad_id,
                                   config.dropout)

        self.affect_embedding = Embedding(config.num_vocab,
                                          config.affect_embedding_size,
                                          config.pad_id,
                                          config.dropout)
        self.affect_embedding.embedding.weight.requires_grad = False

        # 编码器
        self.encoder = Encoder(config.encoder_cell_type,  # rnn类型
                               config.embedding_size+config.affect_embedding_size,  # 输入维度
                               config.encoder_output_size,  # 输出维度
                               config.encoder_num_layers,  # rnn层数
                               config.encoder_bidirectional,  # 是否双向
                               config.dropout)  # dropout概率

        # 输出层
        self.classifier = nn.Sequential(
            nn.Linear(config.encoder_output_size, config.encoder_output_size//2),
            nn.Linear(config.encoder_output_size//2, config.num_classifications),
            nn.Softmax(-1)
        )

    def forward(self, inputs):
        x = inputs['x']  # [batch, len]
        len_x = inputs['len_x']  # [batch]

        embed_x = torch.cat([self.embedding(x), self.affect_embedding(x)], 2)  # [batch, len, embed]
        # state: [layers, batch, dim]
        _, state = self.encoder(embed_x.transpose(0, 1), len_x)
        if isinstance(state, tuple):
            state = state[0]
        context = state[-1, :, :]  # [batch, dim]

        output = self.classifier(context)  # [batch, 7]
        return output

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
                    'affect_embedding': self.affect_embedding.state_dict(),
                    'encoder': self.encoder.state_dict(),
                    'classifier': self.classifier.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step}, path)

    def load_model(self, path):
        r""" 载入模型 """
        checkpoint = torch.load(path)
        self.embedding.load_state_dict(checkpoint['embedding'])
        self.affect_embedding.load_state_dict(checkpoint['affect_embedding'])
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.classifier.load_state_dict(checkpoint['classifier'])
        epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        return epoch, global_step
