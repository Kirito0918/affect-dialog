import torch
import torch.nn as nn


class Attention(nn.Module):
    r""" 计算注意力向量 """
    def __init__(self, x_size,
                 y_size,
                 attention_type,
                 attention_size,
                 dropout=0.1):
        super(Attention, self).__init__()
        assert attention_type in ['dot', 'general', 'concat', 'perceptron']  # 点乘、权重、拼接权重、感知机

        self.linear_q = nn.Sequential(nn.Linear(x_size, attention_size), nn.ReLU())
        self.linear_k = nn.Sequential(nn.Linear(y_size, attention_size), nn.ReLU())
        self.linear_v = nn.Sequential(nn.Linear(y_size, attention_size), nn.ReLU())
        self.dropout = nn.Dropout(p=dropout)

        if attention_type == 'general':
            self.general_w = nn.Linear(attention_size, attention_size, bias=False)
        elif attention_type == 'concat':
            self.concat_w = nn.Linear(attention_size*2, 1, bias=False)
        elif attention_type == 'perceptron':
            self.perceptron_w = nn.Linear(attention_size, attention_size, bias=False)
            self.perceptron_u = nn.Linear(attention_size, attention_size, bias=False)
            self.perceptron_v = nn.Linear(attention_size, 1, bias=False)

        self.attention_type = attention_type
        self.attention_size = attention_size

    def forward(self, x,  # [batch, len_x, x_size]
                y,  # [batch, len_y, y_size]
                masks=None):  # [batch, len_y]
        r""" 计算x对y的注意力向量
        masks: 为bool张量，True的位置为需要注意力的位置
        """
        query = self.linear_q(x)  # [batch, len_x, attention_size]
        key = self.linear_k(y)  # [batch, len_y, attention_size]
        value = self.linear_v(y)  # [batch, len_y, attention_size]
        len_x = query.size(1)

        if self.attention_type == 'dot':  # Q^TK
            weight = query.bmm(key.transpose(1, 2))  # [batch, len_x, len_y]
        elif self.attention_type == 'general':  # Q^TWK
            key = self.general_w(key)  # [batch, len_y, attention_size]
            weight = query.bmm(key.transpose(1, 2))
        elif self.attention_type == 'concat':  # W[Q^T;K]
            len_y = key.size(1)
            query = query.unsqueeze(2).repeat(1, 1, len_y, 1)  # [batch, len_x, len_y, attention_size]
            key = key.unsqueeze(1).repeat(1, len_x, 1, 1)  # [batch, len_x, len_y, attention_size]
            weight = self.concat_w(torch.cat([query, key], 3)).squeeze(-1)  # [batch, len_x, len_y]
        else:  # V^Ttanh(WQ+UK)
            len_y = key.size(1)
            query = self.perceptron_w(query)
            key = self.perceptron_u(key)
            query = query.unsqueeze(2).repeat(1, 1, len_y, 1)  # [batch, len_x, len_y, attention_size]
            key = key.unsqueeze(1).repeat(1, len_x, 1, 1)  # [batch, len_x, len_y, attention_size]
            weight = self.perceptron_v(nn.Tanh(query+key)).squeeze(-1)

        weight = weight / self.attention_size ** 0.5  # scale [batch, len_x, len_y]
        if masks is not None:
            masks = masks.unsqueeze(1).repeat(1, len_x, 1)  # [batch, len_x, len_y]
            weight = weight.masked_fill(masks == 0, torch.tensor(float('-inf')))
        weight = weight.softmax(-1)
        weight = self.dropout(weight)
        attention = weight.bmm(value)

        # attention: [batch, len_x, attention_size]
        # weights: [batch, len_x, encoder_len]
        return attention, weight
