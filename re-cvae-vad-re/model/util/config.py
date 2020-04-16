
class Config(object):
    r""" 模型参数的类 """
    pad_id = 0
    start_id = 1
    end_id = 2
    unk_id = 3

    num_vocab = 25000
    affect_embedding_size = 3
    embedding_size = 300

    # 编码器参数
    encoder_cell_type = 'GRU'  # in ['GRU', 'LSTM']
    encoder_output_size = 128
    encoder_num_layers = 1
    encoder_bidirectional = True  # 是否是双向rnn

    # 解码器参数
    decoder_cell_type = 'GRU'  # in ['GRU', 'LSTM']
    decoder_output_size = 128
    decoder_num_layers = 1

    # 潜变量参数
    latent_size = 100

    # 先验网络参数
    dims_prior = [110]
    dims_recognize = [110]

    batch_size = 16
    method = 'sgd'  # in ['sgd', 'adam']
    lr = 0.0001  # 初始学习率
    lr_decay = 1.0  # 学习率衰减，每过1个epoch衰减的百分比
    weight_decay = 0  # 权值decay
    max_grad_norm = 5
    kl_step = 10000  # 更新多少次参数之后kl项权值达到1
    dropout = 0.0



