import torch
import torch.nn as nn


class Optim(object):
    r""" 优化器 """
    def __init__(self, method,  # 优化方法
                 lr,  # 学习率
                 lr_decay=1.0,  # 学习率衰减
                 weight_decay=0.0,  # 权值decay
                 eps=1e-8,
                 max_grad_norm=5):  # 梯度裁剪
        assert method in ['SGD', 'Adam', 'Adamax', 'RMSprop']
        self.method = method
        self.lr = lr
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.eps = eps
        self.max_grad_norm = max_grad_norm

    def set_parameters(self, parameters):
        r""" 设置需要优化的参数 """
        self.params = [p for p in parameters if p.requires_grad]  # 所有需要梯度的参数
        if self.method == 'SGD':
            self.optimizer = torch.optim.SGD(self.params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.method == 'Adam':
            self.optimizer = torch.optim.Adam(self.params, lr=self.lr, weight_decay=self.weight_decay, eps=self.eps)
        elif self.method == 'Adamax':
            self.optimizer = torch.optim.Adamax(self.params, lr=self.lr, weight_decay=self.weight_decay, eps=self.eps)
        elif self.method == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.params, lr=self.lr, weight_decay=self.weight_decay, eps=self.eps)


    def update_lr(self, epoch):
        r""" 更新学习率 """
        self.lr = self.lr * self.lr_decay ** epoch  # 计算新的学习率
        for param in self.optimizer.param_groups:
            param['lr'] = self.lr

    def step(self):
        r""" 更新参数 """
        if self.max_grad_norm:
            nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()
