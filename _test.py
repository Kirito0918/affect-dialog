import torch

a = [[1, 3, 5], [0, 4]]
max_a = max([max(_) for _ in a])
print(max_a)

b = [[33]]
print(a+b)

print(torch.arange(0, 1000).long())

c = [[[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]],  # 3, 2, 3
     [[0.3, 0.3, 0.4], [0.1, 0.1, 0.8]],
     [[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]]]
c = torch.tensor(c).float()

d = [[0.7, 0.7, 0.3, 0.3], [0.5, 0.3, 0.3, 0.5], [0.2, 0.2, 0.7, 0.4]]  # 3, 4
d = torch.tensor(d).float()
print(c.matmul(d))
d = d.unsqueeze(0).repeat(3, 1, 1)
print(c.bmm(d).size())

e = torch.tensor([[2, 2, 1], [2, 3, 5]]).float()  # 2, 3
f = torch.tensor([[2, 4, 5], [2, 3, 5]]).float()  # 2, 3
print(torch.nn.functional.mse_loss(e, f, reduction='none'))
