import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from pytorch_optimizer import Ranger21
from utils import mean_range

T.manual_seed(42)

# EPOCHS = 10000
#
# model = nn.Linear(100, 100)
# optim = T.optim.AdamW(model.parameters(), lr=1E-5)
# lr_scheduler = T.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)
#
# x = T.ones([100])
# y = T.randn([100])
#
# losses = T.tensor([])
#
# for i in range(EPOCHS):
#     y_pred = model(x)
#     loss = ((y - y_pred) ** 2).sum()
#     loss.backward()
#     for _ in range(80):
#         optim.step()
#
#     losses = T.cat([losses, loss])

seq = T.randn([1, 100, 69])
dim = -2
max_length = 50
block_size = 5

n_blocks = (seq.shape[dim] - max_length) // block_size
n_blocks = n_blocks + int(n_blocks / block_size)
blocks = seq[:, : block_size * n_blocks, :]
seq = seq[:, block_size * n_blocks :, :]
blocks = T.stack(T.split(blocks, block_size, dim=dim), dim=dim-1)
compressed = blocks.mean(dim=dim)

res = T.cat([compressed, seq], dim=dim)

print(res.shape)
