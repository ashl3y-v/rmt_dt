import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
import matplotlib.pyplot as plt

from pytorch_optimizer import Ranger21
from performer_pytorch import Performer

T.manual_seed(42)

dtype = T.bfloat16
device = "cuda" if T.cuda.is_available() else "cpu"

model = Performer(dim=512, dim_head=512, depth=1, heads=8, causal=True)

x = T.randn(1, 2048, 512)
r = model(x)

print(r.shape)
