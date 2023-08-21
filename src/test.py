import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
import matplotlib.pyplot as plt

from pytorch_optimizer import Ranger21

T.manual_seed(42)

dtype = T.bfloat16
device = "cuda" if T.cuda.is_available() else "cpu"

a = T.randn([10, 10, 10], dtype=dtype, device=device)

a = a.unsqueeze([1])
print(a.shape)
