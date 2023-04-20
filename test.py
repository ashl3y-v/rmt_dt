import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from pytorch_optimizer import Ranger21
from utils import mean_range

T.manual_seed(42)

a = T.fill(T.zeros([10]), T.nan)
b = T.ones([10])
a = T.cat([a, b])

print(a.isnan().any())  # ,  # isnan().any())
