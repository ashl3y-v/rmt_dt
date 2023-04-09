import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import transformers
from dt import DecisionTransformer

a = [nn.Linear(10, 10)]

print(a * 5)
