import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from pytorch_optimizer import Ranger21
from utils import mean_range

T.manual_seed(42)
