import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from pytorch_optimizer import Ranger21
from utils import mean_range

T.manual_seed(42)

T.set_autocast_enabled(True)
# T.set_autocast_gpu_dtype(T.float16)
T.set_autocast_cpu_dtype(T.bfloat16)
T.set_autocast_cache_enabled(True)
T.set_autocast_cpu_enabled(True)

a = T.randn([1, 10, 10])
b = T.randn([1, 10, 10])
with T.cpu.amp.autocast():
    c = a @ b

print(c.dtype)
