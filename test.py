import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
import matplotlib.pyplot as plt

from pytorch_optimizer import Ranger21
from utils import mean_range

T.manual_seed(42)

dtype = T.bfloat16
device = "cuda" if T.cuda.is_available() else "cpu"

env = gym.make("CarRacing-v2")

obs, _ = env.reset()

for i in range(20):
    obs, r, terminated, truncated, info = env.step(env.action_space.sample())

obs = T.tensor(obs, dtype=dtype, device=device).permute(2, 0, 1) / 255
pool = nn.MaxPool2d(4)
obs = pool(obs)
print(obs.shape)
plt.matshow(obs.cpu().to(dtype=T.float32).permute(1, 2, 0))

plt.show()
