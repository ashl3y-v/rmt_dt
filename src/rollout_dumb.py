import torch as T
import gymnasium as gym
from torch import nn
from torch.nn import functional as F
import numpy as np
import time

device = "cuda" if T.cuda.is_available() else "cpu"
dtype = T.float16

env = gym.make("HalfCheetah-v4", render_mode="human")
d_s = env.observation_space.shape
d_a = env.action_space.shape

env.reset()

terminated = truncated = False

# while not (terminated or truncated):
for i in range(10):
    a = np.random.randn(6)
    o, r, term, trunc, info = env.step(a)
    time.sleep(0.1)
