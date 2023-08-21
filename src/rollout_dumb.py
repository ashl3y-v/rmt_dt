import torch as T
import gymnasium as gym
from torch import nn
from torch.nn import functional as F
import numpy as np

device = "cuda" if T.cuda.is_available() else "cpu"
dtype = T.float16

env_name = "CarRacing-v2"

env = gym.make(env_name, render_mode="human")
state_dim = env.observation_space.shape
act_dim = env.action_space.shape[0]

env.reset()

i = 0
terminated = truncated = False

go = np.array([0, 1, 0])
no = np.array([0, 0, 1])
action = go
while not (terminated or truncated):
    if i % 5 == 0:
        if (action == go).all():
            action = no
        elif (action == no).all():
            action = go
    observation, reward, terminated, truncated, info = env.step(action)
    i += 1
