import os
import sys
import time
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import gymnasium as gym
from datetime import datetime
from dt import DecisionTransformer
from critic import Critic
from utils import init_env, reset_env

EPOCHS = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

rewards = torch.tensor([], device=device)

env_name = "CarRacing-v2"
encoding_dim = 768

save_name = "critic"  # input("Model save name: ")

env, state_dim, image_dim, act_dim = init_env(env_name)

model = DecisionTransformer(state_dim=encoding_dim, act_dim=act_dim, dtype=dtype, device=device)

load_critic = True
if load_critic:
    critic = torch.load("critic.pt").to(dtype=dtype, device=device)
else:
    critic = Critic(state_dim=encoding_dim, act_dim=act_dim, dtype=dtype, device=device)

for e in range(EPOCHS):
    random.seed(e)
    observation, _ = env.reset()
    states = torch.tensor([], dtype=dtype, device=device)
    rewards = torch.tensor([], dtype=dtype, device=device)
        action = np.array([random.uniform(-1, 1), random.uniform(0.5, 1), random.uniform(0, 0.33)])
    x = torch.cat([model.proc_state(observation), torch.from_numpy(action).to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)], dim=2)
    states = torch.cat([states, x], dim=1)

    terminated = truncated = False
    while not (terminated or truncated):
        action = np.array([random.uniform(-1, 1), random.uniform(0.5, 1), random.uniform(0, 0.33)])

        observation, reward, terminated, truncated, info = env.step(action)

        x = torch.cat([model.proc_state(observation), torch.from_numpy(action).to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)], dim=2)
        # print("x", x.shape)
        states = torch.cat([states, x], dim=1)
        rewards = torch.cat([rewards, torch.tensor(reward, device=device).reshape([1, 1])], dim=1)

    # rtg update
    total_reward = rewards.sum()
    rtgs = torch.zeros([1, states.shape[1]], device=device)
    remaining_reward = total_reward.item()
    for i in range(rtgs.shape[-2]):
        rtgs[i, 0] = remaining_reward
        remaining_reward -= rewards[i, 0]

    print(states.shape)
    rtg_preds = critic(states)

    # train
    loss = critic.train_iter(rtgs, rtg_preds)
    print(e, "loss", loss)

    # for g in critic.optim.param_groups:
    #     g['lr'] = g['lr'] * 0.99

    if e % 10 == 0:
        torch.save(critic, save_name + ".pt")

torch.save(critic, save_name + ".pt")
