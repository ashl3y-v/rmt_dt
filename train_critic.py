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

torch.autograd.set_detect_anomaly(True)

EPOCHS = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

losses = torch.tensor([], device=device)
rewards = torch.tensor([], device=device)

env_name = "CarRacing-v2"
encoding_dim = 768

save_name = "critic"  # input("Model save name: ")

env, state_dim, image_dim, act_dim = init_env(env_name)

model = DecisionTransformer(state_dim=encoding_dim, act_dim=act_dim, dtype=dtype, device=device)
critic = Critic(state_dim=encoding_dim, act_dim=act_dim, dtype=dtype, device=device)

for e in range(EPOCHS):
    observation, _ = env.reset()
    states = torch.tensor([], dtype=dtype, device=device)
    rewards = torch.tensor([], dtype=dtype, device=device)
    states = torch.cat([states, model.proc_state(observation)], dim=2)

    terminated = truncated = False
    while not (terminated or truncated):
        action = np.random.randn(3)

        observation, reward, terminated, truncated, info = env.step(action)

        x = torch.cat([model.proc_state(observation), torch.from_numpy(action).to(device=device).unsqueeze(0).unsqueeze(0)], dim=2)
        states = torch.cat([states, x], dim=2)
        rewards = torch.cat([rewards, torch.tensor(reward, device=device).reshape([1, 1])], dim=1)

    # rtg update
    total_reward = rewards.sum()
    rtgs = torch.zeros([1, states.shape[1]], device=device)
    remaining_reward = total_reward.item()
    for i in range(rtgs.shape[-2]):
        rtgs[:, i, 0] = remaining_reward
        remaining_reward -= rewards[i, 0]

    rtg_preds = critic(states)

    # train
    loss = critic.train_iter(rtgs, rtg_preds)
    losses = torch.cat([losses, loss.reshape([1])])
    print(e, "loss", loss)

    # for g in critic.optim.param_groups:
    #     g['lr'] = g['lr'] * 0.99

    if e % 10 == 0:
        torch.save(losses, "losses.pt")
        torch.save(critic, save_name + ".pt")

torch.save(critic, save_name + ".pt")
torch.save(losses, "losses.pt")
