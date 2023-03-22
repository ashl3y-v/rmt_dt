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
    state = model.proc_state(observation)

    terminated = truncated = False
    while not (terminated or truncated):
        action = torch.randn([1, 1, 3], device=device)
        state_pred, reward_pred = critic(state, action)

        action = action.detach().squeeze().cpu().numpy()
        observation, reward, terminated, truncated, info = env.step(action)

        state = model.proc_state(observation).to(device=device).reshape([1, 1, encoding_dim])

        reward = torch.tensor(reward, device=device).reshape([1, 1])
        
        # train
        loss = critic.train_iter(state, state_pred, reward, reward_pred)

        losses = torch.cat([losses, loss.reshape([1])])
        print(e, "loss", loss)

    for g in critic.optim.param_groups:
        g['lr'] = g['lr'] * 0.99

    if e % 10 == 0:
        torch.save(losses, "losses.pt")
        torch.save(critic, save_name + ".pt")

torch.save(critic, save_name + ".pt")
torch.save(losses, "losses.pt")
