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
from utils import init_env, reset_env

torch.autograd.set_detect_anomaly(True)

args = sys.argv

TARGET_RETURN = 3000
EPOCHS = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

losses = torch.tensor([], device=device)
rewards = torch.tensor([], device=device)

env_name = "CarRacing-v2"
encoding_dim = 768
n_positions = 8192

save_name = input("Model save name: ")

env, state_dim, image_dim, act_dim = init_env(env_name)

load_model = bool(int(args[1]))
load_critic = bool(int(args[2]))

if load_model:
    model = torch.load("model.pt").to(dtype=dtype, device=device)
    if load_critic:
        model.critic = torch.load("critic.pt").to(dtype=dtype, device=device)
else:
    model = DecisionTransformer(state_dim=encoding_dim, act_dim=act_dim, n_positions=n_positions, device=device)
    if load_critic:
        model.critic = torch.load("critic.pt").to(dtype=dtype, device=device)

for e in range(EPOCHS):
    hist, attention_mask = reset_env(env, model, act_dim, encoding_dim, TARGET_RETURN, dtype, device)

    terminated = truncated = False
    while not (terminated or truncated):
        state_pred, action_pred, rtg_pred = hist.predict(model, attention_mask)
        state_pred = state_pred.reshape([1, 1, encoding_dim])
        action_pred = action_pred.reshape([1, 1, act_dim])

        action = action_pred.detach().squeeze().cpu().numpy()
        observation, reward, terminated, truncated, info = env.step(action)

        state = model.proc_state(observation).to(device=device).reshape([1, 1, encoding_dim])

        reward = torch.tensor(reward, device=device).reshape([1, 1])
        hist.append(state, state_pred, action_pred, reward, rtg_pred, 1)

        attention_mask = torch.cat([attention_mask, torch.ones([1, 1], device=device)])

        # print("states", hist.states.shape[1], ", ", end="")
        # delete
        # if hist.states.shape[1] == 89:
        #     terminated = True

        if hist.states.shape[1] == 200:
            terminated = True

        # don't delete
        if hist.states.shape[1] == n_positions:
            terminated = True

    # update rtgs
    total_reward = hist.rtg_update()

    # torch.cuda.empty_cache()

    # train (also do it right)
    loss, critic_loss = model.train_iter(hist)

    losses = torch.cat([losses, loss.reshape([1])])
    rewards = torch.cat([rewards, total_reward.reshape([1])])
    print(e, "loss, critic_loss, total_reward", loss, critic_loss, total_reward)

    if e % 10 == 0:
        torch.save(losses, "losses.pt")
        torch.save(rewards, "rewards.pt")
        torch.save(model, save_name + ".pt")

    torch.cuda.empty_cache()

torch.save(model, save_name + ".pt")
torch.save(losses, "losses.pt")
torch.save(rewards, "rewards.pt")
