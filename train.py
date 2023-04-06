import os
import sys
import argparse
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
from vit import ViT
from utils import init_env, reset_env

# torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(
                    prog='Train Decision Transformer',
                    description='Does it',
                    epilog='Made by Ashley :)')

parser.add_argument('-t', '--timesteps', default=1000)      # option that takes a value
parser.add_argument('-la', '--load_actor',
                    action='store_true')  # on/off flag
parser.add_argument('-lc', '--load_critic',
                    action='store_true')  # on/off flag
parser.add_argument('-sa', '--save_actor',
                    action='store_true')  # on/off flag
parser.add_argument('-sc', '--save_critic',
                    action='store_true')  # on/off flag

args = parser.parse_args()

TARGET_RETURN = 3000
EPOCHS = 10000
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# losses = torch.tensor([], device=device)
# rewards = torch.tensor([], device=device)

env_name = "CarRacing-v2"
state_dim = 768
n_positions = 8192

env, obs_dim, image_dim, act_dim = init_env(env_name)

if args.load_actor:
    model = torch.load("model.pt").to(dtype=dtype, device=device)
else:
    model = DecisionTransformer(state_dim=state_dim, act_dim=act_dim, n_positions=n_positions, device=device)

if args.load_critic:
    critic = torch.load("critic.pt").to(dtype=dtype, device=device)
else:
    critic = Critic(state_dim=state_dim, act_dim=act_dim, reward_dim=1, fc_size=500, dtype=dtype, device=device)


vit = ViT(image_dim=image_dim, dtype=dtype, device=device)

for e in range(EPOCHS):
    replay_buffer, attention_mask = reset_env(env, model, act_dim, state_dim, TARGET_RETURN, dtype, device)

    terminated = truncated = False
    while not (terminated or truncated):
        state_pred, action_pred, rtg_pred = replay_buffer.predict(model, attention_mask)

        action = action_pred.detach().squeeze().cpu().numpy()
        observation, reward, terminated, truncated, info = env.step(action)

        state = vit(observation).to(device=device).reshape([1, 1, state_dim])

        reward = torch.tensor(reward, device=device, requires_grad=False).reshape([1, 1])
        replay_buffer.append(state, action, reward, rtg_pred)

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

        torch.cuda.empty_cache()

    # update rtgs
    total_reward = hist.rtg_update()

    # torch.cuda.empty_cache()

    # train (also do it right)
    loss = model.train_iter(hist)

    losses = torch.cat([losses, loss.reshape([1])])
    rewards = torch.cat([rewards, total_reward.reshape([1])])
    print(e, "loss, total_reward", loss, total_reward)

    if e % 10 == 0:
        if args.save_actor:
            torch.save(model, "actor.pt")
        if args.save_critic:
            torch.save(model, "critic.pt")

    torch.cuda.empty_cache()
