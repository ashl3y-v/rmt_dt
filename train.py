import os
import sys
import argparse
import time
import random
import numpy as np
import torch as T
from torch import nn
from torch.nn import functional as F
import gymnasium as gym
from datetime import datetime
from dt import DecisionTransformer
from critic import Critic
from vit import ViT
from trainer import Trainer
from utils import init_env, reset_env

# torch.autograd.set_detect_anomaly(True)

# this probably does nothing
T.backends.cudnn.benchmark = True

T.autograd.set_detect_anomaly(True)

# TODO:
# use batch normalization maybe
# use automatic mixed precision (however that works)

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

TARGET_RETURN = 10000
EPOCHS = int(args.timesteps)
device = "cuda" if T.cuda.is_available() else "cpu"
dtype = T.float32

steps_per_action = 3

# losses = torch.tensor([], device=device)
# rewards = torch.tensor([], device=device)

env_name = "CarRacing-v2"
state_dim = 768
n_positions = 8192

env, obs_dim, image_dim, act_dim = init_env(env_name)

actor = DecisionTransformer(state_dim=state_dim, act_dim=act_dim, n_positions=n_positions, stdev=0.03, dtype=dtype, device=device)
if args.load_actor:
    actor.load()

critic = Critic(state_dim=state_dim, act_dim=act_dim, fc_size=500, dtype=dtype, device=device)
if args.load_critic:
    critic.load()

vit = ViT(image_dim=image_dim, dtype=dtype, device=device)

trainer = Trainer(actor.parameters(), critic.parameters(), lr_actor=3E-4, lr_critic=1E-3)

for e in range(EPOCHS):
    T.cuda.empty_cache()

    replay_buffer, attention_mask = reset_env(env, vit, act_dim, state_dim, TARGET_RETURN, dtype, device)

    terminated = truncated = False
    while not (terminated or truncated):
        state_pred, action_pred, rtg_pred = replay_buffer.predict(actor, attention_mask)

        # think about n frame-length actions
        action = actor.sample(action_pred)
        action_np = action.detach().squeeze().cpu().numpy()

        for _ in range(steps_per_action):
            observation, reward, terminated, truncated, info = env.step(action_np)

        state = vit(observation).to(device=device).reshape([1, 1, state_dim])

        reward = T.tensor(reward, device=device, requires_grad=False).reshape([1, 1])

        replay_buffer.append(state, action, reward, rtg_pred)

        attention_mask = T.cat([attention_mask, T.ones([1, 1], device=device)])

        # print("states", hist.states.shape[1], ", ", end="")
        # delete
        # if hist.states.shape[1] == 89:
        #     terminated = True

        if replay_buffer.states.shape[1] == 200:
            terminated = True

        # don't delete
        if replay_buffer.states.shape[1] == n_positions:
            terminated = True

    # update rtgs
    total_reward = replay_buffer.rtg_update()

    # train (also do it right)
    critic_loss, actor_loss = trainer.learn(critic, replay_buffer)

    print(e, "critic_loss:", critic_loss.item(), "actor_loss:", actor_loss.item(), "total_reward:", total_reward.item())

    if e % 50 == 0:
        if args.save_actor:
            actor.save()
        if args.save_critic:
            critic.save()
