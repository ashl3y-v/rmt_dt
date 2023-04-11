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

T.manual_seed(42)

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

parser.add_argument('-t', '--timesteps', default=100)      # option that takes a value
parser.add_argument('-lm', '--load_model',
                    action='store_true')  # on/off flag
parser.add_argument('-sm', '--save_model',
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

model = DecisionTransformer(state_dim=state_dim, act_dim=act_dim, n_positions=n_positions, stdev=0.03, dtype=dtype, device=device)

if args.load_model:
    model.load()

vit = ViT(image_dim=image_dim, dtype=dtype, device=device)

trainer = Trainer(model.parameters(), epochs=EPOCHS)

for e in range(EPOCHS):
    T.cuda.empty_cache()

    replay_buffer, attention_mask = reset_env(env, vit, act_dim, state_dim, TARGET_RETURN, dtype, device)

    terminated = truncated = False
    while not (terminated or truncated):
        state_pred, action_pred, R_pred = replay_buffer.predict(model, attention_mask)

        # think about n frame-length actions
        action = model.sample(action_pred)
        action_np = action.detach().squeeze().cpu().numpy()

        for _ in range(steps_per_action):
            observation, reward, terminated, truncated, info = env.step(action_np)

        state = vit(observation).to(device=device).reshape([1, 1, state_dim])

        reward = T.tensor(reward, device=device, requires_grad=False).reshape([1, 1])

        replay_buffer.append(state, action, reward, R_pred)

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

    # update Rs
    total_reward, av_r = replay_buffer.R_update()

    # train (also do it right)
    P_loss, R_loss = trainer.learn(replay_buffer)

    print(e, "P_loss:", P_loss.item(), "R_loss:", R_loss.item(), "total_reward:", total_reward.item(), "average return", av_r.item())

    if e % 50 == 0:
        if args.save_model:
            model.save()
