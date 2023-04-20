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

T.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
T.backends.cudnn.allow_tf32 = True

# TODO:
# use batch normalization maybe

parser = argparse.ArgumentParser(
    prog="Train Decision Transformer", description="Does it", epilog="Made by Ashley :)"
)

parser.add_argument("-t", "--timesteps", default=100)  # option that takes a value
parser.add_argument("-lm", "--load_model", action="store_true")  # on/off flag
parser.add_argument("-sm", "--save_model", action="store_true")  # on/off flag

args = parser.parse_args()

TARGET_RETURN = 10000
EPOCHS = int(args.timesteps)
device = "cuda" if T.cuda.is_available() else "cpu"
dtype = T.float32
amp_dtype = T.bfloat16
scaler = T.cuda.amp.grad_scaler.GradScaler(enabled=True)

T.set_autocast_gpu_dtype(amp_dtype)
T.set_autocast_cache_enabled(True)

# losses = torch.tensor([], device=device)
# rewards = torch.tensor([], device=device)

env_name = "CarRacing-v2"
state_dim = 768
n_positions = 8192

steps_per_action = 5

num_envs = 1

env, obs_dim, image_dim, act_dim = init_env(env_name, num_envs=num_envs)

model = DecisionTransformer(
    state_dim=state_dim,
    act_dim=act_dim,
    n_positions=n_positions,
    dtype=dtype,
    device=device,
)

if args.load_model:
    model.load()

vit = ViT(image_dim=image_dim, num_envs=num_envs, dtype=dtype, device=device)

trainer = Trainer(
    model.parameters(), epochs=EPOCHS, scaler=scaler, use_lr_schedule=False
)

for e in range(EPOCHS):
    T.cuda.empty_cache()

    replay_buffer = reset_env(
        env,
        vit,
        act_dim,
        state_dim,
        TARGET_RETURN,
        num_envs=num_envs,
        max_size=150,
        dtype=dtype,
        device=device,
    )
    attention_mask = T.ones(
        [num_envs, replay_buffer.length()], dtype=dtype, device=device
    )

    terminated = truncated = T.tensor([False] * num_envs)
    while not (terminated + truncated).all():
        state_pred, action_pred, R_pred = replay_buffer.predict(model, attention_mask)

        action = model.sample(action_pred)
        action_np = action.detach().cpu().numpy()

        for _ in range(steps_per_action):
            observation, reward, terminated, truncated, info = env.step(action_np)

        terminated, truncated = T.tensor(terminated), T.tensor(truncated)

        state = vit(observation).reshape([num_envs, 1, state_dim])

        reward = T.tensor(reward, device=device, requires_grad=False).reshape(
            [num_envs, 1]
        )

        replay_buffer.append(state, action, reward, R_pred, compress=True)

        attention_mask = T.ones(
            [num_envs, replay_buffer.length()], dtype=dtype, device=device
        )

        # print("states", hist.states.shape[1], ", ", end="")
        # delete
        # if hist.states.shape[1] == 89:
        #     terminated = True

        # if replay_buffer.states.shape[1] == 200:
        #     terminated = True

        # don't delete
        if replay_buffer.states.shape[1] == n_positions:
            terminated = True

    # update Rs
    total_reward, av_r = replay_buffer.R_update()

    # train (also do it right)
    P_loss, R_loss = trainer.learn(replay_buffer)

    print(
        e,
        "P_loss:",
        P_loss.item(),
        "R_loss:",
        R_loss.item(),
        "total_reward:",
        total_reward.item(),
        "average return",
        av_r.item(),
    )

    if e % 50 == 0:
        if args.save_model:
            model.save()
