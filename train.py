import os
import sys
import argparse
import random
import numpy as np
import torch as T
from torch import nn
from torch.nn import functional as F
import gymnasium as gym
from dt import DecisionTransformer
from vit import ViT
from trainer import Trainer
from matplotlib import pyplot as plt
from replay_buffer import ReplayBuffer
import lightning as L

T.manual_seed(0)

# this probably does nothing
T.backends.cudnn.benchmark = True

T.autograd.set_detect_anomaly(True)

T.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
T.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser(
    prog="Train Decision Transformer", description="Does it", epilog="Made by Ashley :)"
)

parser.add_argument("-t", "--timesteps", default=100)  # option that takes a value
parser.add_argument("-lm", "--load_model", action="store_true")  # on/off flag
parser.add_argument("-sm", "--save_model", action="store_true")  # on/off flag

args = parser.parse_args()

EPOCHS = int(args.timesteps)
save_interval = 50
device = "cuda" if T.cuda.is_available() else "cpu"
dtype = T.bfloat16

# T.set_autocast_gpu_dtype(amp_dtype)
# T.set_autocast_cache_enabled(True)

# losses = torch.tensor([], device=device)
# rewards = torch.tensor([], device=device)

env_name = "BipedalWalker-v3"  # "CarRacing-v2"
d_state = 24  # 768
d_reward = 1

steps_per_action = 3

n_env = 2

env = gym.vector.AsyncVectorEnv(
    [
        lambda: gym.make(env_name),
    ]
    * n_env,
    shared_memory=True,
)
# d_obs = env.observation_space.shape
# d_img = (d_obs[-1], d_obs[1], d_obs[2])
d_act = env.action_space.shape[-1]

model = DecisionTransformer(
    d_state=d_state,
    d_act=d_act,
    dtype=dtype,
    device=device,
)

if args.load_model:
    model.load()

# vit = ViT(d_img=d_img, n_env=n_env, dtype=dtype, device=device)

trainer = Trainer(model.parameters(), epochs=EPOCHS)


for e in range(EPOCHS):
    T.cuda.empty_cache()

    obs, _ = env.reset()
    replay_buffer = ReplayBuffer(
        n_env=n_env,
        d_state=d_state,
        d_act=d_act,
        d_reward=d_reward,
        dtype=dtype,
        device=device,
    )

    terminated = truncated = T.tensor([False] * n_env)
    i = 0
    while not (terminated + truncated).all():
        i += 1

        T.cuda.empty_cache()
        before = T.cuda.memory_reserved()
        s_hat, a, prob, artg_hat = replay_buffer.predict(model)

        a_np = a.detach().cpu().numpy()
        a = a.to(dtype=dtype)

        for _ in range(steps_per_action):
            obs, r, terminated, truncated, info = env.step(a_np)

        terminated, truncated = T.tensor(terminated), T.tensor(truncated)

        # s = vit(obs)
        s = T.from_numpy(obs).to(dtype=dtype, device=device).unsqueeze(1)

        r = T.tensor(r, dtype=dtype, device=device, requires_grad=False).reshape(
            [n_env, d_reward]
        )

        replay_buffer.append(s.detach(), a.detach(), r.detach(), artg_hat.detach(), prob)

        print("delta", (T.cuda.memory_reserved() - before) / replay_buffer.s.shape[1])
        print("mem", T.cuda.memory_reserved() / i**2)

        # print("states", hist.states.shape[1], ", ", end="")
        # errors
        # if replay_buffer.length() == 90:
        #     replay_buffer.detach(0, 90)

        # if replay_buffer.states.shape[1] == 200:
        #     terminated = True

        # don't delete
        # if replay_buffer.length() == n_positions:
        #     terminated = True

    # update Rs
    total_reward, av_r = replay_buffer.artg_update()

    # train (also do it right)
    artg_loss, policy_loss = trainer.learn(replay_buffer)

    print(
        e,
        "artg loss:",
        artg_loss.mean().item(),
        "policy loss:",
        policy_loss.mean().item(),
        "total_reward:",
        total_reward.mean().item(),
        "average return",
        av_r.mean().item(),
    )

    if e % save_interval == 0:
        if args.save_model:
            model.save()
