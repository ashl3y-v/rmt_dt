import sys
import torch
import random
import gymnasium as gym
from torch import nn
from torch.nn import functional as F
from datetime import datetime
from dt import DecisionTransformer
from utils import init_env, reset_env
import numpy as np

args = sys.argv

TARGET_RETURN = 3000
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

env_name = "CarRacing-v2"
encoding_dim = 768
n_positions = 8192

env = gym.make(env_name, render_mode="human")
state_dim = env.observation_space.shape
image_dim = [state_dim[2], state_dim[0], state_dim[1]]
act_dim = env.action_space.shape[0]

load_model = bool(int(args[1]))

if load_model:
    model = torch.load("model.pt").to(dtype=dtype, device=device)
else:
    model = DecisionTransformer(state_dim=encoding_dim, act_dim=act_dim, n_positions=n_positions, device=device)

hist, attention_mask = reset_env(env, model, act_dim, encoding_dim, TARGET_RETURN, dtype, device)

terminated = truncated = False
while not (terminated or truncated):
    state_pred, action_pred, rtg_pred = hist.predict(model, attention_mask)
    state_pred = state_pred.reshape([1, 1, encoding_dim])
    action_pred = action_pred.reshape([1, 1, act_dim])

    action = action_pred.detach().squeeze().cpu().numpy()
    # action = np.array([random.uniform(0.75, 1), random.uniform(-1, 1), random.uniform(0, 0.2)])
    action = torch.distributions.Normal(loc=torch.tensor([1, 1, 0], dtype=torch.float32), scale=1,).rsample().to(dtype=torch.float32, device="cpu")
    observation, reward, terminated, truncated, info = env.step(action)
    print(reward)

    state = model.proc_state(observation).to(device=device).reshape([1, 1, encoding_dim])

    reward = torch.tensor(reward, device=device).reshape([1, 1])
    hist.append(state, state_pred, action_pred, reward, rtg_pred, 1)

    attention_mask = torch.cat([attention_mask, torch.ones([1, 1], device=device)])

    # don't delete
    if hist.states.shape[1] == n_positions:
        terminated = True
