import torch
import gymnasium as gym
from hist import Hist


def init_env(env_name):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape
    image_dim = [state_dim[2], state_dim[0], state_dim[1]]
    act_dim = env.action_space.shape[0]

    return env, state_dim, image_dim, act_dim


def reset_env(env, model, act_dim, encoding_dim, TARGET_RETURN, dtype=torch.float32, device="cpu"):
    observation, _ = env.reset()
    actions = torch.zeros((1, 1, act_dim), device=device, dtype=dtype)
    rewards = torch.zeros(1, 1, device=device, dtype=dtype)

    encoding = model.proc_state(observation)
    states = encoding.reshape(1, 1, encoding_dim).to(device=device, dtype=dtype)
    state_preds = encoding.reshape(1, 1, encoding_dim).to(device=device, dtype=dtype)

    rtg_preds = torch.tensor(TARGET_RETURN, device=device, dtype=dtype).reshape(1, 1, 1)

    timestep = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    attention_mask = torch.zeros(1, 1, device=device, dtype=dtype)

    hist = Hist(states, state_preds, actions, rewards, rtg_preds, timestep, device=device)

    return hist, attention_mask
