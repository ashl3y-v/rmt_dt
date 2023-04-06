import torch
import gymnasium as gym
from replay_buffer import ReplayBuffer, init_replay_buffer


def init_env(env_name, **kwargs):
    env = gym.make(env_name, **kwargs)
    obs_dim = env.observation_space.shape
    image_dim = [state_dim[2], state_dim[0], state_dim[1]]
    act_dim = env.action_space.shape[0]

    return env, obs_dim, image_dim, act_dim


def reset_env(env, model, act_dim, state_dim, TARGET_RETURN, dtype=torch.float32, device="cpu"):
    observation, _ = env.reset()

    state = model.proc_state(observation)

    replay_buffer = init_replay_buffer(state, act_dim=act_dim, state_dim=state_dim, TARGET_RETURN=TARGET_RETURN, dtype=dtype, device=device)
    attention_mask = torch.zeros(1, 1, device=device, dtype=dtype)

    return replay_buffer, attention_mask
