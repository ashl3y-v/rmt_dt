import torch as T
import gymnasium as gym
from replay_buffer import ReplayBuffer, init_replay_buffer


def init_env(env_name, **kwargs):
    env = gym.make(env_name, **kwargs)
    obs_dim = env.observation_space.shape
    image_dim = [obs_dim[2], obs_dim[0], obs_dim[1]]
    act_dim = env.action_space.shape[0]

    return env, obs_dim, image_dim, act_dim


def reset_env(
    env, vit, act_dim, state_dim, TARGET_RETURN, dtype=T.float32, device="cpu"
):
    observation, _ = env.reset()

    state = vit(observation)

    replay_buffer = init_replay_buffer(
        state,
        act_dim=act_dim,
        state_dim=state_dim,
        TARGET_RETURN=TARGET_RETURN,
        dtype=dtype,
        device=device,
    )
    attention_mask = T.zeros(1, 1, device=device, dtype=dtype)

    return replay_buffer, attention_mask
