import torch as T
import gymnasium as gym
from replay_buffer import ReplayBuffer


def init_env(env_name, n_env=3, **kwargs):
    env = gym.vector.AsyncVectorEnv(
        [
            lambda: gym.make(env_name, **kwargs),
        ]
        * n_env,
        shared_memory=True,
    )
    d_obs = env.observation_space.shape
    d_img = (d_obs[-1], d_obs[1], d_obs[2])
    d_act = env.action_space.shape[-1]

    return env, d_obs, d_img, d_act


def mean_range(contents, first, second, dim=0):
    before, inside, after = T.tensor_split(contents, (first, second), dim=dim)

    inside = inside.mean(dim=dim, keepdim=True)

    return T.cat([before, inside, after], dim=dim)
