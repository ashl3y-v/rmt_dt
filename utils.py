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
    env,
    vit,
    act_dim,
    state_dim,
    TARGET_RETURN,
    block_size=10,
    max_size=200,
    dtype=T.float32,
    device="cpu",
):
    observation, _ = env.reset()

    state = vit(observation)

    replay_buffer = init_replay_buffer(
        state,
        act_dim=act_dim,
        state_dim=state_dim,
        TARGET_RETURN=TARGET_RETURN,
        block_size=block_size,
        max_size=max_size,
        dtype=dtype,
        device=device,
    )

    return replay_buffer


def mean_range(contents, first, second, dim=0):
    before, inside, after = T.tensor_split(contents, (first, second), dim=dim)

    inside = inside.mean(dim=dim, keepdim=True)

    return T.cat([before, inside, after], dim=dim)
