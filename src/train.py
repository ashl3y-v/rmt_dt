from matplotlib import pyplot as plt
from rmdt import RMDT
from torch import nn
from torch.nn import functional as F
import argparse
import gymnasium as gym
import torch as T

T.manual_seed(0)

T.backends.cudnn.benchmark = True
T.autograd.set_detect_anomaly(True)
T.backends.cuda.matmul.allow_tf32 = True
T.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser()

parser.add_argument("-e", "--epochs", default=4096)
parser.add_argument("-l", "--load")
parser.add_argument("-n", "--n_save", default=128)
parser.add_argument("-s", "--save")

args = parser.parse_args()

device = T.device("cuda" if T.cuda.is_available() else "cpu")
dtype = T.bfloat16

n_env = 3

env = gym.make_vec("HalfCheetah-v4", num_envs=n_env)

rmdt = RMDT(
    d_s=env.observation_space.shape[-1],
    d_a=env.action_space.shape[-1],
    device=device,
    dtype=dtype,
)

if args.load:
    rmdt.load_state_dict(T.load(args.load))


for e in range(args.epochs):
    T.cuda.empty_cache()

    mem = T.zeros([n_env, rmdt.l_m, rmdt.d], device=device, dtype=dtype)

    obs, info = env.reset()
    o = T.tensor(obs, device=device, dtype=dtype).unsqueeze(1)

    buf = T.cat(
        [
            o,
            T.zeros([n_env, 1, rmdt.d_a + rmdt.d_r], device=device, dtype=dtype),
        ],
        dim=-1,
    )

    terminated = truncated = T.tensor([False] * n_env)
    i: int = 0
    while not (terminated + truncated).all():
        i += 1

        T.cuda.empty_cache()

        mem, a = rmdt.from_x(rmdt(T.cat([mem, rmdt.get_o(buf)], dim=-2)))

        a_np = a.detach().to(device="cpu", dtype=T.float32).numpy()

        for a_i in range(a.size(-2)):
            obs, reward, terminated, truncated, info = env.step(a_np[..., a_i, :])
            o = T.tensor(obs, device=device, dtype=dtype).unsqueeze(1)
            r = T.tensor(reward, device=device, dtype=dtype).unsqueeze(1).unsqueeze(2)
            buf = T.cat([buf, T.cat([o, a[..., a_i : a_i + 1, :], r], dim=-1)], dim=-2)

        terminated, truncated = T.tensor(terminated), T.tensor(truncated)

        if i % 100 == 0:
            break

    print(buf.shape)
    assert 1 == 0

    # train

    # print debug info

    if args.save:
        if e % (args.epochs // args.n_save) == 0:
            T.save(rmdt.state_dict(), args.save)
