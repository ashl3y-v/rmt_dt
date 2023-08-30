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

parser = argparse.ArgumentParser(
    prog="Train Decision Transformer", description="Does it", epilog="Made by Ashley :)"
)

parser.add_argument("-t", "--timesteps", default=4096)
parser.add_argument("-l", "--load")
parser.add_argument("-s", "--save")
parser.add_argument("-n", "--n_save", default=128)

args = parser.parse_args()

device = T.device("cuda" if T.cuda.is_available() else "cpu")
dtype = T.bfloat16

env_name = "CarRacing-v2"

n_env = 3

env = gym.make_vec("HalfCheetah-v4", num_envs=n_env)

rmdt = RMDT(
    d_s=env.observation_space.shape[-1],
    d_a=env.action_space.shape[-1],
    d_r=1,
    d_padding=0,
    l_obs=10,
    l_overlap=2,
    l_mem=4,
    n_layer=8,
    n_head=4,
    device=device,
    dtype=dtype,
)

print(rmdt.d_s, rmdt.d_a, rmdt.d_r)

if args.load:
    rmdt.load_state_dict(T.load(args.load))


for e in range(args.timesteps):
    T.cuda.empty_cache()

    mem = T.zeros([rmdt.l_mem, rmdt.d_emb], device=device, dtype=dtype)


    obs, info = env.reset()
    o = T.tensor(obs, device=device, dtype=dtype)

    # s0 si
    # a0 ai
    # r0 ri

    buf = T.cat(
        [
            o.unsqueeze(1),
            T.zeros([n_env, 1, rmdt.d_a + rmdt.d_r], device=device, dtype=dtype),
        ],
        dim=-1,
    )

    a, mem = rmdt.extract_mem_a(rmdt(x))

    terminated = truncated = T.tensor([False] * n_env)
    i = 0
    while not (terminated + truncated).all():
        i += 1

        T.cuda.empty_cache()
        before = T.cuda.memory_reserved()

        x = T.cat([mem, replay_buf[-rmdt.l_obs]], dim=0)
        # s_hat, a, prob, artg_hat = replay_buffer.predict(model)

        a, mem = rmdt.extract_mem_a(rmdt(x))

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

        replay_buffer.append(
            s.detach(), a.detach(), r.detach(), artg_hat.detach(), prob
        )

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
