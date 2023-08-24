import argparse
import torch as T
from torch import nn
from torch.nn import functional as F
import gymnasium as gym
from matplotlib import pyplot as plt
from replay_buffer import ReplayBuffer
from dt import _tokenizer, RMDT

T.manual_seed(0)

T.backends.cudnn.benchmark = True
T.autograd.set_detect_anomaly(True)
T.backends.cuda.matmul.allow_tf32 = True
T.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser(
    prog="Train Decision Transformer", description="Does it", epilog="Made by Ashley :)"
)

parser.add_argument("-t", "--timesteps", default=4096)
parser.add_argument("-l", "--load", default="dt.pt")
parser.add_argument("-s", "--save", default="dt.pt")
parser.add_argument("-n", "--n_save", default=128)

args = parser.parse_args()

EPOCHS = int(args.timesteps)
n_save = args.n_save
device = T.device("cuda" if T.cuda.is_available() else "cpu")
dtype = T.bfloat16

env_name = "CarRacing-v2"

n_env = 2

env = gym.vector.AsyncVectorEnv(
    [
        lambda: gym.make(env_name),
    ]
    * n_env,
    shared_memory=True,
)

d_s = 96
d_a = 3
d_r = 1

tokenizer = _tokenizer()

rmdt = RMDT(
    d_s=d_s,
    d_a=d_a,
    d_r=d_r,
    d_padding=0,
    l_obs=16,
    l_mem=8,
    n_layer=4,
    n_head=10,
    device=device,
    dtype=dtype,
)

if args.load_model:
    rmdt.load_state_dict(T.load(args.load))


for e in range(EPOCHS):
    T.cuda.empty_cache()

    obs, _ = env.reset()
    # fix
    # replay_buffer = ReplayBuffer(
    #     n_env=n_env,
    #     d_state=d_state,
    #     d_act=d_act,
    #     d_reward=d_reward,
    #     dtype=dtype,
    #     device=device,
    # )

    mem = T.zeros([rmdt.l_mem, rmdt.d_emb], device=device, dtype=dtype)

    terminated = truncated = T.tensor([False] * n_env)
    i = 0
    while not (terminated + truncated).all():
        i += 1

        T.cuda.empty_cache()
        before = T.cuda.memory_reserved()

        x_h = rmdt()
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
