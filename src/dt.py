import torch as T
import torch.nn as nn
import torch.nn.functional as F

from torchrl import *

class Residual(nn.Module):
    def __init__(self, f: nn.Module, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.f = f

    def forward(self, x: T.Tensor):
        n = self.f(x)
        return F.interpolate(x.unsqueeze(1), n.shape[1:]).squeeze(1) + n


def _unpack_rec(x: list) -> list:
    y = []
    [y.extend(x[i]) for i in range(len(x))]
    return y


def _tokenizer(
    c: list = [3, 6, 12, 24, 48, 96, 96],
    k: list = [8, 8, 8, 8, 8, 3],
    s: list = [2, 2, 2, 2, 2, 1],
    p: list = [3, 3, 3, 3, 3, 0],
    dtype=T.bfloat16,
    device=T.device("cuda")
):
    assert len(c) - 1 == len(k) == len(s) == len(p)
    n = len(k)

    return T.compile(
        nn.Sequential(
            *_unpack_rec(
                [
                    [
                        Residual(nn.Conv2d(c[i], c[i + 1], k[i], s[i], p[i])),
                        nn.Mish(inplace=True),
                    ]
                    for i in range(n)
                ]
            ),
            nn.Flatten(),
        ).to(dtype=dtype, device=device)
    )


class DT(nn.Module):
    def __init__(
        self,
        d_state=96,
        d_act=3,
        d_rew = 1,
        d_mem=147,
        d_segment=8,
        n_layer=16,
        n_head=16,
        min_a=T.tensor([-1, 0, 0]),
        max_a=T.tensor([1, 1, 1]),
        dtype = T.bfloat16,
        device = T.device("cuda")
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.d_state = d_state
        self.d_act = d_act
        self.d_rew = d_rew
        self.d_mem = d_mem
        self.d_segment = d_segment
        self.n_layer = n_layer
        self.n_head = n_head

        self.d_emb = d_state + d_act + d_act ** 2 + d_rew

        self.min_a = min_a.to(dtype=dtype, device=device)
        self.max_a = max_a.to(dtype=dtype, device=device)

        self.tokenizer = _tokenizer(dtype=dtype, device=device)

        self.emb = nn.Embedding(d_segment, d_emb)

        # self.to(dtype=dtype, device=device)

    def forward(
        self,
        s: T.Tensor,
        a: T.Tensor,
        r: T.Tensor,
        artg_hat: T.Tensor,
        timestep: T.Tensor or int,
        mask=None,
    ):
        if isinstance(timestep, int):
            timestep = T.tensor(timestep, dtype=T.long, device=self.device).reshape(
                [1, 1]
            )
        mask = mask or T.ones(
            [s.shape[0], s.shape[1]], dtype=self.dtype, device=self.device
        )
        a = T.cat(
            [
                a,
                T.zeros(
                    a.shape[0],
                    a.shape[1],
                    self.d_act**2,
                    dtype=self.dtype,
                    device=self.device,
                ),
            ],
            dim=-1,
        )
        s_hat, a, artg_hat = self.transformer(
            states=s.detach(),
            actions=a.detach(),
            rewards=r.detach(),
            returns_to_go=artg_hat.detach(),
            timesteps=timestep.detach(),
            attention_mask=mask.detach(),
            return_dict=False,
        )
        s_hat, a, artg_hat = s_hat[:, -1:, :], a[:, -1:, :], artg_hat[:, -1:, :]

        mu, cov = self.split_a(a.to(dtype=T.float32))
        cov = cov @ cov.permute(0, 2, 1) + T.eye(
            cov.shape[-1], dtype=cov.dtype, device=cov.device
        ).expand([cov.shape[0], -1, -1])
        a, prob = self.sample(mu, cov)
        a = self.min_a + a * (self.max_a - self.min_a)

        return s_hat, a, prob, artg_hat

    # def split(self, x):
    #     s_hat, a, padding = T.split(
    #         x,
    #         [self.d_state, self.d_act + self.d_act**2, self.padding],
    #         dim=-1,
    #     )
    #
    #     return s_hat, a

    def split_a(self, a):
        mu = a[:, :, : self.d_act]
        mu = mu.reshape(mu.shape[0], mu.shape[-1])
        cov = a[:, :, self.d_act :].reshape(a.shape[0], self.d_act, self.d_act)

        return mu, cov

    def sample(self, mu, cov):
        dist = self.gaussian(mu, cov)
        a = dist.rsample()
        prob = T.exp(dist.log_prob(a))

        return a, prob

    def gaussian(self, mu, cov):
        return T.distributions.MultivariateNormal(loc=mu, covariance_matrix=cov)

    def save(self):
        T.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(T.load(self.file))


if __name__ == "__main__":
    dtype = T.bfloat16
    device = T.device("cuda" if T.cuda.is_available() else "cpu")

    # tok = _tokenizer()
    dt = DT().to(dtype=dtype, device=device)
    print(dt.max_a.dtype, dt.max_a.device)

    #
    # batches = 2
    # seq_len = 69
    # s = T.randn(batches, seq_len, 768, dtype=dtype, device=device)
    # a = T.randn([batches, seq_len, 3], dtype=dtype, device=device)
    # r = T.randn([batches, seq_len, 1], dtype=dtype, device=device)
    # artg_hat = T.randn([batches, seq_len, 1], dtype=dtype, device=device)
    # s_hat, a, mu, cov, r_hat, artg_hat = dt(s, a, r, artg_hat)
    # print(s_hat.shape, a.shape, mu.shape, cov.shape, r_hat.shape, artg_hat.shape)
