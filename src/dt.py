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
    device=T.device("cuda"),
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
        d_obs=96,
        d_act=3,
        d_rew=1,
        d_mem=8,
        d_seg=8,
        n_layer=8,
        n_head=16,
        min_a=T.tensor([-1, 0, 0]),
        max_a=T.tensor([1, 1, 1]),
        dropout=0.1,
        dtype=T.bfloat16,
        device=T.device("cuda"),
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device

        d_emb = d_obs + d_act + d_act**2 + 2 * d_rew

        self.d_state = d_obs
        self.d_act = d_act
        self.d_rew = d_rew
        self.d_mem = d_mem
        self.d_seg = d_seg
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_emb = d_emb

        self.min_a = min_a.to(dtype=dtype, device=device)
        self.max_a = max_a.to(dtype=dtype, device=device)

        self.tokenizer = _tokenizer(dtype=dtype, device=device)

        self.embedding = nn.Embedding(d_seg, d_emb)

        encoder_layer = nn.TransformerEncoderLayer(
            d_emb,
            n_head,
            dim_feedforward=512,
            dropout=dropout,
            activation=F.mish,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, n_layer)

        self.to(dtype=dtype, device=device)

    def forward(self, x: T.Tensor):
        # x: [d_seg + d_mem, d_emb]

        print(x.shape)

        pos = self.emb(
            T.arange(self.d_seg + self.d_mem, dtype=self.dtype, device=self.device)
        )

        x = x + pos

        x_h = self.encoder(x)

        # mu, cov = self.split_a(a.to(dtype=T.float32))
        # cov = cov @ cov.permute(0, 2, 1) + T.eye(
        #     cov.shape[-1], dtype=cov.dtype, device=cov.device
        # ).expand([cov.shape[0], -1, -1])
        # a, prob = self.sample(mu, cov)
        # a = self.min_a + a * (self.max_a - self.min_a)
        #
        # return s_hat, a, prob, artg_hat

    def split(self, x: T.Tensor):
        return T.split(
            x,
            [self.d_obs + self.d_act + self.d_act**2 + 2 * self.d_rew],
            dim=-1,
        )

    def split_a(self, a):
        mu, cov = a[..., : self.d_act], a[..., self.d_act :]
        mu = mu.squeeze(1)
        cov = cov.reshape([a.shape[0], self.d_act, self.d_act])

        return mu, cov


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
