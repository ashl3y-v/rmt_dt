import torch as T
import torch.nn as nn
import torch.nn.functional as F


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
    device: T.device = T.device("cuda"),
    dtype: T.dtype = T.bfloat16,
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


class RMDT(nn.Module):
    def __init__(
        self,
        d_s: int = 17,
        d_a: int = 6,
        d_r: int = 1,
        l_mem: int = 8,
        l_obs: int = 12,
        l_olap: int = 4,
        n_layer: int = 4,
        n_head: int = 8,
        dropout: float = 0.1,
        device: T.device = T.device("cuda"),
        dtype: T.dtype = T.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.d_s = d_s
        self.d_a = d_a
        self.d_r = d_r
        self.d_emb = d_s + d_a + d_r

        self.l_mem = l_mem
        self.l_olap = l_olap
        self.l_obs = l_obs
        self.l_seg = l_mem + l_olap + l_obs

        self.n_layer = n_layer
        self.n_head = n_head

        self.embedding = nn.Embedding(
            self.l_seg, self.d_emb, device=device, dtype=dtype
        )

        encoder_layer = nn.TransformerEncoderLayer(
            self.d_emb,
            n_head,
            dropout=dropout,
            batch_first=True,
            device=device,
            dtype=dtype,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, n_layer)

    def forward(self, x: T.Tensor):
        return self.encoder(
            x + self.embedding(T.arange(self.l_seg, device=self.device))
        )

    def extract_mem_a(self, x: T.Tensor):
        return x[: self.l_mem], x[self.l_mem :, self.d_s :]

    def to_x(self, s: T.Tensor, a: T.Tensor, r: T.Tensor):
        assert s.size(-1) == self.d_s
        assert a.size(-1) == self.d_a
        assert r.size(-1) == self.d_r

        return T.cat([s, a, r], dim=-1)

    def to_x_seg(self, s: T.Tensor, a: T.Tensor, r: T.Tensor):
        assert s.size(-1) == self.d_s
        assert a.size(-1) == self.d_a
        assert r.size(-1) == self.d_r

        s = s[:, -self.l_seg :]
        a = a[:, -self.l_seg :]
        r = r[:, -self.l_seg :]

        return T.cat([s, a, r], dim=-1)

    def from_x(self, x_h: T.Tensor):
        mem = x_h[:, : self.l_mem]
        a = x_h[:, -self.l_obs :]

        # l_mem: int = 8,
        # l_obs: int = 12,
        # l_olap: int = 4,
        # return T.split(
        #     x_h,
        #     [self.d_s, self.d_a, self.d_r],
        #     dim=-1,
        # )

        return mem, a


if __name__ == "__main__":
    dtype = T.bfloat16
    device = T.device("cuda" if T.cuda.is_available() else "cpu")

    rmdt = RMDT(device=device, dtype=dtype)

    n = 3
    l = 32

    s = T.zeros([n, l, rmdt.d_s], device=device, dtype=dtype)
    a = T.zeros([n, l, rmdt.d_a], device=device, dtype=dtype)
    r = T.zeros([n, l, rmdt.d_r], device=device, dtype=dtype)

    mem, a = rmdt.from_x(rmdt(rmdt.to_x_seg(s, a, r)))
    print(mem.shape, a.shape)
    # rmdt.from_x(x_h)

    #
    # batches = 2
    # seq_len = 69
    # s = T.randn(batches, seq_len, 768, dtype=dtype, device=device)
    # a = T.randn([batches, seq_len, 3], dtype=dtype, device=device)
    # r = T.randn([batches, seq_len, 1], dtype=dtype, device=device)
    # artg_hat = T.randn([batches, seq_len, 1], dtype=dtype, device=device)
    # s_hat, a, mu, cov, r_hat, artg_hat = dt(s, a, r, artg_hat)
    # print(s_hat.shape, a.shape, mu.shape, cov.shape, r_hat.shape, artg_hat.shape)
