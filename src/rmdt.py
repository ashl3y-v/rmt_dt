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
        d_s: int = 96,
        d_a: int = 3,
        d_r: int = 1,
        d_padding: int = 0,
        l_obs: int = 16,
        l_mem: int = 8,
        n_layer: int = 4,
        n_head: int = 10,
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
        self.d_padding = d_padding
        self.d_emb = d_s + d_a + d_r + d_padding

        self.l_obs = l_obs
        self.l_mem = l_mem
        self.l_seg = l_mem + l_obs

        self.n_layer = n_layer
        self.n_head = n_head

        self.tokenizer = _tokenizer(device=device, dtype=dtype)

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
        return x[:self.l_mem], x[self.l_mem:, self.d_s:]

    def split_emb(self, x: T.Tensor):
        return T.split(
            x,
            [self.d_s, self.d_a, self.d_r, self.d_padding],
            dim=-1,
        )


if __name__ == "__main__":
    dtype = T.bfloat16
    device = T.device("cuda" if T.cuda.is_available() else "cpu")

    # tok = _tokenizer()
    dt = RMDT(device=device, dtype=dtype)

    x = T.randn([dt.l_seg, dt.d_emb], device=device, dtype=dtype)

    x_h = dt(x)

    #
    # batches = 2
    # seq_len = 69
    # s = T.randn(batches, seq_len, 768, dtype=dtype, device=device)
    # a = T.randn([batches, seq_len, 3], dtype=dtype, device=device)
    # r = T.randn([batches, seq_len, 1], dtype=dtype, device=device)
    # artg_hat = T.randn([batches, seq_len, 1], dtype=dtype, device=device)
    # s_hat, a, mu, cov, r_hat, artg_hat = dt(s, a, r, artg_hat)
    # print(s_hat.shape, a.shape, mu.shape, cov.shape, r_hat.shape, artg_hat.shape)
