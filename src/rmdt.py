import torch as T
import torch.nn as nn
import torch.nn.functional as F


class RMDT(nn.Module):
    def __init__(
        self,
        d_s: int = 17,
        d_a: int = 6,
        d_r: int = 1,
        l_m: int = 6,
        l_o: int = 10,
        l_a: int = 8,
        n_layer: int = 8,
        n_head: int = 8,
        dropout: float = 0.1,
        device: T.device = T.device("cuda"),
        dtype: T.dtype = T.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        assert l_a <= l_o

        self.d_s = d_s
        self.d_a = d_a
        self.d_r = d_r
        self.d = d_s + d_a + d_r

        self.l_m = l_m
        self.l_o = l_o
        self.l_a = l_a
        self.l = l_m + l_o

        self.n_layer = n_layer
        self.n_head = n_head

        self.emb = nn.Embedding(self.l, self.d, device=device, dtype=dtype)

        encoder_layer = nn.TransformerEncoderLayer(
            self.d,
            n_head,
            dropout=dropout,
            batch_first=True,
            device=device,
            dtype=dtype,
        )

        self.enc = nn.TransformerEncoder(encoder_layer, n_layer)

    def forward(self, x: T.Tensor):
        return self.enc(x + self.emb(T.arange(x.size(-2), device=self.device)))

    def get_o(self, r: T.Tensor):
        l = min(self.l_o, r.size(-2))

        return r[..., -l:, :]

    def to_x(self, s: T.Tensor, a: T.Tensor, r: T.Tensor):
        return T.cat([s, a, r], dim=-1)

    def join(self, s: T.Tensor, a: T.Tensor, r: T.Tensor):
        return T.cat([s, a, r], dim=-1)

    def from_x(self, x: T.Tensor):
        if x.size(-2) == self.l:
            return x[..., : self.l_m, :], x[..., -self.l_a :, self.d_s : -self.d_r]
        else:
            return x[..., : self.l_m, :], x[..., self.l_m :, self.d_s : -self.d_r]

    def split(self, x: T.Tensor):
        return x.split([self.d_s, self.d_a, self.d_r], dim=-1)


if __name__ == "__main__":
    dtype = T.bfloat16
    device = T.device("cuda" if T.cuda.is_available() else "cpu")

    rmdt = RMDT(device=device, dtype=dtype)

    n = 3
    l = 32

    s = T.zeros([n, l, rmdt.d_s], device=device, dtype=dtype)
    a = T.zeros([n, l, rmdt.d_a], device=device, dtype=dtype)
    r = T.zeros([n, l, rmdt.d_r], device=device, dtype=dtype)

    x = rmdt.to_x_seg(s, a, r)
    mem, a = rmdt.from_x(rmdt(x))
    print(x.shape, mem.shape, a.shape)
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
