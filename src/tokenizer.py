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
