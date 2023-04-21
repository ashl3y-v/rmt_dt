import time
import torch as T
from torch import nn
import torch.nn.functional as F
from positional_encodings.torch_encodings import (
    PositionalEncoding1D,
)


class DecisionTransformerModel(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_head: int = 8,
        dim_feedforward: int = 2048,
        output_dim=None,
        n_layers: int = 8,
        dropout: float = 0.5,
        dtype=T.bfloat16,
        device="cuda",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device
        self.dtype = dtype

        self.pos_encoder = PositionalEncoding1D(d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=lambda x: F.mish(x),
            dtype=dtype,
            device=device,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=n_layers, norm=nn.LayerNorm(d_model)
        )
        if output_dim:
            self.decoder = nn.Linear(d_model, output_dim)

        self.to(dtype=dtype, device=device)

    def forward(self, x, mask=None):
        x = x + self.pos_encoder(x)
        x = self.transformer_encoder(x, mask)

        if self.output_dim:
            x = self.decoder(x)

        return x


if __name__ == "__main__":
    model = DecisionTransformerModel(
        d_model=512, output_dim=512, n_head=8, n_layers=8, device="cpu"
    )
    x0 = T.randn([1, 1600, 512], dtype=T.bfloat16, device="cpu")
    r0 = model(x0)

    print(r0.shape)
