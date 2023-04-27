import torch as T
import torch.nn as nn
import torch.nn.functional as F


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        d_state=768,
        d_act=3,
        d_reward=1,
        n_layer=12,
        padding=0,
        n_head=8,
        n_position=8192,
        dropout=0.1,
        mu_activation=F.tanh,
        cov_activation=F.relu,
        file="actor.pt",
        dtype=T.bfloat16,
        device="cuda",
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.d_state = d_state
        self.d_act = d_act
        self.n_layer = n_layer
        self.padding = padding
        self.n_head = n_head
        self.n_position = n_position
        self.mu_activation = mu_activation
        self.cov_activation = cov_activation

        d_model = d_state + d_act + d_act**2 + d_reward * 2
        self.d_model = d_model

        self.file = file

        # crashes if length is greater than n_positions
        # attention = HyenaOperator()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dropout=dropout,
            activation=F.mish,
            dtype=dtype,
            device=device,
        )
        # layer.self_attn = attention
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layer, norm=nn.LayerNorm(d_model)
        )

        self.to(dtype=dtype, device=device)

    def forward(self, s, a, r, artg_hat, mask=None):
        x = T.cat(
            [
                s,
                a.detach(),
                T.zeros(a.shape[0], a.shape[1], self.d_act**2),
                r,
                artg_hat,
            ]
        )

        s_hat, a, r_hat, artg_hat = self.transformer(x, mask=None)

        mu, cov = self.split(a)
        mu = self.mu_activation(mu)
        cov = self.mu_activation(cov)
        a = self.sample(mu, cov)

        return s_hat, a, mu, cov, r_hat, artg_hat

    def split(self, a):
        mu = a[:, :, : self.d_act]
        mu = mu.reshape(mu.shape[0], mu.shape[-1])
        cov = a[:, :, self.d_act :].reshape(a.shape[0], self.d_act, self.d_act)
        cov = T.abs(cov @ cov.permute(0, 2, 1))

        return mu, cov

    def sample(self, mu, cov):
        dist = self.gaussian(mu, cov)
        a = dist.rsample()

        return a

    def gaussian(self, mu, cov):
        return T.distributions.MultivariateNormal(loc=mu, covariance_matrix=cov)

    def save(self):
        T.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(T.load(self.file))


if __name__ == "__main__":
    dt = DecisionTransformer()
    print(dt.save)
