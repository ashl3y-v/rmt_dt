import torch as T
import torch.nn as nn
import torch.nn.functional as F
import transformers


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        d_state=768,
        d_act=3,
        n_layer=16,
        n_head=16,
        min_a=-1,
        max_a=1,
        max_ep_len=4096,
        n_positions=4096,
        dtype=T.bfloat16,
        device="cuda",
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.d_state = d_state
        self.d_act = d_act
        self.n_layer = n_layer
        self.n_head = n_head
        self.min_a = min_a
        self.max_a = max_a

        config = transformers.DecisionTransformerConfig(
            state_dim=d_state,
            act_dim=d_act + d_act**2,
            n_layer=n_layer,
            n_head=n_head,
            max_ep_len=max_ep_len,
            n_positions=n_positions,
        )
        self.transformer = transformers.DecisionTransformerModel(config).to(
            dtype=dtype, device=device
        )

        self.to(dtype=dtype, device=device)

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
            timestep = T.tensor(timestep, dtype=T.long, device=self.device)
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
            rewards=r,
            returns_to_go=artg_hat.detach(),
            timesteps=timestep,
            attention_mask=mask,
            return_dict=False,
        )
        s_hat, a, artg_hat = s_hat[:, -1:, :], a[:, -1:, :], artg_hat[:, -1:, :]

        mu, cov = self.split_a(a.to(dtype=T.float32))
        cov = cov @ cov.permute(0, 2, 1) + T.eye(
            cov.shape[-1], dtype=cov.dtype, device=cov.device
        ).expand([cov.shape[0], -1, -1])
        a = self.sample(mu, cov)
        a = self.min_a + a * (self.max_a - self.min_a)

        return s_hat, a, artg_hat

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
        print(mu.shape)
        mu = mu.reshape(mu.shape[0], mu.shape[-1])
        cov = a[:, :, self.d_act :].reshape(a.shape[0], self.d_act, self.d_act)

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
    dtype = T.bfloat16
    device = "cuda" if T.cuda.is_available() else "cpu"

    dt = DecisionTransformer(dtype=dtype, device=device)

    batches = 2
    seq_len = 69
    s = T.randn(batches, seq_len, 768, dtype=dtype, device=device)
    a = T.randn([batches, seq_len, 3], dtype=dtype, device=device)
    r = T.randn([batches, seq_len, 1], dtype=dtype, device=device)
    artg_hat = T.randn([batches, seq_len, 1], dtype=dtype, device=device)
    s_hat, a, mu, cov, r_hat, artg_hat = dt(s, a, r, artg_hat)
    print(s_hat.shape, a.shape, mu.shape, cov.shape, r_hat.shape, artg_hat.shape)
