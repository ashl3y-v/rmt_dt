import torch as T
import torch.nn as nn
import transformers


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim=768,
        act_dim=8,
        n_layers=3,
        n_heads=8,
        n_positions=8192,
        min_action=None,
        max_action=None,
        file="actor.pt",
        dtype=T.float32,
        device="cuda",
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.file = file

        self.state_dim = state_dim
        self.act_dim = act_dim

        self.min_action = min_action
        self.max_action = max_action

        # crashes if length is greater than n_positions
        config = transformers.DecisionTransformerConfig(
            state_dim=state_dim,
            act_dim=act_dim + act_dim**2,
            n_positions=n_positions,
            n_layer=n_layers,
            n_head=n_heads,
            action_tanh=True,
            activation_function="gelu_new",
        )
        self.transformer = transformers.DecisionTransformerModel(config)

        self.to(dtype=dtype, device=device)

    def forward(self, *args, **kwargs):
        cov_pad = T.zeros(
            [
                kwargs["actions"].shape[-3],
                kwargs["actions"].shape[-2],
                self.act_dim**2,
            ],
            device=self.device,
        )
        kwargs["actions"] = T.cat([kwargs["actions"], cov_pad], dim=-1)

        with T.cuda.amp.autocast():
            state_preds, action_preds, reward_preds = self.transformer(*args, **kwargs)

        return (
            state_preds.to(dtype=self.dtype),
            action_preds.to(dtype=self.dtype),
            reward_preds.to(dtype=self.dtype),
        )

    def split(self, actions):
        mu = actions[:, :, : self.act_dim]
        if self.min_action and self.max_action:
            mu = self.min_action + mu * (self.max_action - self.min_action)
        mu = mu.reshape(mu.shape[0], mu.shape[-1])
        cov = actions[:, :, self.act_dim :].reshape(
            actions.shape[0], self.act_dim, self.act_dim
        )
        cov = T.abs(cov @ cov.permute(0, 2, 1))

        return mu, cov

    def sample(self, actions):
        mu, cov = self.split(actions)
        dist = self.gaussian(mu, cov)
        action = dist.rsample()

        return action

    def gaussian(self, mu, cov):
        return T.distributions.MultivariateNormal(loc=mu, covariance_matrix=cov)

    def save(self):
        T.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(T.load(self.file))


if __name__ == "__main__":
    dt = DecisionTransformer()
    print(dt.save)
