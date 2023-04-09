import torch as T
import torch.nn as nn
from torch.distributions import Normal
import transformers


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim=768, act_dim=3, n_layers=3, n_heads=4, stdev=0.01, n_positions=8192, file="actor.pt", dtype=T.float32, device="cpu"):
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.file = file

        self.state_dim = state_dim
        self.act_dim = act_dim

        self.stdev = stdev

        # crashes if length is greater than n_positions
        config = transformers.DecisionTransformerConfig(state_dim=state_dim, act_dim=act_dim, n_positions=n_positions, n_layer=n_layers, n_head=n_heads)
        self.transformer = transformers.DecisionTransformerModel(config)

        self.to(dtype=dtype, device=device)

    def forward(self, *args, **kwargs):
        # outdated, padding act from x to 2x
        # kwargs["actions"] = T.cat([kwargs["actions"], T.zeros(kwargs["actions"].shape, device=self.device)], dim=2)

        state_pred, action_pred, rtg_pred = self.transformer(*args, **kwargs)
        return state_pred, action_pred, rtg_pred

    def sample(self, action_pred):
        dist = self.gaussian(action_pred)
        action = dist.rsample().reshape([1, 1, self.act_dim])

        return action

    def gaussian(self, action_pred):
        return Normal(loc=action_pred, scale=T.fill(action_pred, self.stdev))

    def save(self):
        print("Saving critic")
        T.save(self.state_dict(), self.file)

    def load(self):
        print("Loading critic")
        self.load_state_dict(T.load(self.file))
