import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import transformers


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim=768, act_dim=3, n_layers=3, n_heads = 4, n_positions=8192, lr=3E-4, min_action=-1, max_action=1, name="decision_transformer", chkpt_dir="tmp/ddpg", dtype=torch.float32, device="cpu"):
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.min_action = min_action
        self.max_action = max_action
        self.mean_action = (max_action + min_action) / 2
        self.scale_action = (max_action - min_action) / 2

        self.name = name
        self.checkpoint_file = os.path.join(chkpt_dir, name+"_ddpg")

        # crashes if length is greater than n_positions
        # config = transformers.BartConfig(d_model=state_dim + act_dim + 1, d_embedding=state_dim + act_dim + 1, n_layer=n_layers, n_head=n_heads)
        # by default its in `block_sparse` mode with num_random_blocks=3, block_size=64
        config = transformers.DecisionTransformerConfig(state_dim=state_dim, act_dim=2 * act_dim, n_positions=n_positions, n_layer=n_layers, n_head=n_heads)
        self.transformer = transformers.DecisionTransformerModel(config)

        self.l2_loss = nn.MSELoss()
        self.optim = optim.RAdam(self.parameters(), lr=lr)

        self.to(dtype=dtype, device=device)

    def forward(self, *args, **kwargs):
        kwargs["actions"] = T.cat([kwargs["actions"], T.zeros(kwargs["actions"].shape, device=self.device)], dim=2)

        state_pred, action_pred, rtg_pred = self.transformer(*args, **kwargs)
        action_pred = self.mean_action + self.scale_action * F.tanh(action_pred)
        return state_pred, action_pred, rtg_pred

    def sample(self, action_pred):
        dist = self.action_dist(action_pred)
        action = dist.rsample().reshape([1, 1, self.act_dim])
        prob = dist.log_prob(action)

        return action, prob

    def gamma_noise(self):
        pass

    # def action_dist(self, action_preds):
    #     mid = int(action_preds.shape[-1] / 2)
    #     return Normal(loc=action_preds[:, :, :mid], scale=torch.maximum(action_preds[:, :, mid:], torch.fill(action_preds[:, :, mid:], 0.01)))

    # def prob(self, dist, actions):
    #     return dist.log_prob(actions)

    def save(self):
        print("Saving critic")
        T.save(self.state_dict(), self.checkpoint_file)

    def load(self):
        print("Loading critic")
        self.load_state_dict(T.load(self.checkpoint_file))
