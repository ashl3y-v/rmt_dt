import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import transformers


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim=768, act_dim=3, n_layers=12, n_heads = 25, n_positions=8192, image_dim=[3, 96, 96], dtype=torch.float32, device="cpu"):
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.state_dim = state_dim
        self.act_dim = act_dim

        # crashes if length is greater than n_positions
        config = transformers.TransfoXLConfig(d_model=state_dim + act_dim + 1, d_embedding=state_dim + act_dim + 1, n_layer=n_layers, n_head=n_heads)
        self.transformer = transformers.TransfoXLModel(config)
        self.linear_0 = nn.Linear(state_dim + act_dim + 1, state_dim + 2 * act_dim + 1)

        model_ckpt = "microsoft/beit-base-patch16-224-pt22k-ft22k"
        self.image_dim=image_dim
        self.image_processor = transformers.AutoImageProcessor.from_pretrained(model_ckpt, do_resize=True, device=device)
        self.vit = transformers.BeitModel.from_pretrained(model_ckpt).to(device=device)

        self.l2_loss = nn.MSELoss()
        self.optim = torch.optim.AdamW(self.parameters(), lr=0.001)

    def forward(self, *args, **kwargs):
        x = self.transformer(*args, **kwargs)
        x = F.normalize(F.relu(x.last_hidden_state[:, -1:, :]))
        x = self.linear_0(x)
        return x

    # def get_mu_sigma(self, action_preds):
    #     action_preds = action_preds.squeeze()
    #     mid = int(len(action_preds) / 2)
    #     mu = action_preds[:mid]
    #     sigma = action_preds[mid:]
    #     return mu, sigma

    def split(self, x):
        # state, action, reward
        s = x[:, :, :self.state_dim]
        a = x[:, :, self.state_dim:self.state_dim + 2 * self.act_dim]
        r = x[:, :, self.state_dim + 2 * self.act_dim:]
        return s, a, r

    def action_dist(self, action_preds):
        mid = int(action_preds.shape[-1] / 2)
        return Normal(loc=action_preds[:, :, :mid], scale=torch.maximum(action_preds[:, :, mid:], torch.fill(action_preds[:, :, mid:], 0.01)))

    def prob(self, dist, actions):
        return dist.log_prob(actions)

    def proc_state(self, o):
        with torch.inference_mode():
            o = torch.from_numpy(o).to(dtype=self.dtype, device=self.device).reshape(self.image_dim)
            o = self.image_processor(o, return_tensors="pt").pixel_values.to(device=self.device)
            e = self.vit(o).to_tuple()
            e = e[1].unsqueeze(0)

        return e.clone()

    def loss(self, actions, prob, states, state_preds, rtgs, rtg_preds):
        total_return = rtgs.max() # rtgs - rtgs.mean()
        return -(prob * total_return).sum() + torch.log(self.l2_loss(states, state_preds)) + torch.log(self.l2_loss(rtgs, rtg_preds)) + torch.max(actions.abs())

    def train_iter(self, hist):
        self.optim.zero_grad()

        # do it right
        prob = self.action_dist(hist.actions, 0.01).log_prob(hist.actions)
        loss = self.loss(hist.actions, prob, hist.states, hist.state_preds, hist.rtgs, hist.rtg_preds)

        loss.backward()
        self.optim.step()

        return loss
