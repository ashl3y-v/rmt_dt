from math import sqrt, pi
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import transformers
from critic import Critic


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim=1024, act_dim=4, n_positions=8192, image_dim=[3, 96, 96], critic=None, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.state_dim = state_dim
        self.act_dim = act_dim

        # crashes if length is greater than n_positions
        config = transformers.DecisionTransformerConfig(state_dim=state_dim, act_dim=act_dim, n_positions=n_positions)
        self.transformer = transformers.DecisionTransformerModel(config).to(device=device)

        model_ckpt = "microsoft/beit-base-patch16-224-pt22k-ft22k"
        self.image_dim=image_dim
        self.image_processor = transformers.AutoImageProcessor.from_pretrained(model_ckpt, do_resize=True, device=device)
        self.vit = transformers.BeitModel.from_pretrained(model_ckpt).to(device=device) # return_dict=False

        self.l2_loss = nn.MSELoss()
        self.optim = torch.optim.AdamW(self.parameters(), lr=0.003)

    def forward(self, **kwargs):
        return self.transformer(**kwargs)

    # def get_mu_sigma(self, action_preds):
    #     action_preds = action_preds.squeeze()
    #     mid = int(len(action_preds) / 2)
    #     mu = action_preds[:mid]
    #     sigma = action_preds[mid:]
    #     return mu, sigma

    def action_dist(self, action_preds):
        return Normal(loc=action_preds, scale=1)

    def proc_state(self, o):
        with torch.inference_mode():
            o = torch.from_numpy(o).to(dtype=self.dtype, device=self.device).reshape(self.image_dim)
            o = self.image_processor(o, return_tensors="pt").pixel_values.to(device=self.device)
            e = self.vit(o).to_tuple()
            e = e[1].unsqueeze(0)

        return e.clone()

    def loss(self, prob, states, state_preds, rtgs, rtg_preds):
        advantages = rtgs - rtgs.mean()
        return -prob * advantages + torch.log(self.l2_loss(states, state_preds)) + torch.log(self.l2_loss(rtgs, rtg_preds))

    def train_iter(self, hist):
        self.optim.zero_grad()

        # do it right
        prob = self.prob(hist.actions, hist.actions)
        loss = self.loss(prob, hist.states, hist.state_preds, hist.rtgs, hist.rtg_preds)

        loss.backward()
        self.optim.step()

        return loss
