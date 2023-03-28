import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        if critic:
            self.critic = torch.load(critic).to(dtype=dtype, device=device)
        else:
            self.critic = Critic(state_dim, act_dim, 1, nhead=3, dtype=dtype, device=device)
        self.optim = torch.optim.AdamW(self.parameters(), lr=0.003)

    def forward(self, **kwargs):
        return self.transformer(**kwargs)

    def proc_state(self, o):
        with torch.inference_mode():
            o = torch.from_numpy(o).to(dtype=self.dtype, device=self.device).reshape(self.image_dim)
            o = self.image_processor(o, return_tensors="pt").pixel_values.to(device=self.device)
            e = self.vit(o).to_tuple()
            e = e[1].unsqueeze(0)

        return e.clone()

    def loss(self, loss_critic, states, state_preds, rtgs, rtg_preds):
        return torch.log(loss_critic) + torch.log(self.l2_loss(states, state_preds)) + torch.log(self.l2_loss(rtgs, rtg_preds))

    def train_iter(self, hist):
        self.optim.zero_grad()

        # do it right
        critic_rtg_preds = self.critic(torch.cat([hist.states, hist.actions], dim=2))
        loss_critic = -critic_rtg_preds.sum()
        loss = self.loss(loss_critic, hist.states, hist.state_preds, hist.rtgs, hist.rtg_preds)

        loss.backward(retain_graph=True)
        self.optim.step()

        critic_loss = self.critic.train_iter(hist.rtgs, critic_rtg_preds)

        return loss, critic_loss
