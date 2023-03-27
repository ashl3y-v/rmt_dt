import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers


class Critic(nn.Module):
    def __init__(self, state_dim=768, act_dim=3, reward_dim=1, nhead=3, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.reward_dim = reward_dim
        self.nhead = nhead

        encoder_layer = nn.TransformerEncoderLayer(d_model=state_dim + act_dim, nhead=nhead).to(dtype=dtype, device=device)
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=6).to(dtype=dtype, device=device)
        self.get_reward = nn.Linear(state_dim + act_dim, 1).to(dtype=dtype, device=device)

        self.l2_loss = nn.MSELoss()
        self.optim = torch.optim.AdamW(self.parameters(), lr=0.05)

    def forward(self, x):
        x = self.transformer(x)
        x = self.get_reward(x)

        return x

    def loss(self, rtgs, rtg_preds):
        return self.l2_loss(rtgs.squeeze(), rtg_preds.squeeze())

    def train_iter(self, rtgs, rtg_preds):
        self.optim.zero_grad()

        loss = self.loss(rtgs, rtg_preds)

        loss.backward()
        self.optim.step()

        return loss
