import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Critic(nn.Module):
    def __init__(self, state_dim=768, act_dim=3, reward_dim=1, fc_size=500, lr=3E-4, name="critic", chkpt_dir="tmp/ddpg", dtype=T.float32, device="cpu"):
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.reward_dim = reward_dim

        self.checkpoint_file = os.path.join(chkpt_dir, name+"_ddpg")

        self.input_state = nn.Linear(state_dim, fc_size)
        self.input_action = nn.Linear(act_dim, fc_size)
        self.linear_1 = nn.Linear(fc_size, fc_size)
        self.linear_2 = nn.Linear(fc_size, fc_size)
        self.linear_3 = nn.Linear(fc_size, fc_size)
        self.linear_4 = nn.Linear(fc_size, reward_dim)

        self.l2_loss = nn.MSELoss()
        self.optim = T.optim.RAdam(self.parameters(), lr=lr)

        self.to(self.device)

    def forward(self, s, a):
        x = F.relu(F.normalize(self.input_state(s))) + F.relu(F.normalize(self.input_action(a)))
        x = x + self.linear_1(x)
        x = F.dropout(x, p=0.25)
        x = x + F.relu(F.normalize(x + self.linear_2(x)))
        x = x + F.relu(F.normalize(self.linear_3(x)))
        x = F.dropout(x, p=0.25)
        x = self.linear_4(x)

        return x

    def loss(self, rtgs, rtg_preds):
        return self.l2_loss(rtgs.squeeze(), rtg_preds.squeeze())

    def train_iter(self, rtgs, rtg_preds):
        self.optim.zero_grad()

        loss = self.loss(rtgs, rtg_preds)

        loss.backward()
        self.optim.step()

        return loss

    def save(self):
        print("Saving critic")
        T.save(self.state_dict(), self.checkpoint_file)

    def load(self):
        print("Loading critic")
        self.load_state_dict(T.load(self.checkpoint_file))
