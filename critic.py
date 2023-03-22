import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers


class Critic(nn.Module):
    def __init__(self, state_dim=1024, act_dim=4, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.state_dim = state_dim
        self.act_dim = act_dim

        self.linear0 = nn.Linear(state_dim + act_dim, state_dim + act_dim)
        self.linear1 = nn.Linear(state_dim + act_dim, state_dim + act_dim)
        self.linear2 = nn.Linear(state_dim + act_dim, state_dim + 1)

        self.l2_loss = nn.MSELoss()
        self.optim = torch.optim.AdamW(self.parameters(), lr=0.001)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=2)
        x = F.relu(x + self.linear0(x))
        x = F.relu(x + self.linear1(x))
        x = F.relu(self.linear2(x))
        s_n = x[:, :, :self.state_dim]
        r = x[:, :, self.state_dim:]

        return s_n, r

    def loss(self, state, state_pred, reward, reward_pred):
        return self.l2_loss(state, state_pred) + self.l2_loss(reward, reward_pred)

    def train_iter(self, state, state_pred, reward, reward_pred):
        self.optim.zero_grad()

        loss = self.loss(state, state_pred, reward, reward_pred)

        loss.backward()
        self.optim.step()

        return loss


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Critic(state_dim=768, act_dim=3, device=device)

    s = torch.randn([1, 1, 768])
    a = torch.randn([1, 1, 3])

    s_n, r = model(s, a)
    print(s_n.shape, r.shape)
