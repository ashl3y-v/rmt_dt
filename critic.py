import torch as T
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, state_dim=768, act_dim=3, fc_size=250, n_layers=5, file="critic.pt", dtype=T.float32, device="cpu"):
        super().__init__()
        self.dtype = dtype
        self.device = device
        
        self.file = file

        self.state_dim = state_dim
        self.act_dim = act_dim

        self.bn_0 = nn.BatchNorm1d(num_features=fc_size)
        self.input = nn.Linear(state_dim + act_dim, fc_size)

        linear_layers = [
            nn.Linear(fc_size, fc_size),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(fc_size),
            nn.Mish()
        ] * n_layers
        self.linears = nn.Sequential(*linear_layers)

        self.output = nn.Linear(fc_size, 1)

        self.to(self.device)

    def forward(self, s, a):
        s = s.squeeze()
        a = a.squeeze()

        x = T.cat([s, a], dim=-1)
        x = self.input(x)
        x = self.bn_0(x)

        x = self.linears(x)

        x = self.output(x)

        return x

    def save(self):
        print("Saving critic")
        T.save(self.state_dict(), self.file)

    def load(self):
        print("Loading critic")
        self.load_state_dict(T.load(self.file))
