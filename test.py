import torch
from torch import nn
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Frog(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

frog = Frog()
print(next(frog.parameters()).device)
frog.cuda()
print(next(frog.parameters()).device)
