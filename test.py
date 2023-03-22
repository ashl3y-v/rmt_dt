import torch
from matplotlib import pyplot as plt

device = "cpu"

losses = torch.load("losses.pt").to(device=device).detach()
rewards = torch.load("rewards.pt").to(device=device).detach()

plt.plot(rewards)
plt.show()
