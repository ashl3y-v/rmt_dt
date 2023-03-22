import torch
from matplotlib import pyplot as plt

a = torch.ones([10, 10])

b = torch.fill(torch.zeros([10, 10]), 2)

print(a, torch.zeros(a.shape))
