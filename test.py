import torch
from torch import nn
from matplotlib import pyplot as plt
from dt import DecisionTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DecisionTransformer(act_dim=3, device=device)

action_preds = torch.randn([3])
action = torch.randn([3]) + torch.tensor([2, 2, 2])
r = model.prob(action_preds, action)

print(r)
