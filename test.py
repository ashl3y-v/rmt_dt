import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import transformers
from dt import DecisionTransformer

model = DecisionTransformer()

i = torch.randn([1, 5, 775])

r = model(inputs_embeds=i)

print(r.rhape)
