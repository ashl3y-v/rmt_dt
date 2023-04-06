import torch as T
from torch import nn

class Trainer(nn.Module):
    def __init__(self):
        self.x = 1

    def loss(self, actions, log_probs, rtgs):
        total_return = rtgs.max() # rtgs - rtgs.mean()
        return -(probs * total_return).sum() + actions.abs().sum()

    def learn(self, actor, critic, replay_buffer):
        loss = self.loss(acts, probs, hist.rtgs)

        loss.backward()
        self.optim.step()

        return loss

