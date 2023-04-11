import torch as T
from torch import nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lr_scheduler

class Trainer(nn.Module):
    def __init__(self, params, max_lr=1E-3, steps_P=1, steps_R=3, epochs=100):
        super().__init__()

        self.steps_P = steps_P
        self.steps_R = steps_R

        self.optim = T.optim.AdamW(params, lr=0)
        self.lr_scheduler = lr_scheduler.OneCycleLR(self.optim, max_lr=max_lr, epochs=epochs, steps_per_epoch=steps_P+steps_R)

        self.huber = nn.HuberLoss(delta=1)

    def learn(self, replay_buffer):
        for _ in range(self.steps_R):
            self.optim.zero_grad()
            R_loss = self.huber(replay_buffer.R_preds, replay_buffer.Rs)
            R_loss.backward(retain_graph=True)
            self.optim.step()
            self.lr_scheduler.step()

        for _ in range(self.steps_P):
            self.optim.zero_grad()
            P_loss = -replay_buffer.R_preds.mean()
            P_loss.backward()
            self.optim.step()

        return P_loss, R_loss
