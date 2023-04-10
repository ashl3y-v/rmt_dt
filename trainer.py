import torch as T
from torch import nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_optimizer import Ranger21

class Trainer(nn.Module):
    def __init__(self, params, lr=3E-4, iterations=1000):
        super().__init__()
        self.optim = Ranger21(params, num_iterations=iterations, lr=lr) # optim.AdamW(params, lr=max_lr, amsgrad=True)
        # self.lr_scheduler = lr_scheduler.CyclicLR(self.optim, base_lr, max_lr)

        self.mse_loss = nn.MSELoss(reduction='mean')

    def loss(self, R_preds, Rs):
        return -Rs.mean() + self.mse_loss(R_preds.squeeze(), Rs.squeeze())

    def learn(self, replay_buffer):
        self.optim.zero_grad()
        loss = self.loss(replay_buffer.R_preds, replay_buffer.Rs)
        loss.backward()
        self.optim.step()

        return loss

