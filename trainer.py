import torch as T
from torch import nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lr_scheduler


class Trainer(nn.Module):
    def __init__(
        self,
        params,
        max_lr=1e-3,
        steps_P=1,
        steps_R=3,
        epochs=100,
        clip=0.1,
        scaler=T.cuda.amp.grad_scaler.GradScaler(),
    ):
        super().__init__()

        self.params = params

        self.steps_P = steps_P
        self.steps_R = steps_R

        self.scaler = scaler

        self.clip = clip

        self.optim = T.optim.AdamW(params, lr=0)
        self.lr_scheduler = lr_scheduler.OneCycleLR(
            self.optim, max_lr=max_lr, epochs=epochs, steps_per_epoch=steps_P + steps_R
        )

        self.huber = nn.HuberLoss(delta=1)

    def learn(self, replay_buffer):
        for _ in range(self.steps_R):
            self.optim.zero_grad(set_to_none=True)
            with T.cuda.amp.autocast():
                R_loss = self.huber(replay_buffer.R_preds, replay_buffer.Rs)
            self.scaler.scale(R_loss).backward(retain_graph=True)
            self.scaler.unscale_(self.optim)
            T.nn.utils.clip_grad_norm_(self.params, self.clip)
            self.scaler.step(self.optim)
            self.scaler.update()
            self.lr_scheduler.step()

        for _ in range(self.steps_P):
            self.optim.zero_grad(set_to_none=True)
            with T.cuda.amp.autocast():
                P_loss = -replay_buffer.R_preds.mean()
            self.scaler.scale(P_loss).backward()
            self.scaler.unscale_(self.optim)
            T.nn.utils.clip_grad_norm_(self.params, self.clip)
            self.scaler.step(self.optim)
            self.scaler.update()
            self.lr_scheduler.step()

        return P_loss, R_loss
