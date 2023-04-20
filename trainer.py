import torch as T
from torch import nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lr_scheduler


class Trainer(nn.Module):
    def __init__(
        self,
        params,
        lr=3e-4,
        steps_P=1,
        steps_R=5,
        epochs=100,
        clip=0.25,
        scaler=T.cuda.amp.grad_scaler.GradScaler(),
        autocast=False,
        use_lr_schedule=False,
    ):
        super().__init__()

        self.params = params

        self.steps_P = steps_P
        self.steps_R = steps_R

        self.autocast = autocast
        if autocast:
            self.scaler = scaler

        self.clip = clip

        self.optim = T.optim.AdamW(params, lr=lr)

        self.use_lr_schedule = use_lr_schedule
        if use_lr_schedule:
            self.lr_scheduler = lr_scheduler.OneCycleLR(
                self.optim,
                max_lr=lr,
                epochs=epochs,
                steps_per_epoch=steps_P + steps_R,
            )

        self.huber = nn.HuberLoss(delta=1)

    def learn(self, replay_buffer):
        for _ in range(self.steps_R):
            self.optim.zero_grad(set_to_none=True)
            if self.autocast:
                with T.cuda.amp.autocast():
                    R_loss = self.huber(replay_buffer.R_preds, replay_buffer.Rs)
            else:
                R_loss = self.huber(replay_buffer.R_preds, replay_buffer.Rs)
            self.scaler.scale(R_loss).backward(retain_graph=True)
            self.scaler.unscale_(self.optim)
            T.nn.utils.clip_grad_norm_(self.params, self.clip)
            self.scaler.step(self.optim)
            self.scaler.update()
            if self.use_lr_schedule:
                self.lr_scheduler.step()

        for _ in range(self.steps_P):
            self.optim.zero_grad(set_to_none=True)
            if self.autocast:
                with T.cuda.amp.autocast():
                    P_loss = -replay_buffer.R_preds.mean()
            else:
                P_loss = -replay_buffer.R_preds.mean()
            self.scaler.scale(P_loss).backward()
            self.scaler.unscale_(self.optim)
            T.nn.utils.clip_grad_norm_(self.params, self.clip)
            self.scaler.step(self.optim)
            self.scaler.update()
            if self.use_lr_schedule:
                self.lr_scheduler.step()

        return P_loss, R_loss
