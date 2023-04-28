import torch as T
from torch import nn
from torch.nn import functional as F
from ranger21 import Ranger21

# import torch.optim.lr_scheduler as lr_scheduler


class Trainer(nn.Module):
    def __init__(
        self,
        params,
        lr=3e-4,
        steps_P=1,
        steps_R=3,
        epochs=100,
        # scaler=T.cuda.amp.grad_scaler.GradScaler(),
        # use_lr_schedule=False,
    ):
        super().__init__()

        self.params = params

        self.steps_P = steps_P
        self.steps_R = steps_R

        self.optim = Ranger21(
            params=params,
            lr=lr,
            num_batches_per_epoch=steps_R + steps_P,
            num_epochs=epochs,
        )

        # self.use_lr_schedule = use_lr_schedule
        # if use_lr_schedule:
        #     self.lr_scheduler = lr_scheduler.OneCycleLR(
        #         self.optim,
        #         max_lr=lr,
        #         epochs=epochs,
        #         steps_per_epoch=steps_P + steps_R,
        #     )

        self.huber = nn.HuberLoss(delta=1)

    def learn(self, replay_buffer):
        # manual gradient clipping needed? <- ranger
        for _ in range(self.steps_R):
            self.optim.zero_grad(set_to_none=True)
            artg_loss = self.huber(replay_buffer.artg_hat, replay_buffer.artg)
            artg_loss.backward(retain_graph=True)
            # T.nn.utils.clip_grad_norm_(self.params, self.clip)
            self.optim.step()
            # if self.use_lr_schedule:
            #     self.lr_scheduler.step()

        for _ in range(self.steps_P):
            self.optim.zero_grad(set_to_none=True)
            policy_loss = -replay_buffer.artg_hat.mean()
            policy_loss.backward(retain_graph=True)
            # T.nn.utils.clip_grad_norm_(self.params, self.clip)
            self.optim.step()
            # if self.use_lr_schedule:
            #     self.lr_scheduler.step()

        return artg_loss, policy_loss
