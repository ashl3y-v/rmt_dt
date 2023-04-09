import torch as T
from torch import nn
from torch.nn import functional as F
from torch import optim

class Trainer(nn.Module):
    def __init__(self, actor_params, critic_params, lr_actor=3E-4, lr_critic=3E-4):
        super().__init__()
        self.actor_optim = optim.AdamW(actor_params, lr=lr_actor, amsgrad=True)

        self.critic_optim = optim.AdamW(critic_params, lr=lr_critic, amsgrad=True)

        self.critic_mse_loss = nn.MSELoss(reduction='mean')

    def actor_loss(self, Qs):
        return -Qs.mean()

    def critic_loss(self, Qs, rtgs):
        return self.critic_mse_loss(Qs.squeeze(), rtgs.squeeze())

    def learn(self, actor, critic, replay_buffer):
        Qs = critic(replay_buffer.states, replay_buffer.actions)

        actor.eval()
        self.critic_optim.zero_grad()
        critic_loss = self.critic_loss(Qs, replay_buffer.rtgs)
        critic_loss.backward(retain_graph=True)
        self.critic_optim.step()

        actor.train()
        self.actor_optim.zero_grad()
        actor_loss = self.actor_loss(Qs)
        actor_loss.backward()
        self.actor_optim.step()

        return critic_loss, actor_loss

