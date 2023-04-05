import os
import torch as T
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1E-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        
        self.reset()

    def __call__(self):
        x = self.x_previous + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


















class PPOTrainer():
    def __init__(self, actor, critic, ppo_clip_val=0.2, target_kl_div=0.01, max_policy_train_iters=80, value_train_iters=80, policy_lr=3E-4, value_lr=1E-2):
        self.actor = actor
        self.critic = critic
        self.ppo_clip_val = ppo_clip_val
        self.target_kl_div = target_kl_div
        self.max_policy_train_iters = max_policy_train_iters
        self.value_train_iters = value_train_iters
        
        self.policy_optim = optim.RAdam(self.actor.parameters(), lr=policy_lr)
        self.critic_optim = optim.RAdam(self.critic.parameters(), lr=value_lr)

    def train(self, log_probs, old_log_probs, gaes):
        for _ in range(self.max_policy_train_iters):
            self.policy_optim.zero_grad()

            policy_ratio = torch.exp(log_probs - old_log_probs)
            clipped_ratio = torch.clip(policy_ratio, 1 - self.ppo_clip_val, 1 + self.ppo_clip_val)

            clipped_loss = clipped_ratio * gaes
            full_loss = policy_ratio * gaes

            policy_loss = -torch.min(full_loss, clipped_loss).mean()

            policy_loss.backward()

            self.policy_optim.step()

            kl_div = (old_log_probs - log_probs).mean()

            if kl_div >= self.target_kl_div:
                break
            
    def train_critic(self, states, returns):
        for _ in range(self.value_train_iters):
            self.critic_optim.zero_grad()

            values = self.critic(states)
            value_loss = (returns - values) ** 2
            value_loss = value_loss.mean()

            value_loss.backward()

            self.critic_optim.step()
