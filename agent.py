import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import OwlViTFeatureExtractor
from replay_buffer import ReplayBuffer, init_replay_buffer
from vit import ViT
from dt import DecisionTransformer
from critic import Critic
from trainer import OUActionNoise

class Agent(nn.Module):
    def __init__(self, lr_actor, lr_critic, state_dim, act_dim, reward_dim, image_dim, tau, env, gamma=0.99, fc_size=500, dtype=T.float32, device="cpu"):
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.reward_dim = reward_dim
        self.tau = tau
        self.env = env
        self.gamma = gamma
        # needed ?
        self.fc_size = fc_size
        self.attention_mask = T.zeros(1, 1, device=device, dtype=dtype)

        self.vit = ViT(image_dim=image_dim, dtype=dtype, device=device)

        observation, _ = env.reset()
        self.replay_buffer = init_replay_buffer(self.vit(observation), act_dim, state_dim, dtype=dtype, device=device)

        self.actor = DecisionTransformer(state_dim, act_dim, lr=lr_actor, dtype=dtype, name="actor", device=device)

        self.target_actor = DecisionTransformer(state_dim, act_dim, lr=lr_actor, dtype=dtype, name="target_actor", device=device)

        self.critic = Critic(state_dim, act_dim, reward_dim, fc_size, lr_critic, dtype=dtype, name="critic", device=device)

        self.target_critic = Critic(state_dim, act_dim, reward_dim, fc_size, lr_critic, name="target_critic", dtype=dtype, device=device)

        self.noise = OUActionNoise(mu=T.zeros(2 * act_dim, dtype=dtype, device=device))

        self.update_network_parameters(tau=1)

    def act(self, observation):
        # no stats for batch norm
        self.actor.eval()
        state = self.vit(observation).reshape([1, 1, self.state_dim])
        state_pred, action_pred, rtg_pred = self.replay_buffer.predict(self.actor, self.attention_mask)
        action_pred += self.noise().to(dtype=self.dtype, device=self.device)

        self.actor.train()

        action, prob = self.actor.sample(action_pred)

        return  action.cpu().detach().numpy(), prob

    def append(self, state, action, reward, new_state, rtg_pred, timestep_delta, done):
        self.replay_buffer.append(state, action, reward, new_state, rtg_pred, timestep_delta, done)

    def learn(self):
