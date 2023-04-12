import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from replay_buffer import ReplayBuffer, init_replay_buffer
from vit import ViT
from dt import DecisionTransformer
from critic import Critic
from trainer import OUActionNoise


class Agent(nn.Module):
    def __init__(
        self,
        lr_actor,
        lr_critic,
        state_dim,
        act_dim,
        reward_dim,
        image_dim,
        tau,
        env,
        gamma=0.99,
        fc_size=500,
        dtype=T.float32,
        device="cpu",
    ):
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
        self.replay_buffer = init_replay_buffer(
            self.vit(observation), act_dim, state_dim, dtype=dtype, device=device
        )

        self.actor = DecisionTransformer(
            state_dim, act_dim, lr=lr_actor, dtype=dtype, name="actor", device=device
        )

        self.target_actor = DecisionTransformer(
            state_dim,
            act_dim,
            lr=lr_actor,
            dtype=dtype,
            name="target_actor",
            device=device,
        )

        self.critic = Critic(
            state_dim,
            act_dim,
            reward_dim,
            fc_size,
            lr_critic,
            dtype=dtype,
            name="critic",
            device=device,
        )

        self.target_critic = Critic(
            state_dim,
            act_dim,
            reward_dim,
            fc_size,
            lr_critic,
            name="target_critic",
            dtype=dtype,
            device=device,
        )

        self.noise = OUActionNoise(mu=T.zeros(2 * act_dim, dtype=dtype, device=device))

        self.update_network_parameters(tau=1)

    def act(self):
        # no stats for batch norm
        self.actor.eval()
        state_pred, action_pred, rtg_pred = self.replay_buffer.predict(
            self.actor, self.attention_mask
        )
        action_pred += self.noise().to(dtype=self.dtype, device=self.device)

        self.actor.train()

        action, prob = self.actor.sample(action_pred)

        return action.cpu().detach().numpy(), prob

    def append(self, state, action, reward, new_state, rtg_pred, timestep_delta, done):
        self.replay_buffer.append(
            state, action, reward, new_state, rtg_pred, timestep_delta, done
        )

    def learn(self):
        self.target_actor.eval()
        self.target_critic.eval()
        self.actor.eval()
        self.critic.eval()

        target_actions = self.target_actor(new_state)
        critic_value = self.critic.forward(states, actions)

        target = reward + self.gamma * new_critic_value * done

        self.critic.train()
        self.critic.optim.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optim.step()

        self.critic.eval()
        self.actor.optim.zero_grad()
        state_pred, action_pred, rtg_pred = self.replay_buffer.predict(
            self.actor, self.attention_mask
        )
        self.actor.train()
        actor_loss = -self.critic.forward(state, action)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optim.step()

        # may need to constrain init params cause of sensitivity of learning method

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_state_dict = dict(self.actor.named_parameters())
        target_actor_state_dict = dict(self.target_actor.named_parameters())
        critic_state_dict = dict(self.critic.named_parameters())
        target_critic_state_dict = dict(self.target_critic.named_parameters())

        for name in actor_state_dict:
            actor_state_dict[name] = (
                tau * actor_state_dict[name].clone()
                + (1 - tau) * target_actor_state_dict[name].clone()
            )

        self.target_actor.load_state_dict(actor_state_dict)

        for name in critic_state_dict:
            critic_state_dict[name] = (
                tau * critic_state_dict[name].clone()
                + (1 - tau) * target_critic_state_dict[name].clone()
            )

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save()
        self.target_actor.save()
        self.critic.save()
        self.target_critic.save()

    def load_models(self):
        self.actor.load()
        self.target_actor.load()
        self.critic.load()
        self.target_critic.load()
