import torch as T


class ReplayBuffer():
    def __init__(self, states, actions, rewards, R_preds, timestep=0, dtype=T.float32, device="cpu"):
        self.device = device
        self.dtype = dtype

        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.R_preds = R_preds

        if type(timestep) == type(0):
            self.timestep = T.tensor(timestep, device=device, dtype=T.long).reshape(1, 1)
        else:
            self.timestep = timestep.to(dtype=T.long, device=device).reshape(1, 1)

    def predict(self, model, attention_mask): # use attention_mask
        # with torch.inference_mode():
        state_preds, action_preds, R_preds = model(
            states=self.states,
            actions=self.actions,
            rewards=self.rewards,
            returns_to_go=self.R_preds,
            timesteps=self.timestep,
            attention_mask=attention_mask,
            return_dict=False,
        )

        return state_preds[:, -1:, :], action_preds[:, -1:, :], R_preds[:, -1:, :]

    # save proper stuff, backwards update Rs, format right
    def append(self, state, action, reward, R_pred, timestep_delta=1):
        self.states = T.cat([self.states, state], dim=1)

        self.actions = T.cat([self.actions, action], dim=1)

        self.rewards = T.cat([self.rewards, reward], dim=0)

        self.R_preds = T.cat([self.R_preds, R_pred], dim=1)

        self.timestep = self.timestep + T.tensor(timestep_delta, device=self.device, dtype=T.long).reshape(1, 1)

    def length(self):
        return self.states.shape[1]

    def R_update(self):
        total_reward = self.rewards.sum()
        av_r = total_reward / self.length()
        self.Rs = T.zeros(self.R_preds.shape, device=self.device)
        remaining_reward = total_reward.item()
        for i in range(self.Rs.shape[-2]):
            self.Rs[:, i, 0] = remaining_reward / (self.length() - i)
            remaining_reward -= self.rewards[i, 0]

        # print("total_reward", total_reward)
        return total_reward, av_r

    def compress(self):
        pass

def init_replay_buffer(state, act_dim, state_dim, TARGET_RETURN=9999, dtype=T.float32, device="cpu"):
    actions = T.zeros([1, 1, act_dim], device=device, dtype=dtype)
    rewards = T.zeros(1, 1, device=device, dtype=dtype)

    states = state.reshape(1, 1, state_dim).to(device=device, dtype=dtype)

    R_preds = T.tensor(TARGET_RETURN, device=device, dtype=dtype).reshape(1, 1, 1)

    return ReplayBuffer(states, actions, rewards, R_preds, dtype=dtype, device=device)
