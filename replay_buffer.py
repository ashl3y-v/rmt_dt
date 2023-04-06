import torch as T


class ReplayBuffer():
    def __init__(self, states, actions, rewards, rtg_preds, timestep=0, length=1, dtype=T.float32, device="cpu"):
        self.device = device
        self.dtype = dtype

        self.length = length
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.rtg_preds = rtg_preds

        if type(timestep) == type(0):
            self.timestep = T.tensor(timestep, device=device, dtype=T.long).reshape(1, 1)
        else:
            self.timestep = timestep.to(dtype=T.long, device=device).reshape(1, 1)

    def predict(self, model, attention_mask): # use attention_mask
        # with torch.inference_mode():
        state_preds, action_preds, rtg_preds = model(
            states=self.states,
            actions=self.actions,
            rewards=self.rewards,
            returns_to_go=self.rtg_preds,
            timesteps=self.timestep,
            attention_mask=attention_mask,
            return_dict=False,
        )

        return state_preds[:, -1:, :], action_preds[:, -1:, :], rtg_preds[:, -1:, :]

    # save proper stuff, backwards update RTG, format right
    def append(self, state, action, reward, rtg_pred, timestep_delta=1):
        self.states = T.cat([self.states, state], dim=1)

        self.actions = T.cat([self.actions, action], dim=1)

        self.rewards = T.cat([self.rewards, reward], dim=0)

        self.rtg_preds = T.cat([self.rtg_preds, rtg_pred], dim=1)

        self.timestep = self.timestep + T.tensor(timestep_delta, device=self.device, dtype=T.long).reshape(1, 1)

    def rtg_update(self):
        total_reward = self.rewards.sum()
        self.rtgs = T.zeros(self.rtg_preds.shape, device=self.device)
        remaining_reward = total_reward.item()
        for i in range(self.rtgs.shape[-2]):
            self.rtgs[:, i, 0] = remaining_reward
            remaining_reward -= self.rewards[i, 0]

        # print("total_reward", total_reward)
        return total_reward


def init_replay_buffer(state, act_dim, state_dim, TARGET_RETURN=10000, dtype=T.float32, device="cpu"):
    actions = T.zeros([1, 1, act_dim], device=device, dtype=dtype)
    rewards = T.zeros(1, 1, device=device, dtype=dtype)

    states = state.reshape(1, 1, state_dim).to(device=device, dtype=dtype)

    rtg_preds = T.tensor(TARGET_RETURN, device=device, dtype=dtype).reshape(1, 1, 1)

    return ReplayBuffer(states, actions, rewards, rtg_preds, dtype=dtype, device=device)
