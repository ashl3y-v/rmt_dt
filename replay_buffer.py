import torch as T


class ReplayBuffer():
    def __init__(self, states, actions, probs, rewards, new_states, rtg_preds, terminals, timestep=0, length=1, dtype=T.float32, device="cpu"):
        self.device = device
        self.dtype = dtype

        self.length = length
        self.states = states
        self.new_states = new_states
        self.actions = actions
        self.probs = probs
        self.rewards = rewards
        self.rtg_preds = rtg_preds
        self.terminals = terminals

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
    def append(self, state, action, reward, new_state, rtg_pred, timestep_delta, done):
        self.states = torch.cat([self.states, state], dim=1)

        self.actions = torch.cat([self.actions, action], dim=1)

        # self.probs = torch.cat([self.probs, prob], dim=1)

        self.rewards = torch.cat([self.rewards, reward], dim=0)

        self.new_states = torch.cat([self.new_states, new_state], dim=1)

        self.rtg_preds = torch.cat([self.rtg_preds, rtg_pred], dim=1)

        self.timestep = self.timestep + torch.tensor(timestep_delta, device=self.device, dtype=torch.long).reshape(1, 1)

        self.terminals = torch.cat([self.terminals, done], dim=1)

    def rtg_update(self):
        total_reward = self.rewards.sum()
        self.rtgs = torch.zeros(self.rtg_preds.shape, device=self.device)
        remaining_reward = total_reward.item()
        for i in range(self.rtgs.shape[-2]):
            self.rtgs[:, i, 0] = remaining_reward
            remaining_reward -= self.rewards[i, 0]

        # print("total_reward", total_reward)
        return total_reward


def init_replay_buffer(state, act_dim, state_dim, TARGET_RETURN=10000, dtype=T.float32, device="cpu"):
    actions = T.zeros([1, 1, act_dim], device=self.device, dtype=self.dtype)
    probs = T.ones([1, 1, act_dim], device=self.device, dtype=self.dtype)
    rewards = T.zeros(1, 1, device=self.device, dtype=self.dtype)

    states = state.reshape(1, 1, state_dim).to(device=device, dtype=dtype)
    new_states = state.reshape(1, 1, state_dim).to(device=device, dtype=dtype)

    rtg_preds = T.tensor(TARGET_RETURN, device=device, dtype=dtype).reshape(1, 1, 1)

    timestep = T.tensor(0, device=self.device, dtype=T.long).reshape(1, 1)

    terminals = T.zeros()

    length = 1

    return ReplayBuffer(states, actions, probs, rewards, new_states, rtg_preds, terminals, timestep, length, dtype=dtype, device=device)
