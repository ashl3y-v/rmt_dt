import torch


class Hist():
    def __init__(self, states, state_preds, actions, rewards, rtg_preds, timestep, device="cpu"):
        self.device = device

        self.states = states
        self.state_preds = state_preds
        self.actions = actions
        self.rewards = rewards
        self.rtg_preds = rtg_preds
        self.timestep = timestep

    def predict(self, model, attention_mask):
        # with torch.inference_mode():
        state_preds, action_preds, return_preds = model(
            states=self.states,
            actions=self.actions,
            rewards=self.rewards,
            returns_to_go=self.rtg_preds,
            timesteps=self.timestep,
            attention_mask=attention_mask,
            return_dict=False,
        )

        state_pred = state_preds[:, -1:, :]
        action_pred = action_preds[:, -1:, :]
        rtg_pred = return_preds[:, -1:, :]

        return state_pred, action_pred, rtg_pred

    # save proper stuff, backwards update RTG, format right
    def append(self, state, state_pred, action, reward, rtg_pred, timestep_delta):
        self.states = torch.cat([self.states, state], dim=1)
        self.state_preds = torch.cat([self.state_preds, state_pred], dim=1)

        self.actions = torch.cat([self.actions, action], dim=1)

        self.rewards = torch.cat([self.rewards, reward], dim=0)

        self.rtg_preds = torch.cat([self.rtg_preds, rtg_pred], dim=1)

        self.timestep = self.timestep + torch.tensor(timestep_delta, device=self.device, dtype=torch.long).reshape(1, 1)

    def rtg_update(self):
        total_reward = self.rewards.sum()
        self.rtgs = torch.zeros(self.rtg_preds.shape, device=self.device)
        remaining_reward = total_reward.item()
        for i in range(self.rtgs.shape[-2]):
            self.rtgs[:, i, 0] = remaining_reward
            remaining_reward -= self.rewards[i, 0]

        # print("total_reward", total_reward)
        return total_reward
