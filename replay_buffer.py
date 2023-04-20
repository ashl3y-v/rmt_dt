import torch as T
from torch import nn


class ReplayBuffer(nn.Module):
    def __init__(
        self,
        states,
        actions,
        rewards,
        R_preds,
        timestep=0,
        num_envs=1,
        block_size=10,
        max_size=150,
        dtype=T.float32,
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.R_preds = R_preds

        if isinstance(timestep, int):
            self.timestep = T.tensor(
                [timestep] * num_envs, device=device, dtype=T.long
            ).reshape(num_envs, 1)
        else:
            self.timestep = timestep.to(dtype=T.long, device=device).reshape(
                num_envs, 1
            )

        self.block_size = block_size
        self.max_size = max_size

    def predict(self, model, attention_mask):  # use attention_mask
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
    def append(self, state, action, reward, R_pred, timestep_delta=1, compress=False):
        self.states = T.cat([self.states, state], dim=1)

        assert action.dim() == 3 or action.dim() == 2
        if action.dim() == 2:
            action = action.reshape([action.shape[0], 1, action.shape[-1]])

        self.actions = T.cat([self.actions, action], dim=1)

        self.rewards = T.cat([self.rewards, reward], dim=0)

        self.R_preds = T.cat([self.R_preds, R_pred], dim=1)

        self.timestep = self.timestep + T.tensor(
            timestep_delta, device=self.device, dtype=T.long
        ).reshape(1, 1)

        if compress:
            self.compress()

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

    def compress_seq(self, seq, dim=0):
        n_blocks = (seq.shape[dim] - self.max_size) // self.block_size
        blocks = seq[:, : self.block_size * n_blocks, :]
        seq = seq[:, self.block_size * n_blocks :, :]
        blocks = T.stack(T.split(blocks, self.block_size, dim=dim), dim=dim)
        compressed = blocks.mean(dim=dim)

        return T.cat([compressed, seq], dim=dim)

    def compress(self):
        if self.length() > self.max_size:
            self.states = self.compress_seq(self.states, dim=1)
            self.actions = self.compress_seq(self.actions, dim=1)
            self.rewards = (
                self.compress_seq(self.rewards.unsqueeze(0), dim=1)
                .squeeze()
                .unsqueeze(-1)
            )
            self.R_preds = self.compress_seq(self.R_preds, dim=1)


def init_replay_buffer(
    state,
    act_dim,
    state_dim,
    TARGET_RETURN=9999,
    num_envs=1,
    block_size=10,
    max_size=200,
    dtype=T.float16,
    device="cuda",
):
    actions = T.zeros([num_envs, 1, act_dim], device=device, dtype=dtype)
    rewards = T.zeros(num_envs, 1, device=device, dtype=dtype)

    states = state.reshape(num_envs, 1, state_dim).to(device=device, dtype=dtype)

    R_preds = T.tensor([TARGET_RETURN] * num_envs, device=device, dtype=dtype).reshape(
        num_envs, 1, 1
    )

    return ReplayBuffer(
        states,
        actions,
        rewards,
        R_preds,
        num_envs=num_envs,
        block_size=block_size,
        max_size=max_size,
        dtype=dtype,
        device=device,
    )


if __name__ == "__main__":
    replay_buffer = init_replay_buffer(T.zeros(100), act_dim=3, state_dim=100)
    print(replay_buffer)
