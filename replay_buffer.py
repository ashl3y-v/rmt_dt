import torch as T
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer(nn.Module):
    def __init__(
        self,
        s=None,
        a=None,
        r=None,
        n_env=1,
        d_state=768,
        d_act=3,
        d_reward=1,
        dtype=T.bfloat16,
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.n_env = n_env
        self.d_state = d_state
        self.d_act = d_act
        self.d_reward = d_reward

        self.s = (
            s
            if s != None
            else T.zeros(
                [n_env, 1, d_state],
                dtype=dtype,
                device=device,
            )
        )
        self.a = (
            a
            if a != None
            else T.zeros(
                [n_env, 1, d_act],
                dtype=dtype,
                device=device,
            )
        )
        self.r = (
            r
            if r != None
            else T.zeros(
                [n_env, 1, d_reward],
                dtype=dtype,
                device=device,
            )
        )

    def predict(self, model, mask=None):
        (s_hat, a) = model(
            s=self.s,
            a=self.a,
            mask=mask,
        )

        return s_hat, a

    # save proper stuff, backwards update Rs, format right
    def append(self, s, a, r):
        assert a.dim() == 3 or a.dim() == 2
        assert r.dim() == 3 or r.dim() == 2 or r.dim() == 1
        assert s.dtype == a.dtype == r.dtype == self.dtype
        if a.dim() == 2:
            a = a.unsqueeze(1)
        if r.dim() == 2:
            r = r.unsqueeze(1)
        elif r.dim() == 1:
            r = r.unsqueeze(0).unsqueeze(1)

        self.s = T.cat([self.s, s], dim=1)

        self.a = T.cat([self.a, a], dim=1)

        self.r = T.cat([self.r, r], dim=1)

    def length(self):
        return self.s.shape[1]

    def artg_update(self):
        length = self.r.shape[1]
        total_reward = self.r.sum(dim=1)
        av_r = total_reward / length
        self.artg = T.zeros(self.r.shape, dtype=self.dtype, device=self.device)
        remaining_reward = total_reward  # clone ?
        for i in range(length):
            self.artg[:, i, :] = remaining_reward / (length - i)
            remaining_reward -= self.r[:, i, :]

        return total_reward, av_r

    def clear(self):
        self.s = T.zeros(
            [self.n_env, 1, self.d_state],
            dtype=self.dtype,
            device=self.device,
        )
        self.a = T.zeros(
            [self.n_env, 1, self.d_act],
            dtype=self.dtype,
            device=self.device,
        )
        self.r = T.zeros(
            [self.n_env, 1, self.d_reward],
            dtype=self.dtype,
            device=self.device,
        )
        self.artg = None

    # def compress_seq(self, seq, dim=0):
    #     n_blocks = (seq.shape[dim] - self.max_size) // self.block_size
    #     blocks = seq[:, : self.block_size * n_blocks, :]
    #     seq = seq[:, self.block_size * n_blocks :, :]
    #     blocks = T.stack(T.split(blocks, self.block_size, dim=dim), dim=dim)
    #     compressed = blocks.mean(dim=dim)
    #
    #     return T.cat([compressed, seq], dim=dim)
    #
    # def compress(self):
    #     if self.length() > self.max_size:
    #         self.states = self.compress_seq(self.states, dim=1)
    #         self.actions = self.compress_seq(self.actions, dim=1)
    #         self.rewards = (
    #             self.compress_seq(self.rewards.unsqueeze(0), dim=1)
    #             .squeeze()
    #             .unsqueeze(-1)
    #         )
    #         self.R_preds = self.compress_seq(self.R_preds, dim=1)


if __name__ == "__main__":
    dtype = T.bfloat16
    device = "cuda" if T.cuda.is_available() else "cpu"
    replay_buffer = ReplayBuffer(
        r=T.tensor([[0, 1, 2, 3, 4], [5, 4, 3, 2, 1]]).reshape([2, 5, 1]),
        dtype=dtype,
        device=device,
    )
    replay_buffer.artg_update()
    print(replay_buffer.artg)
