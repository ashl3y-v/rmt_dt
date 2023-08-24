import torch as T
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer(nn.Module):
    def __init__(
        self,
        s=None,
        a=None,
        r=None,
        artg_hat=None,
        prob=None,
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
        self.artg_hat = (
            artg_hat
            if artg_hat != None
            else T.zeros(
                [n_env, 1, d_reward],
                dtype=dtype,
                device=device,
            )
        )
        self.prob = (
            prob
            if prob != None
            else T.zeros(
                [n_env, 1, 1],
                dtype=dtype,
                device=device,
            )
        )

    def predict(self, model, mask=None):
        s_hat, a, prob, artg_hat = model(
            self.s, self.a, self.r, self.artg_hat, self.s.shape[0], mask=mask
        )

        return s_hat, a, prob, artg_hat

    # save proper stuff, backwards update Rs, format right
    def append(self, s, a, r, artg_hat, prob):
        assert a.dim() == 3 or a.dim() == 2
        assert r.dim() == 3 or r.dim() == 2 or r.dim() == 1
        assert s.dtype == a.dtype == r.dtype == self.dtype
        if a.dim() == 2:
            a = a.unsqueeze(1)
        if r.dim() == 2:
            r = r.unsqueeze(1)
        elif r.dim() == 1:
            r = r.unsqueeze(0).unsqueeze(1)
        if artg_hat.dim() == 1:
            artg_hat = artg_hat.unsqueeze(0).unsqueeze(0)
        elif artg_hat.dim() == 2:
            artg_hat = artg_hat.unsqueeze(1)
        if prob.dim() == 1:
            prob = prob.unsqueeze(-1).unsqueeze(-1)
        elif prob.dim() == 2:
            prob = prob.unsqueeze(-1)

        self.s = T.cat([self.s, s], dim=1)

        self.a = T.cat([self.a, a], dim=1)

        self.r = T.cat([self.r, r], dim=1)

        self.artg_hat = T.cat([self.artg_hat, artg_hat], dim=1)

        self.prob = T.cat([self.prob, prob], dim=1)

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

    def detach(self, i0, i1):
        self.s = T.cat(
            [
                self.s[:, :i0, :],
                self.s[:, i0:i1, :].detach(),
                self.s[:, i1:, :],
            ]
        )
        self.a = T.cat(
            [
                self.a[:, :i0, :],
                self.a[:, i0:i1, :].detach(),
                self.a[:, i1:, :],
            ]
        )
        self.r = T.cat(
            [
                self.r[:, :i0, :],
                self.r[:, i0:i1, :].detach(),
                self.r[:, i1:, :],
            ]
        )
        self.artg_hat = T.cat(
            [
                self.artg_hat[:, :i0, :],
                self.artg_hat[:, i0:i1, :].detach(),
                self.artg_hat[:, i1:, :],
            ]
        )


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
