import math
import random
import torch
from torch import nn
from numpy.random import choice


class Trainer:
    def __init__(self, model, lr=0.001, device="cpu"):
        self.device = device

        self.model = model

        self.optim = torch.optim.AdamW(model.parameters(), lr=lr)

    def train(self, epochs, training_epochs):
        # Sample epoch based on reward
        probs = []
        for e in epochs:
            probs.append(e.total_reward)

        for _ in training_epochs:
            e = choice(epochs, 1, p=probs)[0]
            e_len = len(e)
            beg = random.randint(0, math.floor(e_len / 2))
            end = random.randint(math.floor(e_len / 2), e_len)

            e = e[beg:end]

            for n, step in enumerate(e):
                state_preds, action_preds, return_preds = self.model(
                    states=e.states[:n],
                    actions=e.actions[:n],
                    rewards=e.rewards[:n],
                    returns_to_go=e.target_returns[:n],
                    timesteps=torch.tensor(beg + n, device=self.device),
                    attention_mask=torch.zeros(
                        1, n, device=self.device, dtype=torch.float32
                    ),
                    return_dict=False,
                )

                self.optim.zero_grad()
                loss = (
                    -torch.log(step.rtg * action_preds.sum())
                    + self.l2_loss(step.state, state_preds)
                    + self.l2_loss(step.rtg, return_preds)
                )

                loss.backward()
                self.optim.step()


EPOCHS = 1000
lr = 0.001

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("model.pt").to(device=device)
trainer = Trainer(model, lr=lr, device=device)

epochs = torch.load("blablabla.pt")
trainer.train(epochs, EPOCHS)
