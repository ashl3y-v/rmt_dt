import torch
from torch import nn
from dt import DecisionTransformer

# name = "saved_buffer-09-10-31.pt"
# replay_buffer = torch.load(name)

# print(buffer.buffer)

device = torch.device("cuda")

length = 1000
state_dim = 1000
action_dim = 20

model = DecisionTransformer(state_dim=state_dim, act_dim=action_dim, device=device)

# epoch = replay_buffer.buffer[0]
# input = epoch[:-1]
# res = model(input)
# print(input.shape, res.shape, replay_buffer.buffer[0].shape)

# print(res - replay_buffer.buffer[0][-1])

states = torch.randn([1, length, state_dim], device=device)
actions = torch.randn([1, length, action_dim], device=device)
rewards = torch.randn([length, 1], device=device)
rtgs = torch.randn([1, length, 1], device=device)
timestep = torch.tensor(0, device=device)
attention_mask = torch.ones([1, length], device=device)

state_preds, action_preds, return_preds = model(
    states=states,
    actions=actions,
    rewards=rewards,
    returns_to_go=rtgs,
    timesteps=timestep,
    attention_mask=attention_mask,
    return_dict=False,
)

print(states.shape, actions.shape, rtgs.shape)
print(state_preds.shape, action_preds.shape, return_preds.shape)

l2_loss = nn.MSELoss()
optim = torch.optim.AdamW(model.parameters(), lr=0.001)

for i in range(1000):
    state_preds, action_preds, return_preds = model(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=rtgs,
        timesteps=timestep,
        attention_mask=attention_mask,
        return_dict=False,
    )

    optim.zero_grad()
    loss = l2_loss(states, state_preds)

    loss.backward()
    optim.step()

    print(loss)
