import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import transformers
import torch_optimizer


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim=768, act_dim=3, n_layers=3, n_heads = 4, n_positions=8192, image_dim=[3, 96, 96], ppo_clip_value=0.2, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.state_dim = state_dim
        self.act_dim = act_dim

        # crashes if length is greater than n_positions
        # config = transformers.BartConfig(d_model=state_dim + act_dim + 1, d_embedding=state_dim + act_dim + 1, n_layer=n_layers, n_head=n_heads)
        # by default its in `block_sparse` mode with num_random_blocks=3, block_size=64
        config = transformers.DecisionTransformerConfig(state_dim=state_dim, act_dim=2 * act_dim, n_positions=n_positions, n_layer=n_layers, n_head=n_heads)
        self.transformer = transformers.DecisionTransformerModel(config).to(dtype=dtype, device=device)

        model_ckpt = "microsoft/beit-base-patch16-224-pt22k-ft22k"
        self.image_dim=image_dim
        self.image_processor = transformers.AutoImageProcessor.from_pretrained(model_ckpt, do_resize=True, device=device)
        self.vit = transformers.BeitModel.from_pretrained(model_ckpt).to(device=device)

        self.l2_loss = nn.MSELoss()

    def forward(self, *args, **kwargs):
        kwargs["actions"] = torch.cat([kwargs["actions"], torch.zeros(kwargs["actions"].shape, device=self.device)], dim=2)

        x = self.transformer(*args, **kwargs)
        return x

    def action_dist(self, action_preds):
        mid = int(action_preds.shape[-1] / 2)
        return Normal(loc=action_preds[:, :, :mid], scale=torch.maximum(action_preds[:, :, mid:], torch.fill(action_preds[:, :, mid:], 0.01)))

    def prob(self, dist, actions):
        return dist.log_prob(actions)

    def proc_state(self, o):
        with torch.inference_mode():
            o = torch.from_numpy(o).to(dtype=self.dtype, device=self.device).reshape(self.image_dim)
            o = self.image_processor(o, return_tensors="pt").pixel_values.to(device=self.device)
            e = self.vit(o).to_tuple()
            e = e[1].unsqueeze(0)

        return e.clone()

    def loss(self, actions, log_probs, rtgs):
        total_return = rtgs.max() # rtgs - rtgs.mean()
        return -(probs * total_return).sum() + actions.abs().sum()

    def train_iter(self, states, acts, log_probs, old_log_probs, gaes):
        loss = self.loss(acts, probs, hist.rtgs)

        loss.backward()
        self.optim.step()

        return loss
