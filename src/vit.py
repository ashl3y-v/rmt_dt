import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import transformers


class ViT(nn.Module):
    def __init__(self, d_img=[3, 96, 96], n_env=1, dtype=T.bfloat16, device="cuda"):
        super().__init__()
        self.dtype = dtype
        self.device = device

        model_ckpt = "microsoft/beit-base-patch16-224-pt22k-ft22k"
        self.d_img = d_img
        self.n_env = n_env
        self.image_processor = transformers.AutoImageProcessor.from_pretrained(
            model_ckpt, do_resize=True, device=device
        )
        self.vit = transformers.BeitModel.from_pretrained(model_ckpt)

        self.to(dtype=dtype, device=device)

    def forward(self, o):
        self.eval()
        with T.inference_mode():
            o = (
                T.from_numpy(o)
                .to(dtype=T.float32, device=self.device)
                .reshape(self.n_env, *self.d_img)
            )
            o = self.image_processor.preprocess(o, return_tensors="pt").pixel_values.to(
                dtype=self.dtype, device=self.device
            )
            e = self.vit(o).to_tuple()[1]

        # needed
        return e.detach().clone().unsqueeze(1)


if __name__ == "__main__":
    vit = ViT()
    print(vit.forward)
