import math

import torch

from unet import DiffusionUNet, DiffusionUNetLite
from diffusers import UNet2DModel
from utils import beta_scheduler_factory, normalize, denormalize


class DiffusionModelForDenoising(torch.nn.Module):
    def __init__(self, resolution, channels, scheduler="linear", num_timesteps=1000, mode="lite", in_dim=32, dim_mults=(1, 2, 2, 4)):
        super(DiffusionModelForDenoising, self).__init__()

        self.img_res, _ = resolution
        self.num_channels = channels
        self.num_timesteps = num_timesteps
        self.scheduler = scheduler

        # Define the beta scheduler
        betas = beta_scheduler_factory(scheduler, num_timesteps)

        # Define the alphas for the forward pass
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        # Correctly register all the parameters
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

        if mode == "lite":
            self.denoiser = DiffusionUNetLite(3, init_dim=in_dim, dim_mults=dim_mults,
                                              has_attn=(False, False, True, True), theta=10000)
        else:
            self.denoiser = DiffusionUNet(3, init_dim=in_dim, dim_mults=dim_mults, theta=10000)


    def forward_diffusion(self, x_gt, x_real, t):
        B, _, _, _ = x_gt.shape
        factor = t / (self.num_timesteps - 1)
        noisy_sample = torch.einsum("b,bchw->bchw", 1 - factor, x_gt) + torch.einsum("b,bchw->bchw", factor, x_real)
        return noisy_sample

    def reverse_diffusion(self, x, t):
        # Models the reverse so given a corrupted sample, it tries to recover the original sample
        return self.denoiser(x, t)

    def forward(self, x_real, x_gt):
        B, _, _, _ = x_real.shape
        device = x_real.device

        # 1. Sample a random timestep
        t = torch.randint(0, (self.num_timesteps - 1), (B,)).to(device)

        # 2. Perform Forward Diffusion process
        noisy_x = self.forward_diffusion(x_gt, x_real, t)

        # 3. Predict the ground truth given the perturbed images
        predicted_gt = self.denoiser(noisy_x, t)

        return predicted_gt, x_gt

    def sample(self, x_real, denorm=True):
        """Sample from the model for a given number of samples."""
        # Sample from the model
        B = x_real.shape[0]
        device = next(self.parameters()).device
        x_t = x_real.clone().to(device)

        for t in range(self.num_timesteps-1, -1, -1):
            time_tensor = torch.full((B,), t, dtype=torch.long).to(device)
            factor_t0 = time_tensor / (self.num_timesteps-1)
            factor_t1 = (time_tensor - 1) / (self.num_timesteps-1)
            pred_x = self.denoiser(x_t, time_tensor)
            app_x_t0 = torch.einsum("b,bchw->bchw", 1 - factor_t0, pred_x) + torch.einsum("b,bchw->bchw", factor_t0, x_real)
            app_x_t1 = torch.einsum("b,bchw->bchw", 1 - factor_t1, pred_x) + torch.einsum("b,bchw->bchw", factor_t1, x_real)
            x_t = x_t - app_x_t0 + app_x_t1

        # Rescale the output to be in the range [0, 1]
        x_0 = x_t.clamp(-1, 1)
        if denorm:
            x_0 = denormalize(x_0)
        return x_0

