import math

import torch

from unet import DiffusionUNet
from utils import beta_scheduler_factory, denormalize


class DiffusionModel(torch.nn.Module):
    def __init__(self, resolution, channels, scheduler="linear", num_timesteps=1000):
        super(DiffusionModel, self).__init__()

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

        self.denoiser = DiffusionUNet(3, init_dim=64, dim_mults=(1, 2, 4, 8), theta=10000)

    def forward_diffusion(self, x, t, noise):
        # Involves destroying the input sample by adding noise
        # We use the reparametrization trick to be able to directly compute the destroyed image at time t
        B, _, _, _ = x.shape
        alpha_hat = self.alpha_bar[t].reshape(B, *((1,) * (len(x.shape) - 1)))
        noisy_sample = torch.sqrt(alpha_hat) * x + noise * torch.sqrt(1 - alpha_hat)
        return noisy_sample

    def reverse_diffusion(self, x, t):
        # Models the reverse so given a corrupted sample, it tries to recover the original sample
        return self.denoiser(x, t)

    def forward(self, x_zeros):
        B, _, _, _ = x_zeros.shape
        device = x_zeros.device

        # 1. Sample a random timestep
        t = torch.randint(0, self.num_timesteps, (B,), dtype=torch.long).to(device)

        # 2. Perform Forward Diffusion process
        noise = torch.randn_like(x_zeros).to(device)
        noisy_x = self.forward_diffusion(x_zeros, t, noise)

        # 3. Predict the noise given the perturbed images
        predicted_noise = self.denoiser(noisy_x, t)

        return noisy_x, noise, predicted_noise

    def sample(self, N):
        """Sample from the model for a given number of samples."""
        # Sample from the model
        device = next(self.parameters()).device
        x_t = torch.randn((N, self.num_channels, self.img_res, self.img_res)).to(device)

        for t in range(self.num_timesteps - 1, -1, -1):
            z = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)
            z = z.to(device)

            time_tensor = torch.full((N,), t, dtype=torch.long).to(device)

            pred_noise = self.denoiser(x_t, time_tensor)

            pre_scale = 1 / torch.sqrt(self.alphas[t])
            noise_scale = (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bar[t])
            post_sigma = torch.sqrt(self.betas[t]) * z

            x_t = pre_scale * (x_t - noise_scale * pred_noise) + post_sigma

        # Rescale the output to be in the range [0, 1]
        x_0 = denormalize(x_t.clamp(-1, 1))

        return x_0
