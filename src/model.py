import torch

from .unet import DiffusionUNet
from .utils import beta_scheduler_factory


class DiffusionModel(torch.nn.Module):
    def __init__(self, input_channels, scheduler="linear", num_timesteps=1000):
        super(DiffusionModel, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_ch = input_channels

        # Define the beta scheduler
        self.betas = beta_scheduler_factory(scheduler, num_timesteps).to(self.device)

        # Define the alphas for the forward pass
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_oneminus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.unet = DiffusionUNet(self.in_ch)

    def forward_diffusion(self, x, t, noise):
        # Involves destroying the input sample by adding noise
        # We use the reparametrization trick to be able to directly compute the destroyed image at time t
        # Lists of alphas and betas are precomputed for each timestep
        return x * self.sqrt_alphas_cumprod[t] + noise * self.sqrt_oneminus_alphas_cumprod[t]

    def reverse_diffusion(self, x, t):
        # Models the reverse so given a corrupted sample, it tries to recover the original sample
        return self.unet(x, t)

    def forward(self, x, t):
        # Forward pass
        noise = torch.randn_like(x)
        diffused_x = self.forward_diffusion(x, t, noise)
        return self.reverse_diffusion(diffused_x, t), noise

    def sample(self, num_steps):
        # Sample from the model
        x = torch.randn(1, self.in_ch, 256, 256).to(self.device)
        for t in range(num_steps, -1, -1):
            z = torch.randn_like(x) if t > 1 else 0
            x = 1 / self.sqrt_alphas_cumprod[num_steps] * (x - self.unet(x, t) * (1 - self.alphas_cumprod[t]) / self.sqrt_oneminus_alphas_cumprod[t]) + z * self.betas[t]
        return x