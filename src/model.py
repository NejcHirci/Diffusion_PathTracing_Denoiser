import torch

from unet import DiffusionUNet
from utils import beta_scheduler_factory, normalize, denormalize


class DiffusionModel(torch.nn.Module):
    def __init__(self, resolution, channels, scheduler="cosine", num_timesteps=1000):
        super(DiffusionModel, self).__init__()

        self.img_res = resolution
        self.num_channels = channels

        # Define the beta scheduler
        betas = beta_scheduler_factory(scheduler, num_timesteps)

        # Define the alphas for the forward pass
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Calculations for diffusion q(x_t | x_{t-1})
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_oneminus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # Correctly register all the parameters
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_oneminus_alphas_cumprod", sqrt_oneminus_alphas_cumprod)

        self.denoiser = DiffusionUNet(self.img_res, channels=channels)

    def forward_diffusion(self, x, t, noise):
        # Involves destroying the input sample by adding noise
        # We use the reparametrization trick to be able to directly compute the destroyed image at time t
        # Lists of alphas and betas are precomputed for each timestep
        B, _, _, _ = x.shape

        # Reshape the alpha and oneminus_alpha tensors
        sqrt_alphas = self.sqrt_alphas_cumprod[t].reshape(B, *((1,) * (len(x.shape) - 1)))
        sqrt_oneminus_alphas = self.sqrt_oneminus_alphas_cumprod[t].reshape(B, *((1,) * (len(x.shape) - 1)))

        noisy_sample = sqrt_alphas * x + noise * sqrt_oneminus_alphas
        return noisy_sample

    def reverse_diffusion(self, x, t):
        # Models the reverse so given a corrupted sample, it tries to recover the original sample
        return self.denoiser(x, t)

    def forward(self, x_zeros):
        B, _, _, _ = x_zeros.shape
        device = x_zeros.device

        # Normalize the input image
        x_zeros = normalize(x_zeros)

        # 1. Sample a random timestep
        t = torch.randint(0, 1000, (B,), dtype=torch.long).to(device)

        # 2. Perform Forward Diffusion process
        noise = torch.randn_like(x_zeros).to(device)
        perturbed_x = self.forward_diffusion(x_zeros, t, noise)

        # 3. Predict the noise given the perturbed images
        predicted_noise = self.reverse_diffusion(perturbed_x, t)

        return perturbed_x, noise, predicted_noise

    def sample(self, num_steps, N):
        # Sample from the model
        device = next(self.parameters()).device
        x_t = torch.randn((N, self.num_channels, self.img_res, self.img_res)).to(device)

        for t in range(num_steps - 1, -1, -1):
            z = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)
            z = z.to(device)

            time_tensor = torch.tensor([t] * N).to(device)
            predicted_noise = self.reverse_diffusion(x_t, time_tensor)

            alpha = self.alphas[time_tensor].reshape(N, *((1,) * (len(x_t.shape) - 1)))
            sqrt_alpha = self.sqrt_alphas_cumprod[time_tensor].reshape(N, *((1,) * (len(x_t.shape) - 1)))
            sqrt_oneminus_alpha = self.sqrt_oneminus_alphas_cumprod[time_tensor].reshape(N, *((1,) * (len(x_t.shape) - 1)))

            x_t = 1 / sqrt_alpha * (x_t - predicted_noise * (1-alpha) / sqrt_oneminus_alpha) + z * self.betas[t]
            x_t = x_t.clamp(-1., 1)

        # Rescale the output to be in the range [0, 1]
        x_0 = denormalize(x_t)

        return x_0
