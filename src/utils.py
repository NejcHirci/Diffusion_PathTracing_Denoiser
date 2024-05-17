import torch


def beta_scheduler_factory(beta_type, num_timesteps, **kwargs):
    """Returns a beta scheduler for the diffusion model."""
    if beta_type == "linear":
        return linear_beta_scheduler(num_timesteps, **kwargs)
    elif beta_type == "quadratic":
        return quadratic_beta_scheduler(num_timesteps, **kwargs)
    elif beta_type == "cosine":
        return cosine_beta_scheduler(num_timesteps, **kwargs)
    elif beta_type == "sigmoid":
        return sigmoid_beta_scheduler(num_timesteps, **kwargs)
    else:
        raise ValueError(f"Invalid beta scheduler type: {beta_type}")


def linear_beta_scheduler(num_timesteps, beta_start=1e-4, beta_end=0.02):
    """Returns a linear beta scheduler for the diffusion model."""
    return torch.linspace(beta_start, beta_end, num_timesteps)


def quadratic_beta_scheduler(num_timesteps, beta_start=1e-4, beta_end=0.02):
    """Returns a quadratic beta scheduler for the diffusion model."""
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2


def cosine_beta_scheduler(num_timesteps, s=0.008):
    """Returns a cosine beta scheduler for the diffusion model."""
    def f(t):
        return torch.cos((t / num_timesteps + s) / (1 + s) * 0.5 * torch.pi) ** 2
    x = torch.linspace(0, num_timesteps, num_timesteps + 1)
    alphas_cumprod = f(x) / f(torch.tensor([0.0]))
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(betas, 1e-4, 0.999)


def sigmoid_beta_scheduler(num_timesteps, beta_start=1e-4, beta_end=0.02):
    """Returns a sigmoid beta scheduler for the diffusion model."""
    betas = torch.linspace(-6, 6, num_timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def positional_embedding(x, dim, theta):
    pass
