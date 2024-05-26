import math

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


# For the PixelCNN++ implentation I use segments by
# https://github.com/pclucas14/pixel-cnn-pp by Lucas Caccia

class DiffusionUNet(nn.Module):
    def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=3, theta=10000):
        super(DiffusionUNet, self).__init__()

        self.channels = channels
        self.init_dim = init_dim if init_dim else dim

        dims = [self.init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.in_conv = nn.Conv2d(channels, self.init_dim, 3, padding=1)

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            TimeEmbedding(dim, theta),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.downs = nn.ModuleList([nn.ModuleList([
            ConvNextBlock(in_channels=dim_in, out_channels=dim_in, time_dim=time_dim),
            ConvNextBlock(in_channels=dim_in, out_channels=dim_in, time_dim=time_dim),
            Downsample(dim_in, dim_out) if i < len(in_out) - 1 else nn.Conv2d(dim_in, dim_out, 3, padding=1)])
            for i, (dim_in, dim_out) in enumerate(in_out)])

        mid_dim = dims[-1]
        self.mids = nn.ModuleList([
            ConvNextBlock(mid_dim, mid_dim, time_dim=time_dim),
            ConvNextBlock(mid_dim, mid_dim, time_dim=time_dim)
        ])
        self.ups = nn.ModuleList([nn.ModuleList([
            ConvNextBlock(in_channels=dim_out + dim_in, out_channels=dim_out, time_dim=time_dim),
            ConvNextBlock(in_channels=dim_out, out_channels=dim_out, time_dim=time_dim),
            Upsample(dim_out, dim_in) if i < len(in_out) - 1 else nn.Conv2d(dim_out, dim_in, 3, padding=1)
        ]) for i, (dim_in, dim_out) in enumerate(in_out[::-1])])

        self.out_dim = out_dim if out_dim else channels
        self.final_res = ConvNextBlock(dim * 2, dim, time_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time):
        b, _, h, w = x.shape
        x = self.in_conv(x)
        skip_x = x.clone()

        t = self.time_mlp(time)

        unet_stack = []
        for down1, down2, downsample in self.downs:
            x = down1(x, t)
            x = down2(x, t)
            unet_stack.append(x)
            x = downsample(x)

        x = self.mids[0](x, t)
        x = self.mids[1](x, t)

        for up1, up2, upsample in self.ups:
            x = up1(torch.cat((x, unet_stack.pop()), dim=1), t)
            x = up2(x, t)
            x = upsample(x)

        x = torch.cat((x, skip_x), dim=1)
        x = self.final_res(x, t)

        return self.final_conv(x)


class ConvNextBlock(nn.Module):
    """Implementation of ConvNext v2 Block"""
    def __init__(self, in_channels, out_channels, scale=2, time_dim=None, normalize=True):
        super(ConvNextBlock, self).__init__()

        self.mlp = nn.Sequential(
            nn.GELU(), nn.Linear(time_dim, in_channels)
        ) if time_dim else nn.Identity()

        # 7x7 Depthwise Convolution
        self.d_conv = nn.Conv2d(in_channels, in_channels, 7, padding=3, groups=in_channels)

        self.block = nn.Sequential(
            nn.GroupNorm(1, in_channels) if normalize else nn.Identity(),
            nn.Conv2d(in_channels, out_channels * scale, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, out_channels * scale),
            nn.Conv2d(out_channels * scale, out_channels, 3, padding=1)
        )

        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_embedding=None):
        h = self.d_conv(x)
        if time_embedding is not None:
            h += self.mlp(time_embedding).reshape(-1, x.shape[1], 1, 1)
        h = self.block(h)
        return h + self.skip_conv(x)


class TimeEmbedding(nn.Module):
    def __init__(self, dim, theta):
        super(TimeEmbedding, self).__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        out = time[:, None] * emb[None, :]
        out = torch.cat((out.sin(), out.cos()), dim=-1)
        return out


class Downsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super(Downsample, self).__init__()
        self.block = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
            nn.Conv2d(dim * 4, dim_out if dim_out else dim, 1),
        )

    def forward(self, x):
        return self.block(x)


class Upsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super(Upsample, self).__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(dim, dim_out if dim_out else dim, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)
