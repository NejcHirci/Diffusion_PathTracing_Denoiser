import torch
import torch.nn as nn


# For the PixelCNN++ implentation I use segments by
# https://github.com/pclucas14/pixel-cnn-pp by Lucas Caccia

class DiffusionUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiffusionUNet, self).__init__()

        self.encoder = nn

    # Based on PixelCNN++
    # Changes:
    # - Replace weight normalization with group normalization
    # - Introduce cross-attention blocks

    # 6 blocks of 5 ConvNextBlock layers


class ConvNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2, time_dim=None, padding=1, groups=1, normalize=True, dropout=0.1):
        super(ConvNextBlock, self).__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels)
        ) if time_dim else nn.Identity()

        # 7x7 Depthwise Convolution
        self.d_conv = nn.Conv2d(in_channels, in_channels, 7, padding=3, groups=in_channels)

        self.block = nn.Sequential(
            nn.GroupNorm(1, in_channels) if normalize else nn.Identity(),
            nn.Conv2d(in_channels, out_channels * scale, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, out_channels * scale),
            nn.Conv2d(out_channels * scale, out_channels, kernel_size=3, padding=1)
        )

        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_embedding=None):
        h = self.d_conv


class TimeEmbedding(nn.Module):
    def __init__(self, dim, theta, device="cuda"):
        super(TimeEmbedding, self).__init__()

        half_dim = dim // 2
        emb = torch.arrange(half_dim, device=device)
        self.emb = torch.exp(emb * -torch.log(theta) / (half_dim - 1))

    def forward(self, x):
        out = x[:, None] * self.emb[None, :]
        out = torch.cat((out.sin(), out.cos()), dim=-1)
        return out


class Upsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super(Upsample, self).__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(dim, dim_out or dim, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)