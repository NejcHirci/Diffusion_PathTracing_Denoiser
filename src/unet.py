import math

import torch
import torch.nn as nn


# For the PixelCNN++ implentation I use segments by
# https://github.com/pclucas14/pixel-cnn-pp by Lucas Caccia

class DiffusionUNet(nn.Module):
    def __init__(self, im_channels, init_dim=None, dim_mults=(1, 2, 2, 4), has_attn=(False, False, True, True), theta=10000):
        super(DiffusionUNet, self).__init__()

        self.n_resolutions = len(dim_mults)
        self.channels = im_channels
        self.init_dim = init_dim if init_dim else im_channels

        dim = self.init_dim
        dims = [self.init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.in_conv = nn.Conv2d(im_channels, dim, 3, padding=1)

        time_dim = dim * 4
        self.time_mlp = TimeEmbedding(time_dim, theta)

        # First-half of the U-Net where channels are increasing and resolutions are decreasing
        self.downs = nn.ModuleList([nn.ModuleList([
            # First Block
            ResBlock(dim_in, dim_in, time_channels=time_dim),
            AttentionBlock(dim_in) if has_attn[i] else nn.Identity(),
            # Second Block
            ResBlock(dim_in, dim_out, time_channels=time_dim),
            AttentionBlock(dim_out) if has_attn[i] else nn.Identity(),
            # Downsample if not the last block
            Downsample(dim_out) if i < self.n_resolutions - 1 else nn.Identity()])
            for i, (dim_in, dim_out) in enumerate(in_out)])

        # Bottleneck in the middle
        mid_dim = dims[-1]
        self.mids_0 = ResBlock(mid_dim, mid_dim, time_channels=time_dim)
        self.mid_attn = AttentionBlock(mid_dim)
        self.mids_1 = ResBlock(mid_dim, mid_dim, time_channels=time_dim)

        # Second-half of the U-Net where channels are decreasing and resolutions are increasing
        self.ups = nn.ModuleList([nn.ModuleList([
            # First Block
            ResBlock(dim_out + dim_out, dim_out, time_channels=time_dim),
            AttentionBlock(dim_out) if has_attn[-i-1] else nn.Identity(),
            # Second Block
            ResBlock(dim_in + dim_out, dim_in, time_channels=time_dim),
            AttentionBlock(dim_in) if has_attn[-i-1] else nn.Identity(),
            # Final Block to Reduce Channel count
            ResBlock(dim_in + dim_in, dim_in, time_channels=time_dim),
            AttentionBlock(dim_in) if has_attn[-i-1] else nn.Identity(),
            # Upsample if not the last block
            Upsample(dim_in) if i < self.n_resolutions - 1 else nn.Identity()
        ]) for i, (dim_in, dim_out) in enumerate(in_out[::-1])])

        # Final normalization and convolution layers
        self.final_norm = nn.GroupNorm(8, dim)
        self.final_act = Swish()
        self.final_conv = nn.Conv2d(dim, im_channels, 3, padding=1)

    def forward(self, x, time):
        # Get Time Embeding to be added to all
        t = self.time_mlp(time)
        x = self.in_conv(x)

        unet_stack = [x]
        # First Half of the U-Net
        for res1, attn1, res2, attn2, downsample in self.downs:
            # Block 1
            x = res1(x, t)
            x = attn1(x)
            unet_stack.append(x)
            # Block 2
            x = res2(x, t)
            x = attn2(x)
            unet_stack.append(x)
            if isinstance(downsample, Downsample):
                x = downsample(x)
                unet_stack.append(x)

        # Midlle of the U-Net
        x = self.mids_0(x, t)
        x = self.mid_attn(x)
        x = self.mids_1(x, t)

        for res1, attn1, res2, attn2, final_res, final_attn, upsample in self.ups:
            # Block 1
            x = torch.cat((x, unet_stack.pop()), dim=1)
            x = res1(x, t)
            x = attn1(x)
            # Block 2
            x = torch.cat((x, unet_stack.pop()), dim=1)
            x = res2(x, t)
            x = attn2(x)
            # Final Block
            x = torch.cat((x, unet_stack.pop()), dim=1)
            x = final_res(x, t)
            x = final_attn(x)
            x = upsample(x)

        x = self.final_norm(x)
        x = self.final_act(x)
        return self.final_conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, n_groups=32, dropout=0.1):
        super(ResBlock, self).__init__()
        # First Convolutional Layer
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # Second Convolutional Layer
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()

    def forward(self, x, t):
        h = self.conv1(self.act1(self.norm1(x)))

        h = h + self.time_emb(self.time_act(t))[:, :, None, None]

        h = self.conv2(self.act2(self.norm2(h)))

        return h + self.shortcut(x)


class ConvNextBlock(nn.Module):
    """Implementation of ConvNext v2 Block"""
    def __init__(self, in_channels, out_channels, time_dim=None):
        super(ConvNextBlock, self).__init__()

        self.time_emb = nn.Linear(time_dim, in_channels) if time_dim is not None else nn.Identity()
        self.time_act = Swish() if time_dim is not None else nn.Identity()

        # 7x7 Depthwise Convolution
        self.d_conv = nn.Conv2d(in_channels, in_channels, 7, padding=3, groups=in_channels)
        self.norm = nn.GroupNorm(1, in_channels)
        self.pwconv1 = nn.Linear(in_channels, 4 * in_channels)  # Pointwise Convolution with linear layers
        self.act = Swish()
        self.pwconv2 = nn.Linear(4 * in_channels, in_channels if in_channels == out_channels else out_channels)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time=None):
        res = x
        x = self.d_conv(x)
        if time is not None:
            x += self.time_emb(self.time_act(time))[:, :, None, None]
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # B C H W -> B H W C
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # B H W C -> B C H W

        return x + self.skip_conv(res)


class TimeEmbedding(nn.Module):
    def __init__(self, dim, theta):
        super(TimeEmbedding, self).__init__()
        self.dim = dim
        self.theta = theta
        self.lin1 = nn.Linear(dim, 4 * dim)
        self.act = Swish()
        self.lin2 = nn.Linear(4 * dim, dim)

    def forward(self, time):
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=time.device) * -emb)
        out = time[:, None] * emb[None, :]
        out = torch.cat((out.sin(), out.cos()), dim=1)
        out = self.lin1(out)
        out = self.act(out)
        out = self.lin2(out)
        return out


class Downsample(nn.Module):
    def __init__(self, n_channels):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, n_channels):
        super(Upsample, self).__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, n_channels, n_heads=1, d_k=None, n_groups=32):
        super(AttentionBlock, self).__init__()

        if d_k is None:
            d_k = n_channels

        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.out_projection = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k ** -0.5

        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x):
        B, C, H, W = x.shape
        # B C H W -> B C H*W -> B H*W C
        x = x.view(B, C, -1).permute(0, 2, 1)
        qkv = self.projection(x).view(B, -1, self.n_heads, 3 * self.d_k)
        q, k, v = qkv.chunk(3, dim=-1)
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=2)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        res = res.view(B, -1, self.n_heads * self.d_k)
        res = self.out_projection(res)
        res += x
        return res.permute(0, 2, 1).view(B, C, H, W)


class AttentionGate(nn.Module):
    def __init__(self, gate_channel, res_channel, scale=True):
        super(AttentionGate, self).__init__()
        self.gate_conv = nn.Conv2d(gate_channel, gate_channel, 1, 1)  # Pointwise Convolution
        self.res_conv = nn.Conv2d(res_channel, gate_channel, 1, 1)  # Pointwise Convolution
        self.in_conv = nn.Conv2d(gate_channel, 1, 1, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        in_attention = self.relu(self.gate_conv(g) + self.res_conv(x))
        in_attention = self.in_conv(in_attention)
        in_attention = self.sigmoid(in_attention)
        return x * in_attention


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)