"""
Selected Method 3: Palette-style conditional diffusion model for colorization.

Idea
----
Instead of predicting the colors in one shot, a diffusion model learns to *remove noise*.

Training (forward process):
    Take the true colors `ab`, pick a random timestep t in [0, T), and mix in Gaussian noise:
        x_t = sqrt(alpha_bar_t) * ab + sqrt(1 - alpha_bar_t) * noise
    `alpha_bar_t` goes from ~1 (almost clean) at t = 0 to ~0 (pure noise) at t = T - 1.
    A U-Net sees [L, x_t] and the timestep t and predicts `noise`. The loss is the L2 (MSE)
    distance between the predicted and the real noise.

Inference (reverse process with DDIM):
    Start from pure noise and repeatedly ask the U-Net for the noise, which gives an estimate of
    the clean colors x0. DDIM (Song et al., 2021) jumps directly between a few chosen timesteps,
    so 50-250 steps are enough instead of all 1,000.

The original Palette (Saharia et al., 2022) works in RGB. Here the model works in CIELAB, like the
other models: the input lightness L is kept exactly and only the two color channels are generated.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks of the noise-prediction U-Net
# ---------------------------------------------------------------------------

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal encoding of the timestep (same idea as Transformer positional encodings)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, channels), channels)


class ResBlock(nn.Module):
    """
    Residual block conditioned on the timestep.

    The timestep embedding produces a per-channel scale and shift applied after the second
    normalization, so the block can behave differently for very noisy and almost clean inputs.
    The last convolution starts at zero, so every block starts as an identity mapping.
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch * 2)
        self.norm2 = group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_proj(F.silu(t_emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """
    Multi-head self-attention over all pixels of a feature map.

    Every position can look at every other position, which helps the model keep colors consistent
    across a whole object (e.g. one sky color). Used only at low resolution because the cost grows
    with (H * W)^2.
    """

    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.norm = group_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # (B, 3C, H, W) -> three tensors of shape (B, heads, C / heads, H * W)
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.num_heads, c // self.num_heads, h * w).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    """Halve the resolution with a strided convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Double the resolution (nearest neighbour) followed by a convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class AttentionUNet(nn.Module):
    """
    Noise-prediction U-Net.

    Input:  [L, noisy ab]  (B, 3, H, W) and the timestep t (B,)
    Output: predicted noise on ab  (B, 2, H, W)

    With the default settings at 256x256 the encoder works at 256 -> 128 -> 64 -> 32 pixels with
    64 -> 128 -> 256 -> 256 channels. Self-attention runs at 32x32 and in the bottleneck.
    Every encoder output is saved and concatenated into the matching decoder block (skip connections).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (32,),
        num_heads: int = 4,
        dropout: float = 0.1,
        image_size: int = 256,
    ) -> None:
        super().__init__()
        time_dim = base_channels * 4
        self.base_channels = base_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder: `skip_channels` records the channel count of every output saved for the decoder.
        self.down = nn.ModuleList()
        skip_channels = [base_channels]
        ch = base_channels
        resolution = image_size
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResBlock(ch, out_ch, time_dim, dropout)])
                ch = out_ch
                if resolution in attention_resolutions:
                    block.append(SelfAttention(ch, num_heads))
                self.down.append(block)
                skip_channels.append(ch)
            if level != len(channel_mults) - 1:
                self.down.append(nn.ModuleList([Downsample(ch)]))
                skip_channels.append(ch)
                resolution //= 2

        self.mid = nn.ModuleList([
            ResBlock(ch, ch, time_dim, dropout),
            SelfAttention(ch, num_heads),
            ResBlock(ch, ch, time_dim, dropout),
        ])

        # Decoder: mirrors the encoder and consumes the saved skips in reverse order.
        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for i in range(num_res_blocks + 1):
                block = nn.ModuleList([ResBlock(ch + skip_channels.pop(), out_ch, time_dim, dropout)])
                ch = out_ch
                if resolution in attention_resolutions:
                    block.append(SelfAttention(ch, num_heads))
                if level != 0 and i == num_res_blocks:
                    block.append(Upsample(ch))
                    resolution *= 2
                self.up.append(block)

        self.out = nn.Sequential(group_norm(ch), nn.SiLU(), nn.Conv2d(ch, out_channels, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    @staticmethod
    def _run(block: nn.ModuleList, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        for layer in block:
            h = layer(h, t_emb) if isinstance(layer, ResBlock) else layer(h)
        return h

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(timestep_embedding(t, self.base_channels))
        h = self.in_conv(x)
        skips = [h]
        for block in self.down:
            h = self._run(block, h, t_emb)
            skips.append(h)
        h = self._run(self.mid, h, t_emb)
        for block in self.up:
            h = torch.cat([h, skips.pop()], dim=1)
            h = self._run(block, h, t_emb)
        return self.out(h)


# ---------------------------------------------------------------------------
# Diffusion wrapper: training loss and DDIM sampling
# ---------------------------------------------------------------------------

class PaletteColorizer(nn.Module):
    """
    Conditional diffusion colorizer.

    - `diffusion_loss(L, ab)`: training objective (MSE between real and predicted noise).
    - `forward(L)` / `sample(L)`: generate ab from Gaussian noise with DDIM, conditioned on L.
      Because `forward` returns ab like the regression models, the same evaluation code works for all.
    """

    is_diffusion = True

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        sampling_steps: int = 50,
        ddim_eta: float = 0.0,
        **unet_kwargs,
    ) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.sampling_steps = sampling_steps
        self.ddim_eta = ddim_eta
        self.denoiser = AttentionUNet(**unet_kwargs)

        # Linear noise schedule: beta_t is the amount of noise added at step t.
        # alpha_bar_t = prod(1 - beta_s for s <= t) is how much of the clean signal is left at step t.
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod.float(), persistent=False)

    def q_sample(self, ab: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward process: noisy colors at timestep t."""
        alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return alpha_bar.sqrt() * ab + (1.0 - alpha_bar).sqrt() * noise

    def predict_noise(self, L: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The U-Net is conditioned on L by concatenating it with the noisy colors."""
        return self.denoiser(torch.cat([L, x_t], dim=1), t)

    def diffusion_loss(
        self,
        L: torch.Tensor,
        ab: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Pick a random timestep per image, add noise, and score the noise prediction with MSE."""
        t = torch.randint(0, self.timesteps, (L.size(0),), device=L.device, generator=generator)
        noise = torch.randn(ab.shape, device=ab.device, dtype=ab.dtype, generator=generator)
        x_t = self.q_sample(ab, t, noise)
        pred_noise = self.predict_noise(L, x_t, t)
        return F.mse_loss(pred_noise.float(), noise.float())

    @torch.no_grad()
    def sample(
        self,
        L: torch.Tensor,
        steps: Optional[int] = None,
        eta: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        DDIM sampling of ab conditioned on L.

        steps: number of denoising steps (default: `self.sampling_steps`).
        eta:   0 gives deterministic DDIM; 1 adds fresh noise each step like the original DDPM.
        """
        steps = int(steps or self.sampling_steps)
        eta = self.ddim_eta if eta is None else float(eta)
        b, _, h, w = L.shape
        x = torch.randn((b, 2, h, w), device=L.device, dtype=L.dtype, generator=generator)

        # Evenly spaced timesteps from the noisiest (T - 1) down to 0.
        times = torch.linspace(self.timesteps - 1, 0, steps, device=L.device).round().long()

        for i, t in enumerate(times):
            alpha_bar = self.alphas_cumprod[t]
            is_last = i + 1 == steps
            alpha_bar_next = torch.ones_like(alpha_bar) if is_last else self.alphas_cumprod[times[i + 1]]

            # 1. Predict the noise and use it to estimate the clean colors x0 (clamped to the valid range).
            eps = self.predict_noise(L, x, t.repeat(b)).to(x.dtype)
            x0 = ((x - (1.0 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt()).clamp(-1.0, 1.0)
            # 2. Recompute the noise that is consistent with the clamped x0.
            eps = (x - alpha_bar.sqrt() * x0) / (1.0 - alpha_bar).sqrt()

            # 3. Move to the next (less noisy) timestep.
            sigma = eta * ((1.0 - alpha_bar_next) / (1.0 - alpha_bar) * (1.0 - alpha_bar / alpha_bar_next)).sqrt()
            x = alpha_bar_next.sqrt() * x0 + (1.0 - alpha_bar_next - sigma ** 2).clamp(min=0.0).sqrt() * eps
            if eta > 0 and not is_last:
                x = x + sigma * torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)

        return x.clamp(-1.0, 1.0)

    def forward(self, L: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        return self.sample(L, generator=generator)
