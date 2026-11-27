import torch
import torch.nn as nn


class SimpleCNNAutoencoder(nn.Module):
    """
    Baseline Model: Simple CNN Autoencoder without skip connections or pretrained weights.

    Input:  L channel  (B, 1, 256, 256) in [-1, 1]
    Output: ab channel (B, 2, 256, 256) in [-1, 1] via Tanh
    """

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels

        self.encoder = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(1, c, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 -> 64
            nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 -> 32
            nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 -> 16
            nn.Conv2d(c * 4, c * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 -> 8
            nn.Conv2d(c * 8, c * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(
            # 8 -> 16
            nn.ConvTranspose2d(c * 8, c * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(inplace=True),
            # 16 -> 32
            nn.ConvTranspose2d(c * 8, c * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            # 32 -> 64
            nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            # 64 -> 128
            nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            # 128 -> 256
            nn.ConvTranspose2d(c, 2, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)
