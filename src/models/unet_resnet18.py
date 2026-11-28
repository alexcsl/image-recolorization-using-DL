"""
Selected Method 1: U-Net with an ImageNet-pretrained ResNet-18 encoder.

- The pretrained encoder already recognizes objects (sky, grass, faces), which helps pick plausible colors.
- Skip connections pass encoder features at every resolution straight to the decoder, so object edges stay sharp.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class DecoderBlock(nn.Module):
    """Bilinear upsampling + skip concatenation + double conv block."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetResNet18(nn.Module):
    """
    Selected Method 1: U-Net with ImageNet-Pretrained ResNet-18 Encoder.

    - Adapts the first conv layer from 3 RGB channels to 1 L channel by averaging
      the pretrained RGB filter weights so low-level edge detectors are retained.
    - Uses multi-scale skip connections to preserve object boundaries and fine details.

    Input:  L channel  (B, 1, 256, 256) in [-1, 1]
    Output: ab channel (B, 2, 256, 256) in [-1, 1] via Tanh
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        # Adapt conv1 for 1-channel L input
        old_conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(
            1,
            old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                new_conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))
        self.enc_conv1 = nn.Sequential(new_conv1, backbone.bn1, backbone.relu)  # 64, 128x128
        self.enc_pool = backbone.maxpool
        self.enc_layer1 = backbone.layer1  # 64,  64x64
        self.enc_layer2 = backbone.layer2  # 128, 32x32
        self.enc_layer3 = backbone.layer3  # 256, 16x16
        self.enc_layer4 = backbone.layer4  # 512, 8x8

        self.dec4 = DecoderBlock(512, 256, 256)  # 8 -> 16
        self.dec3 = DecoderBlock(256, 128, 128)  # 16 -> 32
        self.dec2 = DecoderBlock(128, 64, 64)    # 32 -> 64
        self.dec1 = DecoderBlock(64, 64, 64)     # 64 -> 128
        self.dec0 = DecoderBlock(64, 1, 32)      # 128 -> 256 (skip from input L)

        self.head = nn.Sequential(
            nn.Conv2d(32, 2, kernel_size=1),
            nn.Tanh(),
        )

    def set_encoder_trainable(self, trainable: bool) -> None:
        encoder_modules = [
            self.enc_conv1,
            self.enc_layer1,
            self.enc_layer2,
            self.enc_layer3,
            self.enc_layer4,
        ]
        for mod in encoder_modules:
            for p in mod.parameters():
                p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = x                          # (B, 1, 256, 256)
        s1 = self.enc_conv1(x)          # (B, 64, 128, 128)
        s2 = self.enc_layer1(self.enc_pool(s1))  # (B, 64, 64, 64)
        s3 = self.enc_layer2(s2)        # (B, 128, 32, 32)
        s4 = self.enc_layer3(s3)        # (B, 256, 16, 16)
        b = self.enc_layer4(s4)         # (B, 512, 8, 8)

        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        d0 = self.dec0(d1, s0)
        return self.head(d0)
