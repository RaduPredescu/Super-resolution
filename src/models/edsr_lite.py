import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual block used in EDSR-style networks.

    EDSR removes BatchNorm layers compared to classical ResNet blocks,
    because BatchNorm can limit the range flexibility needed for image restoration.
    """

    def __init__(self, num_features: int = 64, residual_scale: float = 0.1) -> None:
        super().__init__()

        self.residual_scale = residual_scale

        self.block = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.block(x)
        return x + residual * self.residual_scale


class Upsampler(nn.Module):
    """
    PixelShuffle upsampler for scale x2, x3, or x4.

    For x4, it applies two x2 PixelShuffle stages.
    """

    def __init__(self, scale: int, num_features: int) -> None:
        super().__init__()

        layers = []

        if scale in (2, 4):
            for _ in range(scale // 2):
                layers.append(nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1))
                layers.append(nn.PixelShuffle(2))
                layers.append(nn.ReLU(inplace=True))
        elif scale == 3:
            layers.append(nn.Conv2d(num_features, num_features * 9, kernel_size=3, padding=1))
            layers.append(nn.PixelShuffle(3))
            layers.append(nn.ReLU(inplace=True))
        else:
            raise ValueError("Supported scales are 2, 3, and 4.")

        self.upsampler = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsampler(x)


class EDSRLite(nn.Module):
    """
    Lightweight EDSR-style model for single-image super-resolution.

    Unlike SRCNN, this model receives the true low-resolution image and upsamples
    internally using PixelShuffle.

    Input:  B x 3 x h x w
    Output: B x 3 x H x W, where H = h * scale and W = w * scale
    """

    def __init__(
        self,
        scale: int = 4,
        num_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 8,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()

        self.scale = scale

        self.head = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)

        self.body = nn.Sequential(
            *[
                ResidualBlock(
                    num_features=num_features,
                    residual_scale=residual_scale,
                )
                for _ in range(num_blocks)
            ]
        )

        self.body_conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.upsampler = Upsampler(scale=scale, num_features=num_features)
        self.tail = nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        residual = self.body(x)
        residual = self.body_conv(residual)
        x = x + residual
        x = self.upsampler(x)
        x = self.tail(x)
        return x
