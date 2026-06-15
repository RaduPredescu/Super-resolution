import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, residual_scale: float = 0.1) -> None:
        super().__init__()

        self.residual_scale = residual_scale

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x) * self.residual_scale


class VSRCNN(nn.Module):
    """
    Simple multi-frame Video Super-Resolution CNN.

    Input:
        lr_frames: Tensor [B, T, C, H, W]

    Example:
        T = 7 frames
        C = 3 RGB channels
        input channels after concatenation = 7 * 3 = 21

    Output:
        sr: Tensor [B, 3, H * scale, W * scale]

    The model predicts the HR version of the center frame.
    """

    def __init__(
        self,
        num_frames: int = 7,
        scale: int = 4,
        num_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 8,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()

        if scale not in {2, 3, 4}:
            raise ValueError("scale must be 2, 3, or 4")

        self.num_frames = num_frames
        self.scale = scale
        self.num_channels = num_channels

        in_channels = num_frames * num_channels

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.body = nn.Sequential(
            *[
                ResidualBlock(
                    channels=num_features,
                    residual_scale=residual_scale,
                )
                for _ in range(num_blocks)
            ]
        )

        self.body_conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        if scale == 4:
            self.upsampler = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
            )
        elif scale == 3:
            self.upsampler = nn.Sequential(
                nn.Conv2d(num_features, num_features * 9, kernel_size=3, padding=1),
                nn.PixelShuffle(3),
                nn.ReLU(inplace=True),
            )
        else:
            self.upsampler = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
            )

        self.tail = nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)

    def forward(self, lr_frames: torch.Tensor) -> torch.Tensor:
        if lr_frames.ndim != 5:
            raise ValueError(
                f"Expected input shape [B, T, C, H, W], got {lr_frames.shape}"
            )

        b, t, c, h, w = lr_frames.shape

        if t != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames, got {t}")

        if c != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {c}")

        x = lr_frames.reshape(b, t * c, h, w)

        x = self.head(x)

        residual = x
        x = self.body(x)
        x = self.body_conv(x)
        x = x + residual

        x = self.upsampler(x)
        x = self.tail(x)

        return x