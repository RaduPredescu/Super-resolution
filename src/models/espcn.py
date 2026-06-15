import torch
import torch.nn as nn


class ESPCN(nn.Module):
    """
    ESPCN: Efficient Sub-Pixel Convolutional Neural Network.

    Paper:
    Real-Time Single Image and Video Super-Resolution Using an Efficient
    Sub-Pixel Convolutional Neural Network

    Input:
        LR image tensor [B, 3, H, W]

    Output:
        SR image tensor [B, 3, H * scale, W * scale]
    """

    def __init__(
        self,
        scale: int = 4,
        num_channels: int = 3,
        num_features: int = 64,
    ) -> None:
        super().__init__()

        self.scale = scale
        self.num_channels = num_channels
        self.num_features = num_features

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(num_channels, num_features, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        self.sub_pixel = nn.Sequential(
            nn.Conv2d(
                num_features // 2,
                num_channels * (scale ** 2),
                kernel_size=3,
                padding=1,
            ),
            nn.PixelShuffle(scale),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extraction(x)
        x = self.sub_pixel(x)
        return x