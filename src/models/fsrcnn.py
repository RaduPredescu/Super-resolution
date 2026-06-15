import torch
import torch.nn as nn


class FSRCNN(nn.Module):
    """
    FSRCNN: Fast Super-Resolution Convolutional Neural Network.

    Paper:
    "Accelerating the Super-Resolution Convolutional Neural Network"

    Input:
        LR image tensor [B, 3, H, W]

    Output:
        SR image tensor [B, 3, H * scale, W * scale]
    """

    def __init__(
        self,
        scale: int = 4,
        num_channels: int = 3,
        d: int = 56,
        s: int = 12,
        m: int = 4,
    ) -> None:
        super().__init__()

        self.scale = scale

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=2),
            nn.PReLU(d),
        )

        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s),
        )

        mapping_layers = []

        for _ in range(m):
            mapping_layers.append(nn.Conv2d(s, s, kernel_size=3, padding=1))
            mapping_layers.append(nn.PReLU(s))

        self.mapping = nn.Sequential(*mapping_layers)

        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d),
        )

        self.deconvolution = nn.ConvTranspose2d(
            d,
            num_channels,
            kernel_size=9,
            stride=scale,
            padding=4,
            output_padding=scale - 1,
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=(2 / (module.out_channels * module.weight[0][0].numel())) ** 0.5,
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.normal_(self.deconvolution.weight, mean=0.0, std=0.001)

        if self.deconvolution.bias is not None:
            nn.init.zeros_(self.deconvolution.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extraction(x)
        x = self.shrinking(x)
        x = self.mapping(x)
        x = self.expanding(x)
        x = self.deconvolution(x)

        return x