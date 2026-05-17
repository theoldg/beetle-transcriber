from dataclasses import dataclass
from typing import Any

import torch
from torch import nn, Tensor
from torchvision.ops.misc import ConvNormActivation

from beetle_transcriber import midi


@dataclass
class ConvLayerConfig:
    input_channels: int
    expanded_channels: int
    out_channels: int
    kernel: int
    stride: int


class ConvLayer(nn.Module):
    def __init__(self, config: ConvLayerConfig):
        super().__init__()
        if not (1 <= config.stride <= 2):
            raise ValueError("illegal stride value")

        self.config = config
        layers: list[nn.Module] = []

        # Expand
        layers.append(
            ConvNormActivation(
                config.input_channels,
                config.expanded_channels,
                kernel_size=1,
                norm_layer=nn.BatchNorm1d,
                conv_layer=nn.Conv1d,
            )
        )

        # Depthwise
        layers.append(
            ConvNormActivation(
                config.expanded_channels,
                config.expanded_channels,
                kernel_size=config.kernel,
                stride=config.stride,
                groups=config.expanded_channels,
                norm_layer=nn.BatchNorm1d,
                conv_layer=nn.Conv1d,
            )
        )
        # Project
        layers.append(
            ConvNormActivation(
                config.expanded_channels,
                config.out_channels,
                kernel_size=1,
                norm_layer=nn.BatchNorm1d,
                activation_layer=None,
                conv_layer=nn.Conv1d,
            )
        )

        self.block = nn.Sequential(*layers)

    def forward(
        self,
        input: Tensor,
        residual_input: Tensor | None = None,
    ) -> Tensor:
        """Input shape: (batch, channels, time)."""
        if residual_input is not None:
            input = torch.concat([input, residual_input], dim=1)
        result = self.block(input)
        return result


class UpLayer(nn.Module):
    def __init__(
        self,
        upsample_factor: int,
        conv_config: ConvLayerConfig,
    ):
        super().__init__()
        self.factor = upsample_factor
        self.conv_layer = ConvLayer(conv_config)

    def upsample(self, x: Tensor) -> Tensor:
        batch, channels, time = x.shape
        result = torch.zeros(
            (batch, channels, time * self.factor),
            dtype=x.dtype,
        )
        result[:, :, ::2] = x
        result[:, :, 1::2] = x
        return result

    def forward(
        self,
        x: Tensor,
        residual_input: Tensor | None = None,
    ) -> Tensor:
        x = self.upsample(x)
        x = self.conv_layer(x, residual_input)
        return x


class UNetV1(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.down_layers = nn.ModuleList(
            [
                ConvLayer(
                    ConvLayerConfig(
                        input_channels=128,
                        expanded_channels=128,
                        out_channels=64,
                        kernel=4,
                        stride=2,
                    )
                ),
                ConvLayer(
                    ConvLayerConfig(
                        input_channels=64,
                        expanded_channels=128,
                        out_channels=96,
                        kernel=3,
                        stride=2,
                    )
                ),
            ]
        )

        self.up_layers = nn.ModuleList(
            [
                UpLayer(
                    upsample_factor=2,
                    conv_config=ConvLayerConfig(
                        input_channels=160,
                        expanded_channels=96,
                        out_channels=64,
                        kernel=1,
                        stride=1,
                    ),
                ),
                UpLayer(
                    upsample_factor=2,
                    conv_config=ConvLayerConfig(
                        input_channels=192,
                        expanded_channels=96,
                        out_channels=64,
                        kernel=1,
                        stride=1,
                    ),
                ),
            ]
        )

    def forward(self, spectrogram: Tensor) -> Tensor:
        x = spectrogram
        residuals = []
        for layer in self.down_layers:
            residuals.append(x)
            x = layer(x)

        for i, layer in enumerate(self.up_layers):
            residual = residuals[-i - 1]
            x = layer(x, residual)

        return x
