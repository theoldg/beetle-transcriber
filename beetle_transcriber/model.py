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
        skip_input: Tensor | None = None,
    ) -> Tensor:
        """Input shape: (batch, channels, time)."""
        if skip_input is not None:
            input = torch.concat([input, skip_input], dim=1)
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

    def forward(
        self,
        x: Tensor,
        skip_input: Tensor | None = None,
    ) -> Tensor:
        x = torch.repeat_interleave(x, self.factor, dim=2)
        x = self.conv_layer(x, skip_input)
        return x


class UNetV1(nn.Module):
    def __init__(
        self,
        num_notes: int,
    ):
        super().__init__()
        self.num_notes = num_notes

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
                        out_channels=128,
                        kernel=1,
                        stride=1,
                    ),
                ),
            ]
        )

        self.last_layer = ConvLayer(
            ConvLayerConfig(
                input_channels=128,
                out_channels=self.num_notes * midi.NUM_CHANNELS,
                expanded_channels=256,
                kernel=1,
                stride=1,
            )
        )

    def forward(self, spectrograms: Tensor) -> Tensor:
        x = spectrograms
        skip_inputs = []
        for layer in self.down_layers:
            skip_inputs.append(x)
            x = layer(x)

        for i, layer in enumerate(self.up_layers):
            skip_input = skip_inputs[-i - 1]
            x = layer(x, skip_input)

        x = self.last_layer(x)

        x = torch.transpose(x, 1, 2)
        batch, time, channels_notes = x.shape
        x = x.reshape(batch, time, -1, midi.NUM_CHANNELS)

        return x
