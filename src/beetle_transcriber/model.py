from dataclasses import dataclass
import math

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
    normalize: bool = True


class ConvLayer(nn.Module):
    CONV: nn.Module
    NORM: nn.Module

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
                norm_layer=self.NORM,
                conv_layer=self.CONV,
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
                norm_layer=self.NORM,
                conv_layer=self.CONV,
            )
        )
        # Project
        layers.append(
            ConvNormActivation(
                config.expanded_channels,
                config.out_channels,
                kernel_size=1,
                norm_layer=self.NORM if self.config.normalize else None,
                activation_layer=None,
                conv_layer=self.CONV,
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


class ConvLayer1d(ConvLayer):
    CONV = nn.Conv1d
    NORM = nn.BatchNorm1d


class ConvLayer2d(ConvLayer):
    CONV = nn.Conv2d
    NORM = nn.BatchNorm2d


class UpLayer(nn.Module):
    def __init__(
        self,
        upsample_factor: int,
        conv_layer: ConvLayer,
    ):
        super().__init__()
        self.factor = upsample_factor
        self.conv_layer = conv_layer

    def forward(
        self,
        x: Tensor,
        skip_input: Tensor | None = None,
    ) -> Tensor:
        x = self.conv_layer(x, skip_input)
        x = torch.repeat_interleave(x, self.factor, dim=2)
        if x.ndim == 4:
            x = torch.repeat_interleave(x, self.factor, dim=3)
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
                ConvLayer1d(
                    ConvLayerConfig(
                        input_channels=88,
                        expanded_channels=256,
                        out_channels=128,
                        kernel=5,
                        stride=2,
                    )
                ),
                ConvLayer1d(
                    ConvLayerConfig(
                        input_channels=128,
                        expanded_channels=256,
                        out_channels=256,
                        kernel=5,
                        stride=2,
                    )
                ),
                ConvLayer1d(
                    ConvLayerConfig(
                        input_channels=256,
                        expanded_channels=256,
                        out_channels=256,
                        kernel=3,
                        stride=2,
                    )
                ),
                ConvLayer1d(
                    ConvLayerConfig(
                        input_channels=256,
                        expanded_channels=512,
                        out_channels=512,
                        kernel=3,
                        stride=2,
                    )
                ),
            ]
        )

        self.middle_layer = ConvLayer1d(
            ConvLayerConfig(
                input_channels=512,
                expanded_channels=512,
                out_channels=512,
                kernel=3,
                stride=1,
            )
        )

        self.up_layers = nn.ModuleList(
            [
                UpLayer(
                    upsample_factor=2,
                    conv_layer=ConvLayer1d(
                        ConvLayerConfig(
                            input_channels=1024,
                            expanded_channels=1024,
                            out_channels=512,
                            kernel=5,
                            stride=1,
                        )
                    ),
                ),
                UpLayer(
                    upsample_factor=2,
                    conv_layer=ConvLayer1d(
                        ConvLayerConfig(
                            input_channels=768,
                            expanded_channels=768,
                            out_channels=512,
                            kernel=5,
                            stride=1,
                        )
                    ),
                ),
                UpLayer(
                    upsample_factor=2,
                    conv_layer=ConvLayer1d(
                        ConvLayerConfig(
                            input_channels=768,
                            expanded_channels=512,
                            out_channels=256,
                            kernel=5,
                            stride=1,
                        )
                    ),
                ),
                UpLayer(
                    upsample_factor=2,
                    conv_layer=ConvLayer1d(
                        ConvLayerConfig(
                            input_channels=384,
                            expanded_channels=256,
                            out_channels=128,
                            kernel=5,
                            stride=1,
                        )
                    ),
                ),
            ]
        )

        self.last_up_layer = ConvLayer1d(
            ConvLayerConfig(
                input_channels=216,
                expanded_channels=512,
                out_channels=512,
                kernel=5,
                stride=1,
            )
        )

        self.final_layers = nn.ModuleList(
            [
                ConvLayer1d(
                    ConvLayerConfig(
                        input_channels=512,
                        expanded_channels=1024,
                        out_channels=self.num_notes * midi.NUM_CHANNELS,
                        kernel=5,
                        stride=1,
                        normalize=False,
                    )
                ),
            ]
        )

    def forward(self, spectrograms: Tensor) -> Tensor:
        divisibility_contraint = 2 ** len(self.up_layers)
        assert spectrograms.shape[-1] % divisibility_contraint == 0, (
            "For this number of layers, the time axis "
            f"must be divisible by {divisibility_contraint}"
        )

        x = spectrograms
        skip_inputs = []
        for layer in self.down_layers:
            x = layer(x)
            skip_inputs.append(x)

        x = self.middle_layer(x)

        for i, layer in enumerate(self.up_layers):
            skip_input = skip_inputs[-i - 1]
            x = layer(x, skip_input)

        x = self.last_up_layer(x, spectrograms)

        for layer in self.final_layers:
            x = layer(x)

        x = torch.transpose(x, 1, 2)
        batch, time, channels_notes = x.shape
        x = x.reshape(batch, time, -1, midi.NUM_CHANNELS)

        return x


class HarmonicLowering(nn.Module):
    """
    Find approximately integer multiples of the fundamental
    among the log-spaced frequency bins.

    Yay, music theory!
    """

    HARMONICS = {
        0: 0,
        1: 12,  # 2 ^ (12 / 12) = 2
        2: 19,  # 2 ^ (19 / 12) ~= 3
        3: 24,  # 2 ^ (24 / 12) = 4
        4: 28,  # 2 ^ (28 / 12) ~= 5
        5: 31,  # 2 ^ (31 / 12) ~= 6
    }

    def __init__(
        self,
        included_harmonics: list[int] = [0, 1, 2, 3, 4, 5],
    ):
        super().__init__()
        self.included_harmonics = included_harmonics
        self.offsets = [self.HARMONICS[i] for i in included_harmonics]

    def forward(self, spectrogram: Tensor) -> Tensor:
        """
        In: (batch, freq, time). Out: (batch, harmonics, freq, time).
        Only makes sense if frequency bins are log-spaced by 2^(1/12) (a musical semitone each).
        """
        batch_d, freq_d, time_d = spectrogram.shape
        output = torch.zeros(
            *(batch_d, len(self.offsets), freq_d, time_d),
            dtype=spectrogram.dtype,
            device=spectrogram.device,
        )
        for harmonic_i, offset in enumerate(self.offsets):
            output[:, harmonic_i, : freq_d - offset] = spectrogram[:, offset:]
        return output


class UNetV2(nn.Module):
    def __init__(self, num_notes: int = 88):
        super().__init__()
        self.num_notes = num_notes

        self.harmonic_lowering = HarmonicLowering()

        self.down_layers = nn.ModuleList(
            [
                ConvLayer2d(
                    ConvLayerConfig(
                        input_channels=6,  # Lowered harmonics.
                        expanded_channels=256,
                        out_channels=128,
                        kernel=5,
                        stride=2,
                    )
                ),
                ConvLayer2d(
                    ConvLayerConfig(
                        input_channels=128,
                        expanded_channels=256,
                        out_channels=256,
                        kernel=5,
                        stride=2,
                    )
                ),
                ConvLayer2d(
                    ConvLayerConfig(
                        input_channels=256,
                        expanded_channels=256,
                        out_channels=256,
                        kernel=3,
                        stride=2,
                    )
                ),
                ConvLayer2d(
                    ConvLayerConfig(
                        input_channels=256,
                        expanded_channels=512,
                        out_channels=512,
                        kernel=3,
                        stride=2,
                    )
                ),
            ]
        )

        self.middle_layer = ConvLayer2d(
            ConvLayerConfig(
                input_channels=512,
                expanded_channels=512,
                out_channels=512,
                kernel=3,
                stride=1,
            )
        )

        self.up_layers = nn.ModuleList(
            [
                UpLayer(
                    upsample_factor=2,
                    conv_layer=ConvLayer2d(
                        ConvLayerConfig(
                            input_channels=1024,
                            expanded_channels=1024,
                            out_channels=512,
                            kernel=5,
                            stride=1,
                        )
                    ),
                ),
                UpLayer(
                    upsample_factor=2,
                    conv_layer=ConvLayer2d(
                        ConvLayerConfig(
                            input_channels=768,
                            expanded_channels=768,
                            out_channels=512,
                            kernel=5,
                            stride=1,
                        )
                    ),
                ),
                UpLayer(
                    upsample_factor=2,
                    conv_layer=ConvLayer2d(
                        ConvLayerConfig(
                            input_channels=768,
                            expanded_channels=512,
                            out_channels=256,
                            kernel=5,
                            stride=1,
                        )
                    ),
                ),
                UpLayer(
                    upsample_factor=2,
                    conv_layer=ConvLayer2d(
                        ConvLayerConfig(
                            input_channels=384,
                            expanded_channels=256,
                            out_channels=128,
                            kernel=5,
                            stride=1,
                        )
                    ),
                ),
            ]
        )

        self.last_up_layer = ConvLayer2d(
            ConvLayerConfig(
                input_channels=134,
                expanded_channels=129,
                out_channels=64,
                kernel=5,
                stride=1,
            )
        )

        self.final_layers = nn.ModuleList(
            [
                ConvLayer2d(
                    ConvLayerConfig(
                        input_channels=64,
                        expanded_channels=128,
                        out_channels=midi.NUM_CHANNELS,
                        kernel=5,
                        stride=1,
                        normalize=False,
                    )
                ),
            ]
        )

    def forward(self, spectrograms: Tensor) -> Tensor:
        batch_d, freq_d, time_d = spectrograms.shape

        divisibility_contraint = 2 ** len(self.up_layers)
        assert time_d % divisibility_contraint == 0, (
            "For this number of layers, the time axis "
            f"must be divisible by {divisibility_contraint}"
        )

        freq_d_nearest_pow2 = 2 ** (math.ceil(math.log2(freq_d)))
        spectrograms = torch.concat(
            (
                spectrograms,
                torch.zeros(
                    batch_d,
                    freq_d_nearest_pow2 - freq_d,
                    time_d,
                    device=spectrograms.device,
                ),
            ),
            dim=1,
        )

        spectrograms = self.harmonic_lowering(spectrograms)

        x = spectrograms
        skip_inputs = []
        for i, layer in enumerate(self.down_layers):
            x = layer(x)
            skip_inputs.append(x)

        x = self.middle_layer(x)

        for i, layer in enumerate(self.up_layers):
            skip_input = skip_inputs[-i - 1]
            x = layer(x, skip_input)

        x = self.last_up_layer(x, spectrograms)
        x = x[:, :, : self.num_notes]

        for layer in self.final_layers:
            x = layer(x)

        x = torch.transpose(x, 1, 3)

        return x
