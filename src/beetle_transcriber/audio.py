import dataclasses
from pathlib import Path

import soundfile as sf
import torch
import math
from torch import nn
from torch import Tensor
import torchaudio.transforms as T
import librosa


@dataclasses.dataclass
class SpectrogramConfig:
    sample_rate: int = 44_100
    hop_length: int = 2_048

    # C0, lowest note on extended pianos.
    f_min: float = 27.5
    n_bins: int = 116

    @property
    def n_bins(self):
        return math.ceil(12 * math.log2(self.f_max / self.f_min))


def load_audio_segment(
    file_path: Path,
    start_sec: float,
    duration_sec: float,
    samplerate: int,
) -> torch.Tensor:
    """Seeks and loads a segment of a WAV file directly from disk."""
    info = sf.info(file_path)
    start_frame = int(start_sec * info.samplerate)
    num_frames = int(duration_sec * info.samplerate)
    data, _ = sf.read(
        file_path,
        start=start_frame,
        frames=num_frames,
        dtype="float32",
        always_2d=True,
    )
    data = data.mean(-1)
    data = torch.from_numpy(data)
    data = T.Resample(orig_freq=info.samplerate, new_freq=samplerate)(data)
    return data


class AudioPreprocessor(nn.Module):
    def __init__(self, config: SpectrogramConfig):
        super().__init__()
        self.config = config

    def forward(self, waveform: Tensor) -> Tensor:
        spectrogram = librosa.cqt(
            y=waveform.numpy(),
            sr=self.config.sample_rate,
            hop_length=self.config.hop_length,
            fmin=self.config.f_min,
            n_bins=self.config.n_bins,
        )
        spectrogram = spectrogram.mean(0)
        spectrogram -= spectrogram.mean()
        spectrogram /= spectrogram.std()
        return torch.Tensor(spectrogram)
