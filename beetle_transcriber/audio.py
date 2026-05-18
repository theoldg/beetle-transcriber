import dataclasses

import soundfile as sf
import torch
import torchaudio.transforms as T


@dataclasses.dataclass
class MelSpectrogramConfig:
    # TODO
    # this is good enough for now but we should have
    # mel bins aligned with 12 TET notes 
    # (and actually logarithmic, not mel)

    sample_rate: int = 44_100
    n_fft: int = 2_048
    hop_length: int = 2_048
    n_mels: int = 120
    f_min: float = 10.0
    f_max: float = 16_000.0


def get_audio_duration_sec(file_path: str) -> float:
    """Returns the total duration of a WAV file in seconds without loading it."""
    info = sf.info(file_path)
    return info.duration


def load_audio_segment(
    file_path: str,
    start_sec: float,
    duration_sec: float,
    expected_samplerate: int,
) -> torch.Tensor:
    """Seeks and loads a segment of a WAV file directly from disk."""
    info = sf.info(file_path)
    assert expected_samplerate == info.samplerate
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
    waveform = torch.from_numpy(data).t()
    return waveform


def create_log_mel_spectrogram(
    waveform: torch.Tensor,
    config: MelSpectrogramConfig,
) -> torch.Tensor:
    """Generates a log mel spectrogram from a waveform using the provided config."""
    mel_transform = T.MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        f_min=config.f_min,
        f_max=config.f_max,
    )
    amplitude_to_db = T.AmplitudeToDB(stype="power")
    mel_spec = mel_transform(waveform)
    log_mel_spec = amplitude_to_db(mel_spec)
    return log_mel_spec
