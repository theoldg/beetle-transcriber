from pathlib import Path
import os
from dataclasses import dataclass
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import pandas as pd

from beetle_transcriber.audio import (
    MelSpectrogramConfig,
    load_audio_segment,
    create_log_mel_spectrogram,
)
from beetle_transcriber.midi import MidiPreprocessingConfig, preprocess_midi

MAESTRO_PATH = Path(os.environ["MAESTRO_DATASET_PATH"])


@dataclass
class FileInfo:
    canonical_composer: str
    canonical_title: str
    split: str
    year: str
    midi_filename: str
    audio_filename: str
    duration: float


def load_metadata() -> pd.DataFrame:
    """The rows of the DataFrame have the structure of FileInfo."""
    return pd.read_csv(MAESTRO_PATH / "maestro-v3.0.0.csv")


@dataclass
class PreprocessedSample:
    duration: float

    spectrogram: Tensor
    spectrogram_config: MelSpectrogramConfig

    midi_data: Tensor
    midi_config: MidiPreprocessingConfig


def preprocess_random_segment(
    file_info: FileInfo,
    duration: float,
    spectrogram_config: MelSpectrogramConfig | None = None,
    midi_config: MidiPreprocessingConfig | None = None,
) -> PreprocessedSample:
    audio_path = MAESTRO_PATH / file_info.audio_filename
    assert audio_path.exists()

    midi_path = MAESTRO_PATH / file_info.midi_filename
    assert midi_path.exists()

    if spectrogram_config is None:
        spectrogram_config = MelSpectrogramConfig()

    if midi_config is None:
        # Match time resolution to spectrogram.
        time_resolution = spectrogram_config.hop_length / spectrogram_config.sample_rate
        midi_config = MidiPreprocessingConfig(time_resolution)  # Default min/max note.

    start_point = random.random() * (file_info.duration - duration)
    waveform = load_audio_segment(
        file_path=audio_path,
        start_sec=start_point,
        duration_sec=duration,
        samplerate=spectrogram_config.sample_rate,
    )
    spectrogram = create_log_mel_spectrogram(waveform, spectrogram_config)

    preprocessed_midi = preprocess_midi(
        midi_path,
        config=midi_config,
        start_time=start_point,
        duration=duration,
    )

    return PreprocessedSample(
        duration=duration,
        spectrogram=spectrogram,
        spectrogram_config=spectrogram_config,
        midi_data=preprocessed_midi,
        midi_config=midi_config,
    )


class AudioMidiDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        num_sampled: int,
        sample_duration: float,
        spectrogram_config: MelSpectrogramConfig | None = None,
        midi_config: MidiPreprocessingConfig | None = None,
    ):
        super().__init__()
        self.spectrogram_config = spectrogram_config
        self.midi_config = midi_config
        self.metadata = metadata
        self.num_sampled = num_sampled
        self.sample_duration = sample_duration

    def __len__(self):
        return self.num_sampled

    def num_notes(self):
        if self.midi_config is not None:
            midi_config = self.midi_config
        else:
            midi_config = MidiPreprocessingConfig(0)
        return midi_config.max_note - midi_config.min_note

    def __getitem__(self, _) -> PreprocessedSample:
        random_index = np.random.choice(len(self.metadata))
        file_info = FileInfo(**self.metadata.iloc[random_index])
        return preprocess_random_segment(
            file_info=file_info,
            duration=self.sample_duration,
            spectrogram_config=self.spectrogram_config,
            midi_config=self.midi_config,
        )


@dataclass
class Batch:
    spectrograms: Tensor
    midi_data: Tensor


def _collate(samples: list[PreprocessedSample]) -> Batch:
    return Batch(
        spectrograms=torch.stack([sample.spectrogram for sample in samples]),
        midi_data=torch.stack([sample.midi_data for sample in samples]),
    )


def make_dataloader(
    split: str,
    samples_per_epoch: int,
    batch_size: int,
    sample_duration: float,
    num_workers: int,
):
    metadata = load_metadata()
    metadata = metadata.loc[metadata.split == split]
    assert len(metadata) > 0
    dataset = AudioMidiDataset(metadata, samples_per_epoch, sample_duration)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=_collate,
        num_workers=num_workers,
    )
