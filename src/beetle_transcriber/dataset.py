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
    SpectrogramConfig,
    load_audio_segment,
    AudioPreprocessor,
)
from beetle_transcriber.midi import MidiPreprocessingConfig, preprocess_midi
from beetle_transcriber.config import Config


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
    midi_data: Tensor


def preprocess_random_segment(
    file_info: FileInfo,
    duration: float,
    audio_preprocessor: AudioPreprocessor,
    midi_config: MidiPreprocessingConfig | None = None,
) -> PreprocessedSample:
    audio_path = MAESTRO_PATH / file_info.audio_filename
    assert audio_path.exists()

    if midi_config is None:
        midi_config = MidiPreprocessingConfig()  # Defaults.

    start_point = random.random() * (file_info.duration - duration)
    waveform = load_audio_segment(
        file_path=audio_path,
        start_sec=start_point,
        duration_sec=duration,
        samplerate=audio_preprocessor.config.sample_rate,
    )
    spectrogram = audio_preprocessor(waveform)

    preprocessed_midi = preprocess_midi(
        file_info.midi_filename,
        config=midi_config,
        start_time=start_point,
        duration=duration,
        time_resolution=audio_preprocessor.time_resolution,
    )

    return PreprocessedSample(
        duration=duration,
        spectrogram=spectrogram,
        midi_data=preprocessed_midi,
    )


class AudioMidiDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        num_sampled: int,
        sample_duration: float,
        spectrogram_config: SpectrogramConfig | None = None,
        midi_config: MidiPreprocessingConfig | None = None,
    ):
        super().__init__()
        if len(metadata) == 0:
            raise ValueError("Metadata DataFrame is empty.")
        if spectrogram_config is None:
            spectrogram_config = SpectrogramConfig()
        self.spectrogram_config = spectrogram_config
        self.midi_config = midi_config
        self.metadata = metadata
        self.num_sampled = num_sampled
        self.sample_duration = sample_duration
        self.audio_preprocessor = AudioPreprocessor(spectrogram_config)

    def __len__(self):
        return self.num_sampled

    def __getitem__(self, _) -> PreprocessedSample:
        random_index = np.random.choice(len(self.metadata))
        file_info = FileInfo(**self.metadata.iloc[random_index])
        return preprocess_random_segment(
            file_info=file_info,
            duration=self.sample_duration,
            audio_preprocessor=self.audio_preprocessor,
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


class DataLoadingConfig(Config):
    samples_per_epoch: int
    batch_size: int
    num_workers: int = 8


def make_dataloader(
    split: str,
    sample_duration: float,
    data_loading_config: DataLoadingConfig,
    metadata: pd.DataFrame | None = None,
    spectrogram_config: SpectrogramConfig | None = None,
    midi_config: MidiPreprocessingConfig | None = None,
):
    if metadata is None:
        metadata = load_metadata()
    metadata = metadata.loc[metadata.split == split]
    assert len(metadata) > 0
    dataset = AudioMidiDataset(
        metadata=metadata,
        num_sampled=data_loading_config.samples_per_epoch,
        sample_duration=sample_duration,
        spectrogram_config=spectrogram_config,
        midi_config=midi_config,
    )
    return DataLoader(
        dataset,
        batch_size=data_loading_config.batch_size,
        collate_fn=_collate,
        num_workers=data_loading_config.num_workers,
        persistent_workers=True,
    )
