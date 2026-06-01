from pathlib import Path

import torch
from torch import Tensor, BoolTensor
from torch import nn
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from tqdm import tqdm

from beetle_transcriber.audio import load_audio_segment
from beetle_transcriber.midi import Note, Channel
from beetle_transcriber.audio import AudioPreprocessor


HARDCODED_NOTE_LENGTH = 0.010  # 10 ms


def _decode_single(
    model_outputs: Tensor,
    mask: BoolTensor,
    time_resolution: float,
    min_note: int,
) -> list[Note]:
    notes = []
    for time_i, note_i in mask.nonzero():
        out = model_outputs[time_i, note_i]
        offset = out[Channel.OFFSET] * time_resolution / 2
        bin_time = time_resolution * time_i
        start_time = float(bin_time + offset)
        velocity = out[Channel.VELOCITY] * 128
        notes.append(
            Note(
                note=int(note_i + min_note),
                velocity=int(velocity),
                start_time=start_time,
                end_time=start_time + HARDCODED_NOTE_LENGTH,
            )
        )
    notes = sorted(notes, key=lambda note: note.start_time)
    return notes


def decode_notes(
    model_output: Tensor,
    time_resolution: float,
    min_note: int,
    radius: int,
    threshold: float = 0.7,
) -> list[list[Note]]:
    confidence_score = torch.sigmoid(model_output[..., Channel.CONFIDENCE_MAX])
    confidence_pooled = torch.nn.functional.max_pool1d(
        confidence_score.transpose(1, 2),
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    ).transpose(1, 2)

    mask = (confidence_score == confidence_pooled) & (confidence_score >= threshold)
    return [
        _decode_single(output_single, mask_single, time_resolution, min_note)
        for output_single, mask_single in zip(model_output, mask)
    ]


class WindowedDataset(Dataset):
    def __init__(
        self,
        audio_path: Path,
        audio_preprocessor: AudioPreprocessor,
        duration: float,
    ):
        super().__init__()
        self.hop_size = duration / 2
        self.audio_path = audio_path
        self.audio_preprocessor = audio_preprocessor
        self.duration = duration

    def __len__(self):
        audio_file = sf.SoundFile(self.audio_path)
        file_seconds = len(audio_file) / audio_file.samplerate
        return int((file_seconds - self.duration) / self.hop_size)

    def __getitem__(self, index):
        wave = load_audio_segment(
            file_path=self.audio_path,
            start_sec=self.hop_size * index,
            duration_sec=self.duration,
            samplerate=self.audio_preprocessor.config.sample_rate,
        )
        return self.audio_preprocessor(wave)


def windowed_inference(
    audio_path: Path,
    model: nn.Module,
    audio_preprocessor: AudioPreprocessor,
    duration: float,
    min_note: int,
    nms_radius: int,
    threshold: float = 0.7,
    batch_size: int = 16,
    device: str = "mps",
) -> list[Note]:
    dataset = WindowedDataset(
        audio_path=audio_path,
        audio_preprocessor=audio_preprocessor,
        duration=duration,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
    )
    notes_windowed = []
    with torch.inference_mode():
        for spec in tqdm(dataloader):
            spec = spec.to(device)
            output = model(spec).cpu()
            notes_windowed.extend(
                decode_notes(
                    output,
                    time_resolution=audio_preprocessor.time_resolution,
                    min_note=min_note,
                    threshold=threshold,
                    radius=nms_radius,
                )
            )
    num_windows = len(notes_windowed)
    notes = []
    for i, window_notes in enumerate(notes_windowed):
        for note in window_notes:
            note: Note
            if ((duration / 4 < note.start_time) or (i == 0)) and (
                (note.start_time < duration * 3 / 4) or (i == num_windows - 1)
            ):
                offset = i * duration / 2
                note.start_time += offset
                note.end_time += offset
                notes.append(note)
    return notes
