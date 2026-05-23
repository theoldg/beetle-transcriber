"""Preprocessing MIDI for training. All time is in seconds."""

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import math
from typing import Self

import mido
import torch
import numpy as np

from beetle_transcriber.config import Config

MIDI_CACHE_LOCATION = Path(__file__).parents[2] / "cache" / "midi"


class MidiPreprocessingConfig(Config):
    # Lowest note on the piano (inclusive).
    min_note: int = 21

    # Highest note on the piano (exclusive).
    max_note: int = 109

    smoothing_radius: int = 2
    # In seconds.
    smoothing_std: float = 0.07


@dataclass
class Note:
    # Midi value, 0-127.
    note: int

    # Midi value, 0-127.
    velocity: int

    # In seconds.
    start_time: float
    end_time: float

    def to_numbers(self) -> np.ndarray:
        return np.array(
            [
                self.start_time,
                self.end_time,
                self.note,
                self.velocity,
            ],
            dtype=torch.float32,
        )

    @classmethod
    def from_numbers(cls, numbers: np.ndarray) -> Self:
        start_time, end_time, note, velocity = numbers
        return cls(
            start_time=start_time,
            end_time=end_time,
            note=int(note),
            velocity=int(velocity),
        )


def midi_to_array(path: Path) -> np.array:
    messages = list(mido.MidiFile(path))

    note_map = {}
    notes: list[Note] = []
    time = 0
    for message in messages:
        if hasattr(message, "time"):
            time += message.time
        if message.type != "note_on":
            # Notes off are annotated as "note_on" with velocity 0.
            continue
        if message.velocity != 0:
            # Note start.
            if message.note in note_map:
                # Repeated start: just end the previous note.
                start_message, note_start_time = note_map.pop(message.note)
                notes.append(
                    Note(
                        note=message.note,
                        velocity=start_message.velocity,
                        start_time=note_start_time,
                        end_time=time,
                    )
                )
            note_map[message.note] = message, time
        else:
            # Note end.
            if message.note not in note_map:
                continue
            start_message, note_start_time = note_map.pop(message.note)
            notes.append(
                Note(
                    note=message.note,
                    velocity=start_message.velocity,
                    start_time=note_start_time,
                    end_time=time,
                )
            )

    return np.stack([n.to_numbers() for n in notes])


def _find_notes(file_name: str, start_time: float, duration: float) -> list[Note]:
    cached_npy_file = (MIDI_CACHE_LOCATION / file_name).with_suffix(".npy")
    if not cached_npy_file.exists():
        raise FileNotFoundError(
            f"File not found. Did you `uv run cache_midi.py`? {cached_npy_file}"
        )
    arr = np.load(
        cached_npy_file,
        mmap_mode="r",
    )
    start_i = np.searchsorted(arr[:, 0], start_time)
    end_i = np.searchsorted(arr[:, 0], start_time + duration)
    sub_arr = arr[start_i:end_i]
    notes = []
    for row in sub_arr:
        note = Note.from_numbers(row)
        # Shift to match window start.
        note.start_time -= start_time
        note.end_time -= start_time
        notes.append(note)
    return notes


class Channel:
    CONFIDENCE_SUM = 0
    CONFIDENCE_MAX = 1
    OFFSET = 2
    VELOCITY = 3
    # DURATION = 4


NUM_CHANNELS = 4


def _normalize_sample(data: torch.Tensor, time_resolution: float) -> None:
    data[..., Channel.OFFSET] /= time_resolution / 2

    # The maximum velocity is 127.
    data[..., Channel.VELOCITY] /= 128


def preprocess_midi(
    file_name: str,
    config: MidiPreprocessingConfig,
    time_resolution: float,
    start_time: float,
    duration: float,
) -> torch.Tensor:
    notes = _find_notes(file_name, start_time=start_time, duration=duration)

    num_time_steps = math.ceil(duration / time_resolution)
    num_notes = config.max_note - config.min_note

    data = torch.zeros(
        (num_time_steps, num_notes, NUM_CHANNELS),
        dtype=torch.float32,
    )

    r = config.smoothing_radius
    for note, dt in product(notes, range(-r, r + 1)):
        time_step = dt + round(note.start_time / time_resolution)
        if not (0 <= time_step < num_time_steps):
            continue
        note_index = note.note - config.min_note
        offset = note.start_time - time_step * time_resolution
        weight = np.exp(-0.5 * (offset / config.smoothing_std) ** 2)
        data_point = data[time_step, note_index]
        data_point[Channel.CONFIDENCE_SUM] += weight
        if weight > data_point[Channel.CONFIDENCE_MAX]:
            data_point[Channel.CONFIDENCE_MAX] = weight
            data_point[Channel.VELOCITY] = note.velocity
            data_point[Channel.OFFSET] = offset

    _normalize_sample(data, time_resolution=time_resolution)
    return data
