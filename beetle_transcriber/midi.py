"""Preprocessing MIDI for training. All time is in seconds."""

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import math

import mido
import torch


@dataclass
class MidiPreprocessingConfig:
    # In seconds.
    time_resolution: float

    # Lowest note on the piano (inclusive).
    min_note: int = 21

    # Highest note on the piano (exclusive).
    max_note: int = 109


@dataclass
class Note:
    # Midi value, 0-127.
    note: int

    # Midi value, 0-127.
    velocity: int

    # In seconds.
    start_time: float
    end_time: float


def _find_notes(
    path: Path,
    start_time: float,
    duration: float,
) -> list[Note]:
    file = mido.MidiFile(path)

    note_map = {}
    notes: list[Note] = []
    time = 0
    for message in file:
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

    filtered_notes = []
    for note in notes:
        if note.start_time < start_time:
            continue
        if note.start_time >= start_time + duration:
            break
        note.start_time -= start_time
        note.end_time -= start_time
        note.end_time = min(note.end_time, duration)
        filtered_notes.append(note)

    return filtered_notes


class Channel:
    CONFIDENCE = 0
    OFFSET = 1
    DURATION = 2
    VELOCITY = 3


NUM_CHANNELS = 4


def _normalize_sample(
    data: torch.Tensor,
    config: MidiPreprocessingConfig,
) -> None:
    # Normalized offset ranges from -1 to 1.
    data[..., Channel.OFFSET] /= config.time_resolution / 2

    # Median note duration seems to be around 100 ms.
    data[..., Channel.DURATION] /= 0.1

    # The maximum velocity is 127.
    data[..., Channel.VELOCITY] /= 128


def _get_gaussian_window(radius: int, strength: float):
    x = torch.linspace(-radius, radius, 2 * radius + 1)
    return torch.exp(-((x / strength) ** 2))


def preprocess_midi(
    path: Path,
    config: MidiPreprocessingConfig,
    start_time: float,
    duration: float,
    smoothing_radius: int = 0,
    smoothing_strength: float = 0.5,
) -> torch.Tensor:
    notes = _find_notes(path, start_time=start_time, duration=duration)

    num_time_steps = math.ceil(duration / config.time_resolution)
    num_notes = config.max_note - config.min_note

    data = torch.zeros(
        (num_time_steps, num_notes, NUM_CHANNELS),
        dtype=torch.float32,
    )

    window_weights = _get_gaussian_window(smoothing_radius, smoothing_strength)

    for note, dt in product(notes, range(-smoothing_radius, smoothing_radius + 1)):
        time_step = dt + round(note.start_time / config.time_resolution)
        if not (0 <= time_step < num_time_steps):
            continue
        offset = note.start_time - time_step * config.time_resolution
        note_shifted = note.note - config.min_note
        assert note_shifted >= 0

        data_point = torch.zeros(NUM_CHANNELS, dtype=torch.float32)
        data_point[Channel.CONFIDENCE] = 1.0
        data_point[Channel.OFFSET] = offset
        data_point[Channel.DURATION] = note.end_time - note.start_time
        data_point[Channel.VELOCITY] = note.velocity

        weight = window_weights[dt + smoothing_radius]
        data[time_step, note_shifted] += data_point * weight

    _normalize_sample(data, config)
    return data
