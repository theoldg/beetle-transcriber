"""Preprocessing MIDI for training. All time is in seconds."""

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import math

import mido
import torch
import numpy as np

from beetle_transcriber.config import Config


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
    CONFIDENCE_SUM = 0
    CONFIDENCE_MAX = 1
    OFFSET = 2
    VELOCITY = 3
    # DURATION = 4


NUM_CHANNELS = 4


def _normalize_sample(data: torch.Tensor) -> None:
    # The maximum velocity is 127.
    data[..., Channel.VELOCITY] /= 128


def preprocess_midi(
    path: Path,
    config: MidiPreprocessingConfig,
    time_resolution: float,
    start_time: float,
    duration: float,
) -> torch.Tensor:
    notes = _find_notes(path, start_time=start_time, duration=duration)

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

    _normalize_sample(data)
    return data
