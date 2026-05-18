"""Preprocessing MIDI for training. All time is in seconds."""

from dataclasses import dataclass
from enum import Enum
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

    # Highest note on the piano (exclusivve).
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
    file: mido.MidiFile,
    start_time: float,
    duration: float,
) -> list[Note]:
    note_map = {}
    notes: list[Note] = []
    time = 0
    for message in file:
        if hasattr(message, 'time'):
            time += message.time
            if time > start_time + duration:
                break
        if message.type != 'note_on':
            # Notes off are annotated as "note_on" with velocity 0.
            continue
        if message.velocity != 0:
            # Note start.
            if message.note in note_map:
                raise ValueError(f'Note started twice: {message}')
            note_map[message.note] = message, time
        else:
            # Note end.
            if message.note not in note_map:
                raise ValueError(f'Ended note that had not started: {message}')
            start_message, start_time = note_map.pop(message.note)
            notes.append(Note(
                note=message.note,
                velocity=start_message.velocity,
                start_time=start_time,
                end_time=time,
            ))
    
    filtered_notes = []
    for note in notes:
        if note.start_time < start_time:
            continue
        if note.start_time >= start_time + duration:
            break
        note.start_time -= start_time
        note.end_time -= start_time
        note.end_time = min(note.end_time, duration)

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


def preprocess_midi(
    path: Path,
    config: MidiPreprocessingConfig,
    start_time: float,
    duration: float,
) -> torch.Tensor:
    file = mido.MidiFile(path)
    notes = _find_notes(file, start_time=start_time, duration=duration)
    
    num_time_steps = math.ceil(duration / config.time_resolution)
    num_notes = config.max_note - config.min_note

    data = torch.zeros(
        (num_time_steps, num_notes, NUM_CHANNELS),
        dtype=torch.float32,
    )

    for note in notes:
        time_step = round(note.start_time / config.time_resolution)
        offset = note.start_time - time_step * config.time_resolution
        note_shifted = note.note - config.min_note

        data_point = data[time_step, note_shifted]
        data_point[Channel.CONFIDENCE] = 1.
        data_point[Channel.OFFSET] = offset
        data_point[Channel.DURATION] = note.end_time - note.start_time
        data_point[Channel.VELOCITY] = note.velocity
    
    _normalize_sample(data, config)
    return data
