import torch
from torch import Tensor, BoolTensor

from beetle_transcriber.midi import Note, Channel


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
        velocity = out[Channel.VELOCITY] * 128
        notes.append(Note(
            note=int(note_i + min_note),
            velocity=int(velocity),
            start_time=float(bin_time),
            end_time=float(bin_time + offset),
        ))
    return notes


def decode(
    model_output: Tensor,
    time_resolution: float,
    min_note: int,
    radius: int,
    threshold: float = .7
) -> list[list[Note]]:
    confidence_score = torch.sigmoid(model_output[..., Channel.CONFIDENCE_MAX])
    confidence_pooled = torch.nn.functional.max_pool1d(
        confidence_score.transpose(1, 2),
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    ).transpose(1, 2)

    mask = (
        (confidence_score == confidence_pooled) 
        & (confidence_score >= threshold)
    )
    return [
        _decode_single(output_single, mask_single, time_resolution, min_note)
        for output_single, mask_single
        in zip(model_output, mask)
    ]
