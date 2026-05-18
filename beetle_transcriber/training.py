from dataclasses import dataclass

import torch
from torch import Tensor
from torch import nn

from beetle_transcriber.midi import Channel


@dataclass
class LossConfig:
    cross_entropy_weight: float = 1

    offset_pow: float = 2
    offset_weight: float = 1

    duration_pow: float = 1
    # Ignore duration for now.
    duration_weight: float = 0

    velocity_pow: float = 1
    velocity_weight: float = 1


class Loss(nn.Module):
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config

    def forward(self, model_out: Tensor, ground_truth: Tensor) -> Tensor:
        # TODO: handle segments without any notes
        batch, time, note, channels = model_out.shape
        loss = torch.zeros(batch, time, note)

        is_note = ground_truth[:, :, :, Channel.CONFIDENCE] == 1        
        note_vals = loss[is_note]
        empty_vals = loss[~is_note]

        # Binary cross-entropy for non-notes.
        empty_vals += (
            self.config.cross_entropy_weight
            * -torch.log(1 - model_out[~is_note][:, Channel.CONFIDENCE])
        )
        if not is_note.any():
            return empty_vals.mean()
        
        # Binary cross-entropy for notes.
        note_vals += (
            self.config.cross_entropy_weight
            * -torch.log(model_out[is_note][:, Channel.CONFIDENCE])
        )

        # Offset.
        note_vals += (
            self.config.offset_weight 
            * torch.abs(
                model_out[is_note][:, Channel.OFFSET] - ground_truth[is_note][:, Channel.OFFSET]
            ) ** (self.config.offset_pow)
        )
        # Duration.
        note_vals += (
            self.config.duration_weight
            * torch.abs(
                model_out[is_note][:, Channel.DURATION] - ground_truth[is_note][:, Channel.DURATION]
            ) ** (self.config.duration_pow)
        )
        # Velocity.
        note_vals += (
            self.config.velocity_weight 
            * torch.abs(
                model_out[is_note][:, Channel.VELOCITY] - ground_truth[is_note][:, Channel.VELOCITY]
            ) ** (self.config.velocity_pow)
        )

        note_means = (loss * is_note).sum(-1).sum(-1) / is_note.sum(-1).sum(-1)
        empty_means = (loss * ~is_note).sum(-1).sum(-1) / (~is_note).sum(-1).sum(-1)
        return (note_means + empty_means).mean()
