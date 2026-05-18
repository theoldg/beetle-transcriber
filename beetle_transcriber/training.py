from dataclasses import dataclass

import torch
from torch import Tensor
from torch import nn
import lightning as pl

from beetle_transcriber.midi import Channel
from beetle_transcriber import dataset

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
        batch, time, note, channels = model_out.shape
        loss = torch.zeros(batch, time, note)

        is_note = ground_truth[:, :, :, Channel.CONFIDENCE] == 1        

        # Binary cross-entropy for non-notes.
        loss[~is_note] += (
            self.config.cross_entropy_weight
            * nn.functional.binary_cross_entropy_with_logits(
                input=model_out[~is_note][:, Channel.CONFIDENCE],
                target=torch.zeros((~is_note).sum()),
                reduction='none',
            )
        )

        if not is_note.any():
            # This can happen occationally but shouldn't be more than e.g. 10% of the time.
            # TODO: remove when confident that this is not an issue.
            print('Weird: segment with zero notes')
            return loss[~is_note].mean()
        
        # Binary cross-entropy for notes.
        loss[is_note] += (
            self.config.cross_entropy_weight
            * nn.functional.binary_cross_entropy_with_logits(
                input=model_out[is_note][:, Channel.CONFIDENCE],
                target=torch.ones(is_note.sum()),
                reduction='none',
            )
        )

        # Offset.
        loss[is_note] += (
            self.config.offset_weight 
            * torch.abs(
                model_out[is_note][:, Channel.OFFSET] - ground_truth[is_note][:, Channel.OFFSET]
            ) ** (self.config.offset_pow)
        )
        # Duration.
        loss[is_note] += (
            self.config.duration_weight
            * torch.abs(
                model_out[is_note][:, Channel.DURATION] - ground_truth[is_note][:, Channel.DURATION]
            ) ** (self.config.duration_pow)
        )
        # Velocity.
        loss[is_note] += (
            self.config.velocity_weight 
            * torch.abs(
                model_out[is_note][:, Channel.VELOCITY] - ground_truth[is_note][:, Channel.VELOCITY]
            ) ** (self.config.velocity_pow)
        )

        note_means = (loss * is_note).sum(-1).sum(-1) / is_note.sum(-1).sum(-1)
        empty_means = (loss * ~is_note).sum(-1).sum(-1) / (~is_note).sum(-1).sum(-1)
        return (note_means + empty_means).mean()


class Trainer(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss: Loss,
    ):
        super().__init__()
        self.model = model
        self.loss = loss

    def training_step(self, batch: dataset.Batch, _):
        model_out = self.model(batch.spectrograms)
        loss = self.loss(model_out, batch.midi_data)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

