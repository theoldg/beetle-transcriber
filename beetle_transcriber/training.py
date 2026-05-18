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


@dataclass
class LossOutput:
    loss: Tensor
    bce_note: float
    bce_empty: float
    offset: float
    duration: float
    velocity: float


class Loss(nn.Module):
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config

    def forward(self, model_out: Tensor, ground_truth: Tensor) -> LossOutput:
        batch, time, note, channels = model_out.shape
        loss = torch.zeros(batch, time, note, device=model_out.device)

        is_note = ground_truth[:, :, :, Channel.CONFIDENCE] == 1        

        # Binary cross-entropy for non-notes.
        bce_empty = nn.functional.binary_cross_entropy_with_logits(
            input=model_out[~is_note][:, Channel.CONFIDENCE],
            target=torch.zeros((~is_note).sum(), device=model_out.device),
            reduction='none',
        )
        loss[~is_note] += self.config.cross_entropy_weight * bce_empty

        if not is_note.any():
            # TODO: This can conceivably happen but should be extremely rare.
            # Delete once training is up and running and it happens never or once in a while.
            print('Weird: entire batch with zero notes.')
            return LossOutput(
                loss=loss.mean(),
                bce_empty=bce_empty.mean().item(),
                bce_note=0, offset=0, duration=0, velocity=0,
            )
        
        # Binary cross-entropy for notes.
        bce_note = nn.functional.binary_cross_entropy_with_logits(
            input=model_out[is_note][:, Channel.CONFIDENCE],
            target=torch.ones(is_note.sum(), device=model_out.device),
            reduction='none',
        )
        loss[is_note] += self.config.cross_entropy_weight * bce_note

        # Offset.
        offset_loss = torch.abs(
            model_out[is_note][:, Channel.OFFSET] - ground_truth[is_note][:, Channel.OFFSET]
        ) ** (self.config.offset_pow)
        loss[is_note] += self.config.offset_weight * offset_loss

        # Duration.
        duration_loss = torch.abs(
            model_out[is_note][:, Channel.DURATION] - ground_truth[is_note][:, Channel.DURATION]
        ) ** (self.config.duration_pow)
        loss[is_note] += self.config.duration_weight * duration_loss

        # Velocity.
        velocity_loss = torch.abs(
            model_out[is_note][:, Channel.VELOCITY] - ground_truth[is_note][:, Channel.VELOCITY]
        ) ** (self.config.velocity_pow)
        loss[is_note] += self.config.velocity_weight * velocity_loss

        note_means = (loss * is_note).sum(-1).sum(-1) / is_note.sum(-1).sum(-1).clamp(min=1)
        empty_means = (loss * ~is_note).sum(-1).sum(-1) / (~is_note).sum(-1).sum(-1)

        return LossOutput(
            loss=(note_means + empty_means).mean(),
            bce_note=bce_note.mean().item(),
            bce_empty=bce_empty.mean().item(),
            offset=offset_loss.mean().item(),
            velocity=velocity_loss.mean().item(),
            duration=duration_loss.mean().item(),
        )


class Learner(pl.LightningModule):
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
        if torch.isnan(model_out).any():
            raise RuntimeError('NaN model outputs.')
        loss: LossOutput = self.loss(model_out, batch.midi_data)
        if torch.isnan(loss.loss):
            raise RuntimeError('NaN loss.')

        self.log("train_loss", loss.loss, prog_bar=True, on_step=True)
        self.log("bce_note", loss.bce_note, prog_bar=True, on_step=True)
        self.log("bce_empty", loss.bce_empty, prog_bar=True, on_step=True)
        self.log("offset_loss", loss.offset, prog_bar=True, on_step=True)
        self.log("duration_loss", loss.duration, prog_bar=True, on_step=True)
        self.log("velocity_loss", loss.velocity, prog_bar=True, on_step=True)

        return loss.loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
