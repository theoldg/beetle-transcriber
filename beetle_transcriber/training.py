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
    # Chosen by fair dice roll, guaranteed to be random.
    empty_weight: float = 15

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
    metrics: dict[str, float | Tensor]


class Loss(nn.Module):
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config

    def calculate_classification_metrics(
        self,
        model_out: Tensor,
        ground_truth: Tensor,
    ) -> dict[str, float | Tensor]:
        is_note = ground_truth[:, :, :, Channel.CONFIDENCE] == 1
        is_note_model = model_out[..., Channel.CONFIDENCE] >= 0
        precision = (is_note_model & is_note).sum() / is_note_model.sum()
        recall = (is_note_model & is_note).sum() / is_note.sum()
        f1_score = 2 * precision * recall / (precision + recall)
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    def forward(self, model_out: Tensor, ground_truth: Tensor) -> LossOutput:
        batch, time, note, channels = model_out.shape
        loss = torch.zeros(batch, time, note, device=model_out.device)
        metrics = {}

        is_note = ground_truth[:, :, :, Channel.CONFIDENCE] == 1

        # Binary cross-entropy for non-notes.
        bce_empty = nn.functional.binary_cross_entropy_with_logits(
            input=model_out[~is_note][:, Channel.CONFIDENCE],
            target=torch.zeros((~is_note).sum(), device=model_out.device),  # type: ignore
            reduction="none",
        )
        loss[~is_note] += (
            self.config.cross_entropy_weight * self.config.empty_weight * bce_empty
        )
        metrics["bce_empty"] = bce_empty.mean()

        if not is_note.any():
            # TODO: This can conceivably happen but should be extremely rare.
            # Delete once training is up and running and it happens never or once in a while.
            print("Weird: entire batch with zero notes.")
            return LossOutput(
                loss=loss.mean(),
                metrics=metrics,
            )

        # Binary cross-entropy for notes.
        bce_note = nn.functional.binary_cross_entropy_with_logits(
            input=model_out[is_note][:, Channel.CONFIDENCE],
            target=torch.ones(is_note.sum(), device=model_out.device),  # type: ignore
            reduction="none",
        )
        loss[is_note] += self.config.cross_entropy_weight * bce_note
        metrics["bce_note"] = bce_note.mean()

        # Offset.
        offset_loss = torch.abs(
            model_out[is_note][:, Channel.OFFSET]
            - ground_truth[is_note][:, Channel.OFFSET]
        ) ** (self.config.offset_pow)
        loss[is_note] += self.config.offset_weight * offset_loss
        metrics["offset_loss"] = offset_loss.mean()

        # Duration.
        duration_loss = torch.abs(
            model_out[is_note][:, Channel.DURATION]
            - ground_truth[is_note][:, Channel.DURATION]
        ) ** (self.config.duration_pow)
        loss[is_note] += self.config.duration_weight * duration_loss
        metrics["duration_loss"] = duration_loss.mean()

        # Velocity.
        velocity_loss = torch.abs(
            model_out[is_note][:, Channel.VELOCITY]
            - ground_truth[is_note][:, Channel.VELOCITY]
        ) ** (self.config.velocity_pow)
        loss[is_note] += self.config.velocity_weight * velocity_loss
        metrics["velocity_loss"] = velocity_loss.mean()

        note_means = (loss * is_note).sum(-1).sum(-1) / is_note.sum(-1).sum(-1).clamp(
            min=1
        )
        empty_means = (loss * ~is_note).sum(-1).sum(-1) / (~is_note).sum(-1).sum(-1)

        metrics |= self.calculate_classification_metrics(model_out, ground_truth)
        return LossOutput(
            loss=(note_means + empty_means).mean(),
            metrics=metrics,
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
            raise RuntimeError("NaN model outputs.")
        loss: LossOutput = self.loss(model_out, batch.midi_data)
        if torch.isnan(loss.loss):
            raise RuntimeError("NaN loss.")

        self.log("train_loss", loss.loss, prog_bar=True, on_step=True)

        for metric_name, value in loss.metrics.items():
            self.log(metric_name, value, prog_bar=True, on_step=True)

        return loss.loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
