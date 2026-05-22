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

    confidence_sum_pow: float = 2
    confidence_sum_weight: float = 1

    offset_pow: float = 2
    offset_weight: float = 1

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
        is_note = ground_truth[..., Channel.CONFIDENCE_MAX] == 1
        is_note_model = model_out[..., Channel.CONFIDENCE_MAX] >= 0
        precision = (is_note_model & is_note).sum() / (is_note_model.sum() or 1)
        recall = (is_note_model & is_note).sum() / (is_note.sum() or 1)
        f1_score = 2 * precision * recall / ((precision + recall) or 1)
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    def forward(self, model_out: Tensor, ground_truth: Tensor) -> LossOutput:
        batch, time, note, channels = model_out.shape
        loss = torch.zeros(batch, time, note, device=model_out.device)
        metrics = {}
        empty_idx = ground_truth[..., Channel.CONFIDENCE_SUM] == 0

        # Max confidence: binary cross-entropy.
        bce = nn.functional.binary_cross_entropy_with_logits(
            input=model_out[..., Channel.CONFIDENCE_MAX],
            target=ground_truth[..., Channel.CONFIDENCE_MAX],
            reduction="none",
        )
        loss += self.config.cross_entropy_weight * bce
        metrics["bce"] = bce[~empty_idx].mean()

        # Total confidence: regression.
        confidence_sum_loss = torch.abs(
            model_out[..., Channel.CONFIDENCE_SUM]
            - ground_truth[..., Channel.CONFIDENCE_SUM]
        ) ** (self.config.confidence_sum_pow)
        loss += self.config.confidence_sum_weight * confidence_sum_loss
        metrics["confidence_sum"] = confidence_sum_loss[~empty_idx].mean()

        sub_loss_weight = ground_truth[..., Channel.CONFIDENCE_MAX]

        # Offset: regression.
        offset_loss = sub_loss_weight * torch.abs(
            model_out[..., Channel.OFFSET] - ground_truth[..., Channel.OFFSET]
        ) ** (self.config.offset_pow)
        loss += self.config.offset_weight * offset_loss
        metrics["offset_loss"] = offset_loss[~empty_idx].mean()

        # Velocity: regression.
        velocity_loss = sub_loss_weight * torch.abs(
            model_out[..., Channel.VELOCITY] - ground_truth[..., Channel.VELOCITY]
        ) ** (self.config.velocity_pow)
        loss += self.config.velocity_weight * velocity_loss
        metrics["velocity_loss"] = velocity_loss[~empty_idx].mean()

        empty_loss = loss[empty_idx].mean()
        nonempty_loss = loss[~empty_idx].mean()
        metrics["emtpy"] = empty_loss
        metrics["nonempty"] = nonempty_loss

        metrics |= self.calculate_classification_metrics(model_out, ground_truth)

        return LossOutput(loss=empty_loss + nonempty_loss, metrics=metrics)


@dataclass
class LearningConfig:
    learning_rate: float = 5e-4


class Learner(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss: Loss,
        config: LearningConfig,
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.config = config

    def training_step(self, batch: dataset.Batch, _):
        model_out = self.model(batch.spectrograms)
        if torch.isnan(model_out).any():
            raise RuntimeError("NaN model outputs.")
        loss: LossOutput = self.loss(model_out, batch.midi_data)
        if torch.isnan(loss.loss):
            raise RuntimeError("NaN loss.")

        self.log("train/loss", loss.loss, prog_bar=True, on_step=True)

        for metric_name, value in loss.metrics.items():
            self.log("train/" + metric_name, value, prog_bar=True, on_step=True)

        return loss.loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
        return optimizer
