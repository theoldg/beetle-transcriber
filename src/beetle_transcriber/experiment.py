"""Configuration piping and file utils for training runs."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from beetle_transcriber.config import Config
from beetle_transcriber import models
from beetle_transcriber.config import Config
from beetle_transcriber.audio import SpectrogramConfig
from beetle_transcriber.midi import MidiPreprocessingConfig
from beetle_transcriber.training import (
    LearningConfig,
    LossConfig,
    Learner,
)
from beetle_transcriber.dataset import (
    DataLoadingConfig,
)


@dataclass
class ModelSpec:
    model_cls: type[nn.Module]
    config_schema: type[Config]


MODELS = {
    "v1": ModelSpec(models.UNetV1, models.UNetV1Config),
    "v2": ModelSpec(models.UNetV2, models.UNetV2Config),
}


class ModelConfig(Config):
    name: str
    config: dict


class TrainingConfig(Config):
    model: ModelConfig
    num_notes: int = 88
    window_length_seconds: float = 2.96
    spectrogram: SpectrogramConfig
    midi: MidiPreprocessingConfig
    learning: LearningConfig
    loss: LossConfig
    gradient_accumulation: int = 1
    train_dataloader: DataLoadingConfig
    valid_dataloader: DataLoadingConfig
    precision: str = "32-true"
    early_stopping_patience: int | None = None
    max_epochs: int = 1_000


def init_model_from_config(model_config: ModelConfig) -> nn.Module:
    model_spec = MODELS[model_config.name]
    model_config = model_spec.config_schema(**model_config.config)
    model = model_spec.model_cls(model_config)
    return model


def find_latest_checkpoint(experiment_dir: Path) -> Path:
    checkpoints = list(
        (experiment_dir / "lightning_logs" / "version_0" / "checkpoints").glob("*.ckpt")
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {experiment_dir}.")
    if len(checkpoints) > 1:
        raise ValueError(f"Multiple checkpoints found in {experiment_dir}")
    return checkpoints[0]


def load_model_for_inference(experiment_dir: Path | str) -> nn.Module:
    experiment_dir = Path(experiment_dir)
    config = TrainingConfig.load_from_yaml(experiment_dir / "config.yaml")
    model = init_model_from_config(config.model)
    checkpoint_path = find_latest_checkpoint(experiment_dir)
    learner = Learner(model, None, None)  # This is a hack.
    learner.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    learner.eval()
    return learner.model
