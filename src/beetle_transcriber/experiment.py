"""Configuration piping and file utils for training runs."""

from pathlib import Path

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from beetle_transcriber.config import Config
from beetle_transcriber import models
from beetle_transcriber.audio import SpectrogramConfig
from beetle_transcriber.midi import MidiPreprocessingConfig
from beetle_transcriber.training import (
    LearningConfig,
    LossConfig,
    Loss,
    Learner,
)
from beetle_transcriber.dataset import (
    DataLoadingConfig,
    make_dataloader,
    load_metadata,
)


class ExperimentConfig(Config):
    model: models.ModelConfig
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


def find_latest_checkpoint(experiment_dir: Path) -> Path:
    checkpoints = list(
        (experiment_dir / "lightning_logs" / "version_0" / "checkpoints").glob("*.ckpt")
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {experiment_dir}.")
    if len(checkpoints) > 1:
        raise ValueError(f"Multiple checkpoints found in {experiment_dir}")
    return checkpoints[0]


def run_experiment(
    config: ExperimentConfig,
    target_dir: Path,
    checkpoint_path: Path | None,
):
    model = models.init_model_from_config(config.model)
    metadata = load_metadata()

    train_dataloader = make_dataloader(
        split="train",
        sample_duration=config.window_length_seconds,
        data_loading_config=config.train_dataloader,
        spectrogram_config=config.spectrogram,
        midi_config=config.midi,
        metadata=metadata,
    )
    valid_dataloader = make_dataloader(
        split="validation",
        sample_duration=config.window_length_seconds,
        data_loading_config=config.valid_dataloader,
        spectrogram_config=config.spectrogram,
        midi_config=config.midi,
        metadata=metadata,
    )
    loss = Loss(config.loss)
    learner = Learner(
        model=model,
        loss=loss,
        config=config.learning,
    )
    callbacks = [
        ModelCheckpoint(
            monitor="valid/loss",
            filename="{epoch:02d}-{val_loss:.4f}",
            save_top_k=1,
            mode="min",
        ),
    ]
    if config.early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor="valid/loss",
                patience=config.early_stopping_patience,
                mode="min",
                verbose=True,
            )
        )
    trainer = pl.Trainer(
        callbacks=callbacks,
        default_root_dir=target_dir,
        accumulate_grad_batches=config.gradient_accumulation,
        precision=config.precision,
        max_epochs=config.max_epochs,
        num_sanity_val_steps=0,  # Nothing sane about any of this.
    )
    trainer.fit(
        learner,
        train_dataloader,
        valid_dataloader,
        ckpt_path=checkpoint_path,
    )
