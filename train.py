from pathlib import Path
from itertools import count
import shutil
from dataclasses import dataclass

from fire import Fire
import yaml
from torch import nn
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from beetle_transcriber.config import Config
from beetle_transcriber.audio import SpectrogramConfig
from beetle_transcriber.midi import MidiPreprocessingConfig
from beetle_transcriber.training import LearningConfig, LossConfig, Loss, Learner
from beetle_transcriber.dataset import DataLoadingConfig, make_dataloader, load_metadata
from beetle_transcriber import models


@dataclass
class ModelSpec:
    model_cls: nn.Module
    config_schema: Config


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
    early_stopping_patience: int = -1


def train(
    config: TrainingConfig,
    target_dir: Path,
    checkpoint_path: Path | None,
):
    model_spec = MODELS[config.model.name]
    model_config = model_spec.config_schema(**config.model.config)
    model = model_spec.model_cls(model_config)

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
    if config.early_stopping_patience > 0:
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
    )
    trainer.fit(
        learner,
        train_dataloader,
        valid_dataloader,
        ckpt_path=checkpoint_path,
    )


DEFAULT_OUT_DIR = Path("experiments")


def find_latest_checkpoint(target_dir: Path) -> Path:
    checkpoints = list(
        (target_dir / "lightning_logs" / "version_0" / "checkpoints").glob("*.ckpt")
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {target_dir}.")
    if len(checkpoints) > 1:
        raise ValueError(f"Multiple checkpoints found in {target_dir}")
    return checkpoints[0]


def main(config: str, target_dir: str | None = None, resume_from: str | None = None):
    config = Path(config)
    if not config.exists():
        raise FileNotFoundError(config)
    config_parsed = TrainingConfig(**yaml.safe_load(config.read_text()))

    if target_dir is None:
        for i in count():
            candidate_dir = DEFAULT_OUT_DIR / str(i)
            if not candidate_dir.exists():
                break
        target_dir = candidate_dir
    else:
        target_dir = Path(target_dir)
        if target_dir.exists():
            raise FileExistsError(target_dir)

    if resume_from is not None:
        checkpoint_path = find_latest_checkpoint(Path(resume_from))
    else:
        checkpoint_path = None

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config, target_dir / "config.yaml")

    train(config_parsed, target_dir, checkpoint_path)


if __name__ == "__main__":
    Fire(main)
