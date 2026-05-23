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
from beetle_transcriber.dataset import DataLoadingConfig, make_dataloader
from beetle_transcriber import model


@dataclass
class ModelSpec:
    model_cls: nn.Module
    config_schema: Config


MODELS = {
    'v1': ModelSpec(model.UNetV1, model.UNetV1Config),
    'v2': ModelSpec(model.UNetV2, model.UNetV2Config),
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
    gradient_accumulation: int = 8
    train_dataloader: DataLoadingConfig
    valid_dataloader: DataLoadingConfig


def train(
    config: TrainingConfig,
    target_dir: Path,
):
    model_spec = MODELS[config.model.name]
    model_config = model_spec.config_schema(**config.model.config)
    model = model_spec.model_cls(model_config)

    train_dataloader = make_dataloader(
        split='train',
        sample_duration=config.window_length_seconds,
        data_loading_config=config.train_dataloader,
        spectrogram_config=config.spectrogram,
        midi_config=config.midi,
    )
    valid_dataloader = make_dataloader(
        split='validation',
        sample_duration=config.window_length_seconds,
        data_loading_config=config.valid_dataloader,
        spectrogram_config=config.spectrogram,
        midi_config=config.midi,
    )
    loss = Loss(config.loss)
    learner = Learner(
        model=model,
        loss=loss,
        config=config.learning,
    )
    trainer = pl.Trainer(
        callbacks=[
            EarlyStopping(
                monitor="valid/loss",
                patience=15,
                mode="min",
                verbose=True,
            ),
            ModelCheckpoint(
                monitor="valid/loss",
                filename="beetle-unet-{epoch:02d}-{val_loss:.4f}",
                save_top_k=1,
                mode="min"
            ),
        ],
        default_root_dir=target_dir,
        accumulate_grad_batches=config.gradient_accumulation,
    )
    trainer.fit(learner, train_dataloader, valid_dataloader)


DEFAULT_OUT_DIR = Path('experiments')


def main(config: str, target_dir: str | None = None):
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
    
    target_dir.mkdir(parents=True)
    shutil.copy(config, target_dir / 'config.yaml')

    train(config_parsed, target_dir)


if __name__ == '__main__':
    Fire(main)
