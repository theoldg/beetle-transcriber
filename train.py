from pathlib import Path
from itertools import count
import shutil

from fire import Fire
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping


from beetle_transcriber.training import Loss, Learner
from beetle_transcriber.dataset import make_dataloader, load_metadata
from beetle_transcriber.experiment import (
    find_latest_checkpoint,
    init_model_from_config,
    TrainingConfig,
)


def train(
    config: TrainingConfig,
    target_dir: Path,
    checkpoint_path: Path | None,
):
    model = init_model_from_config(config.model)
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


DEFAULT_OUT_DIR = Path("experiments")


def main(config: str, target_dir: str | None = None, resume_from: str | None = None):
    config_path = Path(config)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    config_parsed = TrainingConfig.load_from_yaml(config_path)

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
    shutil.copy(config_path, target_dir / "config.yaml")

    train(config_parsed, target_dir, checkpoint_path)


if __name__ == "__main__":
    Fire(main)
