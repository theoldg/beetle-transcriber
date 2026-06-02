from pathlib import Path
from itertools import count
import shutil

from fire import Fire

from beetle_transcriber.experiment import (
    find_latest_checkpoint,
    ExperimentConfig,
    run_experiment,
)

DEFAULT_OUT_DIR = Path("experiments")


def main(config: str, target_dir: str | None = None, resume_from: str | None = None):
    config_path = Path(config)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    config_parsed = ExperimentConfig.load_from_yaml(config_path)

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

    run_experiment(config_parsed, target_dir, checkpoint_path)


if __name__ == "__main__":
    Fire(main)
