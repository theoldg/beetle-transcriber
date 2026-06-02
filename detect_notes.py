from pathlib import Path

from fire import Fire
import pandas as pd

from beetle_transcriber.experiment import ExperimentConfig
from beetle_transcriber.inference import load_model_for_inference
from beetle_transcriber.audio import AudioPreprocessor
from beetle_transcriber.inference import windowed_inference


def main(
    input: str,
    output: str,
    experiment_path: str = 'trained_model',
    device: str = "mps",
    threshold: float = 0.7,
):
    experiment_path = Path(experiment_path)
    config = ExperimentConfig.load_from_yaml(experiment_path / "config.yaml")
    audio_preprocessor = AudioPreprocessor(config.spectrogram)
    model = load_model_for_inference(experiment_path)
    model = model.to(device)
    notes = windowed_inference(
        audio_path=input,
        model=model,
        audio_preprocessor=audio_preprocessor,
        duration=config.window_length_seconds,
        min_note=config.midi.min_note,
        nms_radius=config.midi.smoothing_radius,
        threshold=threshold,
        device=device,
    )
    dataframe = pd.DataFrame(notes)
    dataframe.to_csv(output, index=False)


if __name__ == "__main__":
    Fire(main)
