from torch import nn
from pathlib import Path
from fire import Fire
import pandas as pd

from beetle_transcriber.dataset import load_metadata, get_maestro_path
from beetle_transcriber.midi import find_notes
from beetle_transcriber.inference import windowed_inference, load_model_for_inference
from beetle_transcriber.audio import AudioPreprocessor
from beetle_transcriber.evaluation import match_notes, MatchingResult
from beetle_transcriber.experiment import ExperimentConfig


def evaluate_entire_dataset(
    model: nn.Module,
    audio_preprocessor: AudioPreprocessor,
    duration: float,
    min_note: int,
    nms_radius: int,
    threshold: float = 0.7,
    batch_size: int = 32,
    device: str = "mps",
    num_workers: int = 10,
):
    metadata = load_metadata().query('split == "validation"')

    audio_paths = [
        get_maestro_path() / audio_filename
        for audio_filename in metadata.audio_filename
    ]
    detected_note_lists = windowed_inference(
        audio_paths=audio_paths,
        model=model,
        audio_preprocessor=audio_preprocessor,
        duration=duration,
        min_note=min_note,
        nms_radius=nms_radius,
        threshold=threshold,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        verbose=True,
    )
    true_note_lists = [
        find_notes(
            midi_filename,
            start_time=0,
            duration=float("inf"),
        )
        for midi_filename in metadata.midi_filename
    ]
    matching_results = [
        match_notes(
            true_notes=true_notes,
            detected_notes=detected_notes,
        )
        for true_notes, detected_notes in zip(true_note_lists, detected_note_lists)
    ]
    return matching_results


def metrics_per_tolerance(result: MatchingResult) -> dict:
    ret = {}
    for tolerance in (5, 10, 15, 20, 30, 50):
      r = result.apply_tolerance(tolerance)
      ret[tolerance] = {
        'recall': r.recall,
        'precision': r.precision,
        'f1_score': r.f1_score,
      }
    return ret


def main(
    experiment_path: str,
    device: str = "mps",
    threshold: float = 0.7,
):
    experiment_path = Path(experiment_path)
    config = ExperimentConfig.load_from_yaml(experiment_path / "config.yaml")
    audio_preprocessor = AudioPreprocessor(config.spectrogram)
    model = load_model_for_inference(experiment_path)
    model = model.to(device)
    matching_results = evaluate_entire_dataset(
        model=model,
        audio_preprocessor=audio_preprocessor,
        duration=config.window_length_seconds,
        min_note=config.midi.min_note,
        nms_radius=config.midi.smoothing_radius,
        threshold=threshold,
        device=device,
    )
    # return matching_results
    merged_results = MatchingResult.sum(matching_results)
    print(pd.DataFrame(metrics_per_tolerance(merged_results)))


if __name__ == "__main__":
    Fire(main)
