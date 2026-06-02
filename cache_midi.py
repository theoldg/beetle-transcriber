"""
At some point it turned out that loading MIDI files dynamically during training
is the bottleneck. This script rewrites them to numpy arrays which can then
be written to disk, memory mapped, and binary searched for much quicker reading.
"""

from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from pathlib import Path

from beetle_transcriber.dataset import load_metadata, get_maestro_path
from beetle_transcriber.midi import MIDI_CACHE_LOCATION, midi_to_array


def load_and_cache(name: str) -> None:
    f = get_maestro_path() / name
    arr = midi_to_array(f)
    target_path: Path = (MIDI_CACHE_LOCATION / name).with_suffix(".npy")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(target_path, arr)


if __name__ == "__main__":
    metadata = load_metadata()
    with ProcessPoolExecutor(18) as ex:
        loaded_message_lists = list(
            tqdm(
                ex.map(load_and_cache, metadata.midi_filename),
                total=len(metadata),
            )
        )
    print("Done!")
