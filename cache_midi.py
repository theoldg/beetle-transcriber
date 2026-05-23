from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from pathlib import Path


from beetle_transcriber.dataset import load_metadata, MAESTRO_PATH
from beetle_transcriber.midi import MIDI_CACHE_LOCATION, midi_to_array


def load_and_cache(name: str) -> None:
    f = MAESTRO_PATH / name
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
