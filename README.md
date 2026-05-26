# Beetle Transcriber

In this project I attempt to create an audio-to-MIDI transcription model for piano, with near-zero prior knowledge about this problem. Beetles are deaf.

Trained on the [MAESTRO v3](https://magenta.withgoogle.com/datasets/maestro) dataset, the model takes a raw piano recording and outputs a structured prediction of which notes were played, when and at what velocity.

---

## Results

After a few days of iteration, the v2 model (2D UNet with harmonic lowering) reaches **>85% F1 score** at the current time resolution on the MAESTRO validation split.

Currently, the model only predicts onsets and key press velocity. Decoding back from the continuous predictions to a discrete MIDI file is WIP. Predicting note length is a step up in difficulty, which I will handle as soon as I get >90% F1 score on offset detection. It seems like [Onsets and Frames](https://magenta.withgoogle.com/onsets-frames) does this via an "activity" channel in the model output, trained to output 1 if a note is ringing.

Here is an example of the input, ground truth onset map and predicted onset map. The shown metrics are for the specific sample, which is of course slightly cherry picked.

![Onset map prediction](pictures/1.png)

---

## Architecture

### Overview

The model is a **UNet** with encoder-decoder structure and skip connections, inspired by MobileNet's depthwise-separable convolutions. Two variants are implemented:

- 1D UNet: Frequency bins are treated as channels, convolutions run along the time axis only.
- 2D UNet: Operates over both time and frequency jointly.

Both variants output a tensor of shape `(batch, time, notes, channels)` — a structured prediction for every time step and every piano key.

### Harmonic lowering

Since the CQT uses log-spaced frequency bins, the 2nd harmonic is always 12 bins up, the 3rd harmonic is 19 bins up, and so on.
Harmonic lowering exploits this by stacking shifted copies of the spectrogram into a multi-channel spectrogram.
This gives the model explicit access to each note's harmonic series as aligned channels, without having to learn the relationship from scratch.

The `HarmonicLowering` module in `models.py` is a loose implementation of [this paper](https://www.isca-archive.org/interspeech_2020/takeuchi20_interspeech.pdf).

### Convolutional blocks

To effectively train locally on my macbook, the models must be lightweight. The core convolutional building block is an inverted residual / separable convolution  with BatchNorm and ReLU, stolen form MobileNet. The upwards layers of the UNet use `repeat_interleave` upsampling (`AB -> AABB`) and the same convolutional blocks.

---

## Output format

The model predicts a **4-channel tensor** for every `(time_step, note)` pair:

| Channel | Meaning |
|---|---|
| `CONFIDENCE_MAX` | Peak confidence (used as the binary note-present signal) |
| `CONFIDENCE_SUM` | Total smoothed probability mass at this time/note location |
| `OFFSET` | Sub-step timing offset of the note onset (normalized) |
| `VELOCITY` | MIDI velocity (normalized to 0–1) |

A note is considered detected if `CONFIDENCE_MAX >= 0` after sigmoid.

> **Side note**: I included both `CONFIDENCE_MAX` and `CONFIDENCE_SUM` to provide some over-determination which will hopefully help with the decoding once I get to it.

---

## MIDI preprocessing

MIDI files are preprocessed into the 4-channel tensor format described above and cached to disk as `.npy` files for fast random access during training.

**Time resolution** is determined by the spectrogram hop length (e.g. ~50 ms at the default settings). Each note is smeared along the time axis using a Gaussian to prevent pumishing the model too hard for off-by-one errors.

> **Collisions**: if two identical notes start within the same time step, this is reflected by `CONFIDENCE_SUM`, but the one nearest to the center of the time bin overwrites all the remaining channels (`CONFIDENCE_MAX`, `OFFSET` and `VELOCITY`).

---

## Audio preprocessing

Audio is loaded as mono, resampled to 44,100 Hz, and transformed into a **Constant-Q Transform (CQT)** spectrogram. CQT uses log-spaced frequency bins aligned to musical semitones.

Current spectrogram settings:

| Parameter | Value |
|---|---|
| Sample rate | 44,100 Hz |
| Hop length | 2,048 samples (~46 ms) |
| Frequency bins | 116 (covering C0 and above) |
| Min frequency | 27.5 Hz (A0) |

---

## Loss

The loss combines several terms over all `(time, note)` locations:

- **Binary cross-entropy** on `CONFIDENCE_MAX` — the main note detection signal, up-weighted relative to other terms.
- **L2 regression** on `CONFIDENCE_SUM` — encourages calibrated confidence across nearby time steps.
- **L1 regression** on `OFFSET` and `VELOCITY` — weighted by ground-truth confidence so these only matter at real note locations.

Empty (no-note) locations are down-weighted relative to note locations to avoid the model collapsing to all-zeros.

---

## Getting started

### 1. Get the data

Download the [MAESTRO V3](https://magenta.withgoogle.com/datasets/maestro) dataset and unzip it to a path of your choice.

Then, set the `MAESTRO_DATASET_PATH` environment variable, either directly:

```bash
export MAESTRO_DATASET_PATH=/path/to/maestro-v3
```

Or by writing it to a `.env` file in the project root:

```
MAESTRO_DATASET_PATH=/path/to/maestro-v3
```

### 2. Install dependencies

If you don't have it, [get uv](https://docs.astral.sh/uv/getting-started/installation/). Then:

```bash
uv sync
```

### 3. Preprocess the MIDI files

```bash
uv run cache_midi.py
```

This preprocesses all MIDI files to `.npy` format and stores them under `cache/midi/`. Only needs to be run once.

### 4. Run training

```bash
uv run train.py --config=configs/baseline_v2.yaml
```

Checkpoints and TensorBoard logs are saved to `experiments/0/` (auto-incremented).

---

## Config

Training is configured via YAML. It is validated with Pydantic, so any config error will show up before training starts.
See `configs/baseline_v2.yaml` for a working example, or start in `train.py` to understand the schema definition.

---

## Dev log

### 23 May — YAML configs + MIDI dataloader optimization

Refactored training to use YAML config files. Profiled the dataloader and found the MIDI preprocessing was the bottleneck (slower than audio loading). Fixed by caching MIDI to `.npy` and using memory-mapped reads with binary search.

Downloaded the rest of the dataset and achieved ~82% validation F1.

### 22 May — Harmonic lowering + 2D convolutions

Switched from 1D to 2D convolutions and added the harmonic lowering module. Results improved significantly: **>85% F1 score** on 10% of the dataset, up from ~50% with the 1D model.

### 21 May — First training

The 1D UNet is training and detecting notes, achieving ~50% F1 score. Key problem: binary temporal representation is too strict, even small timing errors are penalised heavily.

Switched to smoothed Gaussian targets: the binary target score for a time bin is now computed according to a Gaussian centered at the exact onset time.

### 17-18 May - Setup, data loading, minimal model

CQT preprocessing, basic UNet skeleton, learning how to read MIDI, designing model output format, loss function, debugging tensor shapes.

---

## TODO

- Decoding back to MIDI (at least prototype)
- LR scheduler, dropout / regularization
- Data augmentation (pitch shift, time stretch, noise)
- Note duration prediction (this will require an architecture shift).
- Inference script + MIDI export
