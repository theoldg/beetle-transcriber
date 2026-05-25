# Beetle Transcriber

In this project I attempt to create an audio-to-MIDI transcription model for piano, with near-zero prior knowledge about this problem. Beetles are deaf.

Trained on the [MAESTRO v3](https://magenta.withgoogle.com/datasets/maestro) dataset, the model takes a raw piano recording and outputs a structured prediction of which notes were played, when and at what velocity.

---

## Results

Currently, the model only predicts onsets and key press velocity. Decoding back from the continuous predictions to a discrete MIDI file is WIP.

After a few days of iteration, the v2 model (2D UNet with harmonic lowering) reaches **>85% F1 score** at the current time resolution on the MAESTRO validation split.

Here is an example of the input, ground truth onset map and predicted onset map. The shown metrics are for the specific sample.

![Onset map prediction](pictures/1.png)

---

## Architecture

### Overview

The model is a **UNet** with encoder-decoder structure and skip connections, inspired by MobileNet's depthwise-separable convolutions. Two variants are implemented:

- **v1** — 1D UNet. Frequency bins are treated as channels, convolutions run along the time axis only.
- **v2** — 2D UNet. Operates over both time and frequency jointly.

Both variants output a tensor of shape `(batch, time, notes, channels)` — a structured prediction for every time step and every piano key.

### Harmonic lowering

Since the CQT uses log-spaced frequency bins, the 2nd harmonic is always 12 bins up, the 3rd harmonic is 19 bins up, and so on.

`HarmonicLowering` exploits this by stacking shifted copies of the spectrogram into a multi-channel spectrogram. This gives the model explicit access to each note's harmonic series as aligned channels, without having to learn the relationship from scratch.

### Convolutional blocks

To effectively train locally on my macbook, the models must be lightweight. Both models use the same `ConvLayer` building block: a 3-step expand → depthwise → project pattern (stolen from MobileNetV3 inverted residuals), with BatchNorm and ReLU throughout.

The decoder uses `UpLayer`, which combines a convolution with skip connection concatenation followed by `repeat_interleave` upsampling.

---

## Output format

The model predicts a **4-channel tensor** for every `(time_step, note)` pair:

| Channel | Meaning |
|---|---|
| `CONFIDENCE_SUM` | Total smoothed probability mass at this time/note location |
| `CONFIDENCE_MAX` | Peak confidence (used as the binary note-present signal) |
| `OFFSET` | Sub-step timing offset of the note onset (normalized) |
| `VELOCITY` | MIDI velocity (normalized to 0–1) |

A note is considered detected if `CONFIDENCE_MAX >= 0` after sigmoid.

---

## MIDI preprocessing

MIDI files are preprocessed into the 4-channel tensor format described above and cached to disk as `.npy` files for fast random access during training.

**Time resolution** is determined by the spectrogram hop length (e.g. ~50 ms at the default settings). Each note is smeared along the time axis using a Gaussian to prevent pumishing the model too hard for off-by-one errors.

> **Collision note**: if two identical notes start within the same time step, only the later one is kept.

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

### 1. Download MAESTRO

Download the [MAESTRO V3](https://magenta.withgoogle.com/datasets/maestro) dataset and unzip it to a path of your choice.

### 2. Set the dataset path

```bash
export MAESTRO_DATASET_PATH=/path/to/maestro-v3
```

Or write it to a `.env` file in the project root:

```
MAESTRO_DATASET_PATH=/path/to/maestro-v3
```

### 3. Install dependencies

```bash
uv sync
```

### 4. Cache the MIDI files

```bash
uv run cache_midi.py
```

This preprocesses all MIDI files to `.npy` format and stores them under `cache/midi/`. Only needs to be run once.

### 5. Run training

```bash
uv run train.py --config=configs/baseline_v2.yaml
```

Checkpoints and TensorBoard logs are saved to `experiments/0/` (auto-incremented).

---

## Config

Training is fully config-driven via YAML. See `configs/baseline_v2.yaml` for a working example. 

---

## Dev log

### 23 May — YAML configs + MIDI dataloader optimization

Refactored training to use YAML config files. Profiled the dataloader and found the MIDI preprocessing was the bottleneck (slower than audio loading). Fixed by caching MIDI to `.npy` and using memory-mapped reads with binary search.

### 22 May — Harmonic lowering + 2D convolutions

Switched from 1D to 2D convolutions and added the harmonic lowering module. Results improved significantly: **>85% F1 score**, up from ~50% with the 1D model.

### 21 May — First working model

The 1D UNet is training and detecting notes, achieving ~50% F1 score. Some key problems identified:

- Binary piano roll targets are too strict — even small timing errors are penalised heavily.
- No velocity or sub-step offset prediction.

Solved both by switching to smoothed Gaussian targets with explicit offset and velocity channels, and by rebalancing the loss between empty and non-empty locations.

### Initial setup

Dataset loading, CQT preprocessing, basic UNet skeleton, Lightning training loop.

---

## TODO

- Decoding back to MIDI (at least prototype)
- LR scheduler, dropout / regularization
- Data augmentation (pitch shift, time stretch, noise)
- Note duration prediction (this will require an architecture shift).
- Inference script + MIDI export
