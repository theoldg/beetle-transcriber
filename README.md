# `beetle-transcriber`

In this project I attempt to create an audio-to-MIDI ML model for music transcription, without any prior knowledge on this topic. Beetles are deaf.

## Data

We use the [Maestro V3](https://magenta.withgoogle.com/datasets/maestro) dataset.

## MIDI preprocessing

The MIDI is preprocessed into a "ML friendly" format as follows. First, a time resolution is chosen (e.g. 50 ms).
Then, the file is represented as a 3d float tensor with shape `(num time steps, num notes, channels)`. Each note 
is written to a given "slot" determined by time and pitch, and has 4 associated channels.

For example, at time resolution 50 ms, a C4 played from `T_start = 10.003` to `T_end = 10.5003` with velocity 57 would 
be written at index `[200, 60]` as `[1, 0.003, 0.5, 57]` (pre-normalization), because:

- `200` is the time step of the start of the note (50 ms * 200 = 10 s)
- `60` is the MIDI note ID for C4
- `1` represents the model confidence that a note has started at this time step. Most time/note locations contain a 0.
- `0.003` is the offset of the note start within the time step
- `0.5` is the duration in seconds
- `57` is the velocity.

These values are then normalized, check the code for details.

## Audio preprocessing

Audio is preprocessed as a log-mel spectrogram.

## Model

A 1d UNet / MobileNet-ish custom job. Mel bins are treated as channels. The output is reshaped from `(..., time, D)`
into `(..., time, notes, channels)`.


