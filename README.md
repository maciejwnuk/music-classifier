# Music classifier by mood

Music classifier using a custom convolutional neural network (CNN) on mel spectrograms, categorizing tracks into four mood classes.

## Description

The pipeline converts WAV audio files into grayscale mel spectrogram images, then trains a lightweight 4-block CNN from scratch to classify them by mood. Songs are split into fixed-length segments; at inference time, per-segment predictions are averaged to produce a single mood label per track.

### Features

- **Mood classes**: `aggressive`, `chill`, `groovy`, `hype`
- **Spectrogram generation** — configurable FFT resolution, hop size, dynamic range, and segment duration
- **Song-level train/test split** — segments from the same track never leak across sets
- **Data augmentation** — time/frequency shifting and SpecAugment-style frequency masking
- **Inference** — per-segment softmax probabilities averaged into a single prediction per track

### Pipeline

- WAV files → spectrograms.py → PNG spectrograms
- PNG spectrograms → train.py → model.pt
- model.pt → predict.py → mood label

## Setup

Requires Python ≥ 3.14 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

## Usage

### 1. Prepare data

Place WAV files into category subdirectories under `data/sorted/`:

```
data/sorted/
├── aggressive/
├── chill/
├── groovy/
└── hype/
```

### 2. Generate spectrograms

```sh
uv run spectrograms.py
```

Spectrogram images are saved to `data/spectrograms/<category>/`.

### 3. Train

```sh
uv run train.py
```

Saves the best checkpoint to `model.pt` and a confusion matrix plot to `confusion_matrix.png`.

### 4. Predict

```
uv run predict.py [OPTIONS] <FILES...>
```

| Argument | Description |
|---|---|
| `FILES` | One or more paths to WAV files |

| Flag | Default | Description |
|---|---|---|
| `--model` | `model.pt` | Path to model checkpoint |

#### Example

```sh
uv run predict.py "PEEKABOO - SPUNKY.wav"
```

## Configuration

All hyperparameters are in [`config.py`](config.py):

| Parameter | Default | Description |
|---|---|---|
| `DYNAMIC_RANGE` | `70` | Range of dB to cover in spectrograms |
| `SAMPLE_RATE` | `22050` | Audio target sample rate |
| `SEGMENT_DURATION` | `5.0` | Segment length in seconds |
| `FFT_POINTS` | `1024` | FFT resolution (window size) |
| `FFT_HOP` | `256` | FFT hop size |
| `N_MELS` | `128` | Number of Mel frequency bins |
| `IMG_SIZE` | `224 × 224` | Model input size |
| `BATCH_SIZE` | `32` | Training batch size |
| `NUM_EPOCHS` | `30` | Training epochs |
| `LEARNING_RATE` | `1e-3` | Base learning rate |
| `TEST_SPLIT` | `0.2` | Fraction of songs held out for validation |

## License

[MIT](LICENSE.md) — Copyright (c) 2026 Maciej Wnuk
