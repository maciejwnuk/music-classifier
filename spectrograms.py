from pathlib import Path
from typing import List
from maad import sound, util
from PIL import Image
import numpy as np
from tqdm import tqdm

from config import (
    SAMPLE_RATE, FFT_POINTS, FFT_HOP,
    SEGMENT_DURATION, IMG_SIZE,
    SPECTROGRAMS_DIR, SORTED_DIR,
    CATEGORIES
)

def wav_to_segments(filepath: Path) -> List[Image.Image]:
    left,  fs_left  = sound.load(filepath, channel = "left", detrend = False)
    right, fs_right = sound.load(filepath, channel = "right", detrend = False)

    if fs_left != fs_right:
        raise Exception(
            "Sampling rate differs in left and right channels"
            " "
            "when processing file: " + str(input)
        )

    fs = fs_left

    # Let's go mono
    signal = (left + right) / 2

    if fs != SAMPLE_RATE:
        signal = sound.resample(signal, fs, SAMPLE_RATE)
        fs = SAMPLE_RATE

    fft, time, _freq, _loc = sound.spectrogram(
        signal, fs,
        nperseg = FFT_POINTS,
        noverlap = FFT_POINTS - FFT_HOP
    )

    fft_db = util.power2dB(fft)

    dt = time[1] - time[0]
    frames = int(SEGMENT_DURATION / dt)

    images = []

    for i in range(0, len(time), frames):
        segment = fft_db[:, i:(i + frames)]

        # Discard the segment if it's shorter than SEGMENT_DURATION
        if segment.shape[1] < frames:
            continue

        # Discard segments that are mostly silence
        if np.mean(segment) < -110.0:
            continue

        # Normalize to save image
        min = np.min(segment)
        max = np.max(segment)

        if max - min == 0:
            continue

        segment = np.flipud(
            (segment - min) / (max - min) * 255
        ).astype(np.uint8)

        img = Image.fromarray(segment, "L").resize(
            IMG_SIZE,
            resample = Image.Resampling.LANCZOS
        )

        images.append(img)

    return images

def main():
    SPECTROGRAMS_DIR.mkdir(
        parents  = True,
        exist_ok = True
    )

    tracks_processed = 0
    category_count = { cat: 0 for cat in CATEGORIES }

    for category in CATEGORIES:
        in_dir = SORTED_DIR / category
        out_dir = SPECTROGRAMS_DIR / category

        if not in_dir.exists():
            print(f"Directory not found, skipping: {in_dir}")
            continue

        out_dir.mkdir(
            parents  = True,
            exist_ok = True
        )

        files = list(in_dir.glob("*.wav"))

        for path in tqdm(
            files,
            desc = f"Processing: {category}"
        ):
            segments = wav_to_segments(path)

            for i, segment in enumerate(segments):
                out_path = out_dir / f"{path.stem}_seg{i:03d}.png"

                segment.save(out_path)

            tracks_processed += 1

            category_count[category] += len(segments)

    print(f"\nTotal tracks processed: {tracks_processed}")
    print(f"Total spectrograms generated: {sum(category_count.values())}")
    print("Per-category counts:")

    for cat, count in category_count.items():
        print(f"- {cat}: {count} spectrograms")

if __name__ == "__main__":
    main()
