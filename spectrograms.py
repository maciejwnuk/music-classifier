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
        segment = np.flipud(np.clip(
            (segment + 100.0) * 2.55, 0, 255
        )).astype(np.uint8)

        img = Image.fromarray(segment).resize(
            IMG_SIZE,
            resample = Image.Resampling.BILINEAR
        )

        images.append(img)

    return images

def main():
    SPECTROGRAMS_DIR.mkdir(
        parents  = True,
        exist_ok = True
    )

    tracks_processed = 0
    spectrograms_generated = 0

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
            spectrograms_generated += segments_count

            category_count[category] += segments_count

    print("\n--- Summary ---")
    print(f"Total tracks processed: {tracks_processed}")
    print(f"Total spectrograms generated: {spectrograms_generated}")
    print("Per-category counts:")

    for cat, count in category_count.items():
        print(f"\t{cat}: {count} spectrograms")

if __name__ == "__main__":
    main()
