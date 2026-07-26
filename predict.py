
import sys
import argparse

from pathlib import Path

import torch
from torch.nn import functional as F

from config import get_device, BASE_DIR, CATEGORIES

import resnet

from train import get_transform
from spectrograms import wav_to_segments

def main():
    parser = argparse.ArgumentParser(
        description = "Run inference on WAV files using trained mood model"
    )

    parser.add_argument(
        "files",
        nargs = "*",
        type = Path,
        help = "Paths to WAV files"
    )

    parser.add_argument(
        "--model",
        type = Path,
        default = BASE_DIR / "model.pt",
        help = "Path to model checkpoint"
    )

    args = parser.parse_args()

    files = []

    if args.files:
        files.extend(args.files)

    if not files:
        parser.print_help()

        sys.exit(1)

    device = get_device()

    print(f"Using device: {device}")

    if not args.model.exists():
        print(f"Error: Model not found at {args.model}")

        sys.exit(1)

    print(f"Loading model from {args.model}...")

    checkpoint = torch.load(args.model, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "classes" in checkpoint:
        class_to_idx = checkpoint["classes"]
        categories = [None] * len(class_to_idx)
        for name, idx in class_to_idx.items():
            categories[idx] = name
    else:
        categories = CATEGORIES

    model = resnet.model

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    transform = get_transform()

    for path in files:
        print(f"Analyzing {path.name}...")

        segments = wav_to_segments(path)

        tensors = [transform(segment) for segment in segments]

        tensors = torch.stack(tensors) # pyright: ignore[reportArgumentType]
        tensors = tensors.to(device)

        with torch.no_grad():
            outputs = model(tensors)
            probs = F.softmax(outputs, dim=1)

        avg_probs = probs.mean(dim=0)

        top_probs, top_indices = torch.topk(avg_probs, 3)

        print(f"{'Mood':<12} | {'Confidence'}")
        print("-" * 30)

        for prob, idx in zip(top_probs, top_indices):
            print(f"{categories[idx]:<12} | {prob.item() * 100:.2f}%")

if __name__ == '__main__':
    main()
