"""Shared configuration module"""

import torch
from pathlib import Path

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

# Directories
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

AUDIO_DIR = DATA_DIR / "audio"
SORTED_DIR = DATA_DIR / "sorted"
SPECTROGRAMS_DIR = DATA_DIR / "spectrograms"

# Classes
CATEGORIES = [
    "aggressive",
    "chill",
    "groovy",
    "hype",
]

NUM_CLASSES = len(CATEGORIES)

# Hyperparameters
DYNAMIC_RANGE    = 70                  # Range of dB to cover
SAMPLE_RATE      = 22050               # Audio target sample rate
SEGMENT_DURATION = 5.0                 # Audio segment duration in seconds
FFT_POINTS       = 512                 # FFT resolution
FFT_HOP          = 256                 # FFT hop size
IMG_SIZE         = (224, 224)          # ResNet Input size

BATCH_SIZE    = 32                     # Batch size
NUM_EPOCHS    = 50                     # Number of runs
LEARNING_RATE = 1e-4                   # Learning rate (loss parameter)
TEST_SPLIT    = 0.2                    # Fraction of dataset to validate
