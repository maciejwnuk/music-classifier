"""Shared configuration module"""

from pathlib import Path

# Directories
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

AUDIO_DIR = DATA_DIR / "audio"
SORTED_DIR = DATA_DIR / "sorted"
SPECTROGRAMS_DIR = DATA_DIR / "spectrograms"

# Classes
CATEGORIES = [
    "aggressive",
    "atmospheric",
    "chill",
    "dark",
    "groovy",
    "hype",
    "melancholic",
    "uplifting",
]

NUM_CLASSES = len(CATEGORIES)

# Hyperparameters
SAMPLE_RATE      = 22050               # Audio target sample rate
SEGMENT_DURATION = 5.0                 # Audio segment duration in seconds
FFT_POINTS       = 1024                # FFT resolution
FFT_HOP          = 512                 # FFT hop size
IMG_SIZE         = (224, 224)          # ResNet Input size

BATCH_SIZE    = 32                     # Batch size
NUM_EPOCHS    = 30                     # Number of runs
LEARNING_RATE = 1e-4                   # Learning rate (loss parameter)
VAL_SPLIT     = 0.2                    # Fraction of dataset to validate
