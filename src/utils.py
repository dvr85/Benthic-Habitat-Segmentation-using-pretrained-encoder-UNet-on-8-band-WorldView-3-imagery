"""Shared helpers: device selection, class palette, segmap decoding."""

from __future__ import annotations

import numpy as np
import torch

CLASS_NAMES = [
    "Background",
    "Seagrass",
    "Coral",
    "Macroalgae",
    "Sand",
    "Land",
    "Ocean",
]

# Palette used by the dataset's colored/ annotations. Derived directly from
# raw_labeled_data by sampling each class index in the single-channel .tif
# against the matching colored .png. Test-set annotations use the same palette.
CLASS_COLORS = np.array(
    [
        [0, 0, 0],         # Background
        [134, 164, 117],   # Seagrass
        [255, 127, 80],    # Coral
        [101, 138, 42],    # Macroalgae
        [203, 189, 147],   # Sand
        [139, 98, 76],     # Land
        [127, 205, 255],   # Ocean
    ],
    dtype=np.uint8,
)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def decode_segmap(pred_mask: np.ndarray) -> np.ndarray:
    """Integer class map -> uint8 HxWx3 RGB via CLASS_COLORS."""
    rgb = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    for i, color in enumerate(CLASS_COLORS):
        rgb[pred_mask == i] = color
    return rgb


def encode_rgb_to_class(rgb: np.ndarray) -> np.ndarray:
    """Uint8 HxWx3 RGB mask -> HxW int64 class map using CLASS_COLORS.

    Raises if the image contains any RGB triplet not in the palette.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected uint8 HxWx3 RGB, got {rgb.shape} {rgb.dtype}")
    out = np.full(rgb.shape[:2], -1, dtype=np.int64)
    for i, color in enumerate(CLASS_COLORS):
        out[np.all(rgb == color, axis=-1)] = i
    if (out < 0).any():
        bad = rgb[out < 0]
        sample = tuple(bad[0].tolist())
        raise ValueError(f"RGB triplet {sample} not in CLASS_COLORS palette")
    return out
