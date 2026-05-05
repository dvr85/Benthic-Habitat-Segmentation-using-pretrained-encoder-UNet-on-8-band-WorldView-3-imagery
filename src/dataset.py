"""Patch-based BenthicDataset for 8-band WV3 imagery.

Two modes:
  * mode='train': returns `patches_per_epoch` random crops drawn from the
    cached training images. Optional flip / 90-deg-rotation augmentation
    keeps image and mask synchronized.
  * mode='val': returns a deterministic, exhaustive grid of non-overlapping
    crops from the cached validation images. Same set every epoch so the
    val metric is comparable across checkpoints.

Splits are by image stem so a given image's pixels are never in both train
and val. Splitting is deterministic given `seed`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from src.patches import grid_patch_origins, random_patch_origin


def _list_tifs(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() == ".tif")


def split_image_stems(
    images_dir: Path, val_split: float, seed: int
) -> tuple[list[str], list[str]]:
    """Deterministically split image stems into (train, val) by val_split."""
    stems = [p.stem for p in _list_tifs(images_dir)]
    n_val = max(1, int(round(len(stems) * val_split)))
    rng = random.Random(seed)
    shuffled = stems.copy()
    rng.shuffle(shuffled)
    val_stems = sorted(shuffled[:n_val])
    train_stems = sorted(shuffled[n_val:])
    return train_stems, val_stems


class BenthicDataset(Dataset):
    def __init__(
        self,
        images_dir: str | Path,
        annotations_dir: str | Path,
        band_stats_path: str | Path,
        stems: list[str],
        mode: Literal["train", "val"],
        patch_size: int = 256,
        patches_per_epoch: int = 400,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        if mode not in {"train", "val"}:
            raise ValueError(f"mode must be 'train' or 'val', got {mode}")
        if mode == "val" and augment:
            raise ValueError("augment=True is only valid for mode='train'")

        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.mode = mode
        self.patch_size = patch_size
        self.patches_per_epoch = patches_per_epoch
        self.augment = augment

        with open(band_stats_path) as f:
            stats = json.load(f)
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(stats["std"], dtype=torch.float32).view(-1, 1, 1)

        # Cache full-resolution images and masks in RAM. 9 small images * 8 bands
        # is well under 100 MB so this is free.
        self.images: list[torch.Tensor] = []
        self.masks: list[torch.Tensor] = []
        self.stems: list[str] = []
        for stem in stems:
            img_path = self.images_dir / f"{stem}.tif"
            ann_path = self.annotations_dir / f"{stem}.tif"
            if not img_path.exists() or not ann_path.exists():
                raise FileNotFoundError(
                    f"Missing image or annotation for stem '{stem}'"
                )
            with rasterio.open(img_path) as src:
                arr = src.read().astype(np.float32)
            with rasterio.open(ann_path) as src:
                mask = src.read(1).astype(np.int64)
            self.images.append(torch.from_numpy(arr))
            self.masks.append(torch.from_numpy(mask))
            self.stems.append(stem)

        if not self.images:
            raise ValueError(f"No images loaded for mode={mode}")

        # Reject any image smaller than the patch size.
        for stem, img in zip(self.stems, self.images):
            _, h, w = img.shape
            if h < patch_size or w < patch_size:
                raise ValueError(
                    f"Image {stem} is {h}x{w}, smaller than patch {patch_size}"
                )

        if mode == "train":
            # Per-instance RNG so workers don't collide; seeded for reproducibility.
            self._rng = np.random.default_rng(seed)
            self._val_grid: list[tuple[int, int, int]] = []
        else:
            # Build a deterministic grid: list of (image_idx, y, x).
            self._rng = None
            grid: list[tuple[int, int, int]] = []
            for img_idx, img in enumerate(self.images):
                _, h, w = img.shape
                origins = grid_patch_origins(h, w, patch_size, stride=patch_size)
                for y, x in origins:
                    grid.append((img_idx, y, x))
            self._val_grid = grid

    def __len__(self) -> int:
        if self.mode == "train":
            return self.patches_per_epoch
        return len(self._val_grid)

    def _augment(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            image = torch.flip(image, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        if random.random() < 0.5:
            image = torch.flip(image, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        k = random.randint(0, 3)
        if k:
            image = torch.rot90(image, k=k, dims=[-2, -1])
            mask = torch.rot90(mask, k=k, dims=[-2, -1])
        return image, mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "train":
            assert self._rng is not None
            img_idx = int(self._rng.integers(0, len(self.images)))
            img = self.images[img_idx]
            mask = self.masks[img_idx]
            _, h, w = img.shape
            y, x = random_patch_origin(h, w, self.patch_size, self._rng)
        else:
            img_idx, y, x = self._val_grid[idx]
            img = self.images[img_idx]
            mask = self.masks[img_idx]

        ps = self.patch_size
        img_patch = img[:, y : y + ps, x : x + ps].clone()
        mask_patch = mask[y : y + ps, x : x + ps].clone()

        # Z-score AFTER cropping so we don't keep a normalized full image around.
        img_patch = (img_patch - self.mean) / self.std

        if self.augment:
            img_patch, mask_patch = self._augment(img_patch, mask_patch)

        return img_patch, mask_patch
