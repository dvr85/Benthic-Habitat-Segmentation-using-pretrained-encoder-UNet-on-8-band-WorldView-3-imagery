"""Patch helpers: training random crops, validation grid, tile-and-stitch inference.

Tile-and-stitch lets us run a model that was trained on 256x256 patches over
a full-resolution test image without losing detail. We slide a window with
overlap, average overlapping predictions weighted by a 2D Hann window so
patch-edge pixels (which had less context) contribute less.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PatchPosition:
    image_index: int
    y: int
    x: int


def random_patch_origin(
    height: int, width: int, patch_size: int, rng: np.random.Generator
) -> tuple[int, int]:
    """Pick a random top-left (y, x) such that a [patch_size, patch_size] crop fits."""
    if height < patch_size or width < patch_size:
        raise ValueError(
            f"Image {height}x{width} smaller than patch {patch_size}x{patch_size}"
        )
    y = int(rng.integers(0, height - patch_size + 1))
    x = int(rng.integers(0, width - patch_size + 1))
    return y, x


def grid_patch_origins(
    height: int, width: int, patch_size: int, stride: int
) -> list[tuple[int, int]]:
    """Deterministic grid of (y, x) origins covering the image, with the last
    row/column shifted inward so the entire image is covered without overrun.
    """
    if height < patch_size or width < patch_size:
        raise ValueError(
            f"Image {height}x{width} smaller than patch {patch_size}x{patch_size}"
        )
    ys = list(range(0, height - patch_size + 1, stride))
    if ys[-1] != height - patch_size:
        ys.append(height - patch_size)
    xs = list(range(0, width - patch_size + 1, stride))
    if xs[-1] != width - patch_size:
        xs.append(width - patch_size)
    return [(y, x) for y in ys for x in xs]


def hann_window_2d(patch_size: int, device: torch.device | None = None) -> torch.Tensor:
    """2D Hann window via outer product. Tapers from 0 at edges to 1 in the
    middle so patch-edge pixels are downweighted during stitching.
    """
    w = torch.hann_window(patch_size, periodic=False)
    win2d = torch.outer(w, w)
    # Avoid exact zeros at the borders so edges still contribute *something*.
    win2d = win2d.clamp(min=1e-3)
    if device is not None:
        win2d = win2d.to(device)
    return win2d


@torch.no_grad()
def tile_and_stitch_predict(
    model: torch.nn.Module,
    image_chw: torch.Tensor,
    patch_size: int,
    stride: int,
    num_classes: int,
    device: torch.device,
    batch_size: int = 8,
) -> torch.Tensor:
    """Run patch-based inference over a normalized full-resolution image.

    Args:
        image_chw: (C, H, W) float tensor, ALREADY z-score normalized.
        patch_size: square patch size used during training.
        stride: pixel stride between adjacent patches; smaller = more overlap.
        num_classes: number of output channels of the model.
        device: where to run the model.
        batch_size: number of patches to forward per chunk.

    Returns:
        (H, W) int64 tensor of class indices, full resolution.
    """
    image_chw = image_chw.to(device)
    _, h, w = image_chw.shape

    # Pad up to multiples of patch_size on the right/bottom if needed; reflect-pad
    # avoids introducing fake "Background" pixels at the borders.
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    if pad_h or pad_w:
        image_chw = F.pad(image_chw, (0, pad_w, 0, pad_h), mode="reflect")
    H, W = image_chw.shape[1], image_chw.shape[2]

    origins = grid_patch_origins(H, W, patch_size, stride)
    win = hann_window_2d(patch_size, device=device)

    logit_sum = torch.zeros((num_classes, H, W), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((H, W), dtype=torch.float32, device=device)

    # Batch the patches for efficiency.
    for start in range(0, len(origins), batch_size):
        chunk = origins[start : start + batch_size]
        batch = torch.stack(
            [image_chw[:, y : y + patch_size, x : x + patch_size] for y, x in chunk],
            dim=0,
        )
        logits = model(batch)  # (B, C, P, P)
        probs = torch.softmax(logits, dim=1)
        # Weight every patch's class probabilities by the Hann window before summing.
        weighted = probs * win.unsqueeze(0).unsqueeze(0)
        for i, (y, x) in enumerate(chunk):
            logit_sum[:, y : y + patch_size, x : x + patch_size] += weighted[i]
            weight_sum[y : y + patch_size, x : x + patch_size] += win

    logit_sum /= weight_sum.clamp(min=1e-6)
    pred = logit_sum.argmax(dim=0)  # (H, W)
    pred = pred[:h, :w].contiguous()  # un-pad
    return pred.to("cpu", dtype=torch.int64)
