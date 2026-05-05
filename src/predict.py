"""Single-image inference helper.

Runs tile-and-stitch over the full-resolution image so the output is at the
input image's native resolution (no resize-then-predict downscaling).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Must precede torch import so MPS unsupported ops fall back to CPU silently.
if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import rasterio
import torch
from PIL import Image

from src.model import build_model
from src.patches import tile_and_stitch_predict
from src.utils import decode_segmap, get_device

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_band_stats(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with open(path) as f:
        stats = json.load(f)
    mean = torch.tensor(stats["mean"], dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32).view(-1, 1, 1)
    return mean, std


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model_cfg = cfg["model"]
    model = build_model(
        encoder_name=model_cfg["encoder_name"],
        in_channels=model_cfg["in_channels"],
        classes=model_cfg["num_classes"],
        encoder_weights=None,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def normalize_full_image(
    image_path: Path, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Read an 8-band TIFF and apply z-score; returns (C, H, W) float32 tensor."""
    with rasterio.open(image_path) as src:
        arr = src.read().astype(np.float32)
    tensor = torch.from_numpy(arr)
    return (tensor - mean) / std


def predict_full_resolution(
    model: torch.nn.Module,
    image_chw: torch.Tensor,
    cfg: dict,
    device: torch.device,
) -> np.ndarray:
    """Run tile-and-stitch with the same patch_size + stride used during training."""
    patch_size = cfg["data"]["patch_size"]
    stride = cfg["data"].get("inference_stride", patch_size // 2)
    return tile_and_stitch_predict(
        model=model,
        image_chw=image_chw,
        patch_size=patch_size,
        stride=stride,
        num_classes=cfg["model"]["num_classes"],
        device=device,
    ).numpy().astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--band_stats", default=PROJECT_ROOT / "configs" / "band_stats.json", type=Path
    )
    args = parser.parse_args()

    device = get_device()
    model, cfg = load_checkpoint(args.checkpoint, device)
    mean, std = load_band_stats(args.band_stats)

    normalized = normalize_full_image(args.image, mean, std)
    pred = predict_full_resolution(model, normalized, cfg, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(decode_segmap(pred)).save(args.output)
    print(f"wrote {args.output}  (shape: {pred.shape})")


if __name__ == "__main__":
    main()
