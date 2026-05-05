"""Compute per-band mean and std over the labeled training GeoTIFFs.

Writes results to configs/band_stats.json. The 9 WV3 images fit easily in
memory, so we load them all and compute stats directly over concatenated
per-band pixel arrays (weighted naturally by image size).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "data" / "raw_labeled_data" / "images"
OUTPUT_PATH = PROJECT_ROOT / "configs" / "band_stats.json"


def list_tif_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() == ".tif")


def main() -> None:
    tif_paths = list_tif_files(IMAGES_DIR)
    if not tif_paths:
        raise SystemExit(f"No .tif files found under {IMAGES_DIR}")

    per_band_pixels: list[np.ndarray] = [[] for _ in range(8)]
    for path in tif_paths:
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float64)
        if arr.shape[0] != 8:
            raise ValueError(f"Expected 8 bands, got {arr.shape[0]} in {path.name}")
        for b in range(8):
            per_band_pixels[b].append(arr[b].ravel())

    means = [float(np.concatenate(pixels).mean()) for pixels in per_band_pixels]
    stds = [float(np.concatenate(pixels).std()) for pixels in per_band_pixels]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mean": means,
        "std": stds,
        "source": str(IMAGES_DIR.relative_to(PROJECT_ROOT)),
        "num_images": len(tif_paths),
    }
    with OUTPUT_PATH.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    for i, (m, s) in enumerate(zip(means, stds)):
        print(f"  band {i + 1}: mean={m:.2f}  std={s:.2f}")


if __name__ == "__main__":
    main()
