"""Evaluate a trained checkpoint at full resolution via tile-and-stitch.

Run:
    python -m src.evaluate                              # uses defaults from config.yaml
    python -m src.evaluate eval.test_set=Agana          # Agana Bay only
    python -m src.evaluate eval.test_set=Manell         # Manell-Geus only
    python -m src.evaluate eval.checkpoint=path/to.pth
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Must precede torch import so MPS unsupported ops fall back to CPU silently.
if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import hydra
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from PIL import Image

from src.metrics import SegmentationMetrics
from src.patches import tile_and_stitch_predict
from src.utils import CLASS_NAMES, decode_segmap, encode_rgb_to_class, get_device

log = logging.getLogger(__name__)


def load_band_stats(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with open(path) as f:
        stats = json.load(f)
    mean = torch.tensor(stats["mean"], dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32).view(-1, 1, 1)
    return mean, std


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    from src.model import build_model

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
) -> tuple[torch.Tensor, tuple[int, int]]:
    with rasterio.open(image_path) as src:
        arr = src.read().astype(np.float32)
    h, w = arr.shape[1], arr.shape[2]
    tensor = torch.from_numpy(arr)
    tensor = (tensor - mean) / std
    return tensor, (h, w)


def rgb_composite(image_path: Path) -> np.ndarray:
    with rasterio.open(image_path) as src:
        arr = src.read(out_dtype="float32")
    bands = arr[[4, 2, 1], ...]  # WV3 bands 5/3/2 -> R/G/B-ish composite
    out = np.zeros((bands.shape[1], bands.shape[2], 3), dtype=np.float32)
    for i in range(3):
        b = bands[i]
        lo, hi = np.percentile(b, [2, 98])
        out[..., i] = np.clip((b - lo) / max(hi - lo, 1e-6), 0, 1)
    return out


def save_error_map(
    output_path: Path, image_path: Path, gt: np.ndarray, pred: np.ndarray, stem: str
) -> None:
    rgb = rgb_composite(image_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb)
    axes[0].set_title(f"{stem}\nRGB composite (bands 5/3/2)")
    axes[0].axis("off")
    axes[1].imshow(decode_segmap(gt))
    axes[1].set_title("Ground truth")
    axes[1].axis("off")
    axes[2].imshow(decode_segmap(pred))
    axes[2].set_title("Prediction (tile-and-stitch)")
    axes[2].axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(cm: np.ndarray, output_path: Path, title: str) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            v = norm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def evaluate_test_set(
    test_id: str,
    model: torch.nn.Module,
    num_classes: int,
    patch_size: int,
    inference_stride: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    project_root: Path,
    results_dir: Path,
) -> tuple[dict, np.ndarray, list[dict]]:
    images_dir = project_root / "data" / "test_data" / test_id / "images"
    annotations_dir = project_root / "data" / "test_data" / test_id / "annotations"
    preds_dir = results_dir / f"predictions_{test_id}"
    error_maps_dir = results_dir / "error_maps" / test_id
    preds_dir.mkdir(parents=True, exist_ok=True)

    metrics = SegmentationMetrics(num_classes)
    per_image: list[dict] = []

    tif_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() == ".tif")
    for img_path in tif_paths:
        stem = img_path.stem
        ann_path = annotations_dir / f"{stem}.png"
        if not ann_path.exists():
            raise FileNotFoundError(f"Missing GT mask: {ann_path}")

        normalized, (h, w) = normalize_full_image(img_path, mean, std)
        pred = tile_and_stitch_predict(
            model=model,
            image_chw=normalized,
            patch_size=patch_size,
            stride=inference_stride,
            num_classes=num_classes,
            device=device,
        ).numpy().astype(np.int64)

        Image.fromarray(decode_segmap(pred)).save(preds_dir / f"{stem}.png")

        gt_rgb = np.array(Image.open(ann_path).convert("RGB"))
        gt = encode_rgb_to_class(gt_rgb)
        if gt.shape != pred.shape:
            # Annotation might be a slightly different size; resize via nearest.
            gt = np.array(
                Image.fromarray(gt.astype(np.uint8)).resize(
                    (pred.shape[1], pred.shape[0]), Image.NEAREST
                )
            ).astype(np.int64)

        # Per-image metrics for error analysis.
        per_image_metrics = SegmentationMetrics(num_classes)
        per_image_metrics.update(
            torch.eye(num_classes)[pred].permute(2, 0, 1).unsqueeze(0).float(),
            torch.from_numpy(gt).unsqueeze(0),
        )
        im_stats = per_image_metrics.compute()
        per_image.append(
            {
                "stem": stem,
                "macro_dice": im_stats["macro_dice"],
                "macro_iou": im_stats["macro_iou"],
                "pixel_accuracy": im_stats["pixel_accuracy"],
                "per_class_dice": im_stats["per_class_dice"],
            }
        )

        save_error_map(error_maps_dir / f"{stem}.png", img_path, gt, pred, stem)

        logits = torch.eye(num_classes)[pred].permute(2, 0, 1).unsqueeze(0).float()
        metrics.update(logits, torch.from_numpy(gt).unsqueeze(0))

    stats = metrics.compute()
    cm = np.array(stats["confusion_matrix"], dtype=np.int64)
    save_confusion_matrix(
        cm,
        results_dir / f"confusion_matrix_{test_id}.png",
        title=f"Test set {test_id} — normalized confusion matrix",
    )
    return stats, cm, per_image


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = get_device()
    log.info("device: %s", device)

    project_root = Path(to_absolute_path("."))
    ckpt_path = Path(to_absolute_path(cfg.eval.checkpoint))
    band_stats_path = Path(to_absolute_path(cfg.data.band_stats_path))

    model, ckpt_cfg = load_checkpoint(ckpt_path, device)
    num_classes = ckpt_cfg["model"]["num_classes"]
    patch_size = ckpt_cfg["data"]["patch_size"]
    inference_stride = ckpt_cfg["data"].get("inference_stride", cfg.data.inference_stride)
    mean, std = load_band_stats(band_stats_path)

    results_dir = project_root / cfg.results_dir / cfg.run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    test_ids = ["Agana", "Manell"] if cfg.eval.test_set == "both" else [str(cfg.eval.test_set)]

    all_metrics: dict = {}
    combined = SegmentationMetrics(num_classes)
    per_image_rows: list[dict] = []

    for tid in test_ids:
        stats, cm, per_image = evaluate_test_set(
            test_id=tid,
            model=model,
            num_classes=num_classes,
            patch_size=patch_size,
            inference_stride=inference_stride,
            mean=mean,
            std=std,
            device=device,
            project_root=project_root,
            results_dir=results_dir,
        )
        all_metrics[f"test_set_{tid}"] = {
            "pixel_accuracy": stats["pixel_accuracy"],
            "macro_iou": stats["macro_iou"],
            "macro_dice": stats["macro_dice"],
            "per_class_iou": dict(zip(CLASS_NAMES, stats["per_class_iou"])),
            "per_class_dice": dict(zip(CLASS_NAMES, stats["per_class_dice"])),
            "confusion_matrix": cm.tolist(),
        }
        combined.confusion += torch.tensor(cm, dtype=torch.int64)
        for row in per_image:
            per_image_rows.append({"test_set": tid, **row})
        log.info(
            "[test_set %s] macro_dice=%.4f  macro_iou=%.4f  pixel_acc=%.4f",
            tid, stats["macro_dice"], stats["macro_iou"], stats["pixel_accuracy"],
        )

    if cfg.eval.test_set == "both":
        comb_stats = combined.compute()
        comb_cm = np.array(comb_stats["confusion_matrix"], dtype=np.int64)
        save_confusion_matrix(
            comb_cm,
            results_dir / "confusion_matrix_combined.png",
            title="Combined test sets — normalized confusion matrix",
        )
        all_metrics["combined"] = {
            "pixel_accuracy": comb_stats["pixel_accuracy"],
            "macro_iou": comb_stats["macro_iou"],
            "macro_dice": comb_stats["macro_dice"],
            "per_class_iou": dict(zip(CLASS_NAMES, comb_stats["per_class_iou"])),
            "per_class_dice": dict(zip(CLASS_NAMES, comb_stats["per_class_dice"])),
            "confusion_matrix": comb_cm.tolist(),
        }
        log.info(
            "[combined] macro_dice=%.4f  macro_iou=%.4f  pixel_acc=%.4f",
            comb_stats["macro_dice"], comb_stats["macro_iou"], comb_stats["pixel_accuracy"],
        )

    with (results_dir / "metrics.json").open("w") as f:
        json.dump(all_metrics, f, indent=2)
    with (results_dir / "per_image_metrics.json").open("w") as f:
        json.dump(per_image_rows, f, indent=2)
    log.info("wrote %s/metrics.json", results_dir.relative_to(project_root))


if __name__ == "__main__":
    main()
