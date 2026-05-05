"""Single-stage training over patch-based BenthicDataset.

Run with Hydra. Override anything from the CLI:
    python -m src.train training.epochs=10 model.encoder_name=resnet34
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import sys
from pathlib import Path

# Must precede torch import so MPS unsupported ops fall back to CPU.
if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.dataset import BenthicDataset, split_image_stems
from src.losses import HybridLoss
from src.metrics import SegmentationMetrics
from src.model import build_model
from src.utils import CLASS_NAMES, get_device

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: HybridLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    num_classes: int,
) -> tuple[float, dict]:
    
    training = optimizer is not None
    model.train(training)
    metrics = SegmentationMetrics(num_classes)
    total_loss = 0.0
    total_batches = 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, masks)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().item())
        total_batches += 1
        metrics.update(logits.detach(), masks)
    avg_loss = total_loss / max(1, total_batches)
    return avg_loss, metrics.compute()


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)

    device = get_device()
    log.info("device: %s", device)

    # Resolve all input paths against the project root, not Hydra's cwd.
    images_dir = Path(to_absolute_path(cfg.data.images_dir))
    annotations_dir = Path(to_absolute_path(cfg.data.annotations_dir))
    band_stats_path = Path(to_absolute_path(cfg.data.band_stats_path))

    train_stems, val_stems = split_image_stems(images_dir, cfg.data.val_split, cfg.seed)
    log.info("train stems (%d): %s", len(train_stems), train_stems)
    log.info("val stems (%d): %s", len(val_stems), val_stems)

    train_ds = BenthicDataset(
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        band_stats_path=band_stats_path,
        stems=train_stems,
        mode="train",
        patch_size=cfg.data.patch_size,
        patches_per_epoch=cfg.data.patches_per_epoch,
        augment=cfg.data.augment,
        seed=cfg.seed,
    )
    val_ds = BenthicDataset(
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        band_stats_path=band_stats_path,
        stems=val_stems,
        mode="val",
        patch_size=cfg.data.patch_size,
        augment=False,
        seed=cfg.seed,
    )
    log.info("train patches/epoch: %d   val patches: %d", len(train_ds), len(val_ds))

    # num_workers=0 is required on MPS.
    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0
    )

    model = build_model(
        encoder_name=cfg.model.encoder_name,
        in_channels=cfg.model.in_channels,
        classes=cfg.model.num_classes,
        encoder_weights=cfg.model.encoder_weights,
    ).to(device)

    class_weights = torch.tensor(
        list(cfg.loss.class_weights), dtype=torch.float32, device=device
    )
    loss_fn = HybridLoss(
        class_weights=class_weights,
        ce_weight=cfg.loss.ce_weight,
        dice_weight=cfg.loss.dice_weight,
    )

    optimizer = Adam(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)

    project_root = Path(to_absolute_path("."))
    run_dir = project_root / cfg.results_dir / cfg.run_name
    ckpt_dir = project_root / cfg.checkpoint_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "train_log.csv"
    fieldnames = [
        "epoch", "lr", "train_loss", "val_loss",
        "val_pixel_acc", "val_macro_iou", "val_macro_dice",
        *[f"val_iou_{c}" for c in CLASS_NAMES],
        *[f"val_dice_{c}" for c in CLASS_NAMES],
    ]
    with log_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    # Snapshot resolved config alongside the checkpoint for evaluate.py.
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    best_macro_dice = -1.0
    best_path = ckpt_dir / f"{cfg.run_name}_best.pth"
    last_path = ckpt_dir / f"{cfg.run_name}_last.pth"

    for epoch in range(1, cfg.training.epochs + 1):
        train_loss, _ = run_epoch(
            model, train_loader, loss_fn, optimizer, device, cfg.model.num_classes
        )
        val_loss, val_stats = run_epoch(
            model, val_loader, loss_fn, None, device, cfg.model.num_classes
        )
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch, "lr": lr, "train_loss": train_loss, "val_loss": val_loss,
            "val_pixel_acc": val_stats["pixel_accuracy"],
            "val_macro_iou": val_stats["macro_iou"],
            "val_macro_dice": val_stats["macro_dice"],
        }
        for c, name in enumerate(CLASS_NAMES):
            row[f"val_iou_{name}"] = val_stats["per_class_iou"][c]
            row[f"val_dice_{name}"] = val_stats["per_class_dice"][c]
        with log_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        log.info(
            "epoch %03d/%d  train_loss=%.4f  val_loss=%.4f  val_macro_dice=%.4f  "
            "val_macro_iou=%.4f  val_px_acc=%.4f",
            epoch, cfg.training.epochs, train_loss, val_loss,
            val_stats["macro_dice"], val_stats["macro_iou"], val_stats["pixel_accuracy"],
        )

        torch.save(
            {"model": model.state_dict(), "config": cfg_dict, "epoch": epoch},
            last_path,
        )
        if val_stats["macro_dice"] > best_macro_dice:
            best_macro_dice = val_stats["macro_dice"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg_dict,
                    "epoch": epoch,
                    "val_macro_dice": best_macro_dice,
                },
                best_path,
            )

    summary = {
        "run_name": cfg.run_name,
        "epochs_run": cfg.training.epochs,
        "best_val_macro_dice": best_macro_dice,
        "best_checkpoint": str(best_path.relative_to(project_root)),
        "train_stems": train_stems,
        "val_stems": val_stems,
    }
    with (run_dir / "train_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    log.info("done. best val macro-Dice = %.4f", best_macro_dice)


if __name__ == "__main__":
    main()
