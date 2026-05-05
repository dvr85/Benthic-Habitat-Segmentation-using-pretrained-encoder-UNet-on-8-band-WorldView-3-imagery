"""Running accumulator: per-class IoU, Dice, pixel accuracy, confusion matrix."""

from __future__ import annotations

import numpy as np
import torch


class SegmentationMetrics:
    """Accumulate batches via `update(logits, target)` then call `compute()`.

    Confusion matrix is maintained on CPU with torch.bincount to avoid MPS
    edge cases with scatter-add on int64.
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self.confusion = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64
        )

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        preds = logits.argmax(dim=1).detach().to("cpu").long().reshape(-1)
        tgt = target.detach().to("cpu").long().reshape(-1)
        valid = (tgt >= 0) & (tgt < self.num_classes)
        preds, tgt = preds[valid], tgt[valid]
        idx = tgt * self.num_classes + preds
        counts = torch.bincount(idx, minlength=self.num_classes * self.num_classes)
        self.confusion += counts.view(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        cm = self.confusion.numpy().astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        iou_denom = tp + fp + fn
        dice_denom = 2 * tp + fp + fn
        iou = np.divide(tp, iou_denom, out=np.zeros_like(tp), where=iou_denom > 0)
        dice = np.divide(2 * tp, dice_denom, out=np.zeros_like(tp), where=dice_denom > 0)

        total = cm.sum()
        pixel_acc = float(tp.sum() / total) if total > 0 else 0.0

        present = cm.sum(axis=1) > 0
        macro_iou = float(iou[present].mean()) if present.any() else 0.0
        macro_dice = float(dice[present].mean()) if present.any() else 0.0

        return {
            "per_class_iou": iou.tolist(),
            "per_class_dice": dice.tolist(),
            "macro_iou": macro_iou,
            "macro_dice": macro_dice,
            "pixel_accuracy": pixel_acc,
            "confusion_matrix": cm.astype(np.int64).tolist(),
            "present_classes": present.tolist(),
        }
