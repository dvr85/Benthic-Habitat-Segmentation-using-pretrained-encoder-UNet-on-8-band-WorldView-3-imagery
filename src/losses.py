"""Hybrid Cross-Entropy + Dice loss for multi-class segmentation."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class HybridLoss(nn.Module):
    """Weighted convex combination of weighted CE and multi-class Dice.

    Default 0.7 CE / 0.3 Dice balances class-frequency awareness (CE with
    class weights) against region-overlap quality (Dice).
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        ce_weight: float = 0.7,
        dice_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.ce_w = ce_weight
        self.dice_w = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_long = target.long()
        return self.ce_w * self.ce(logits, target_long) + self.dice_w * self.dice(
            logits, target_long
        )
