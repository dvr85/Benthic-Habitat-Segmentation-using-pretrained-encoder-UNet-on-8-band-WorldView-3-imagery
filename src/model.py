"""Factory for an SMP UNet adapted to 8-band WorldView-3 input."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(
    encoder_name: str = "resnet50",
    in_channels: int = 8,
    classes: int = 7,
    encoder_weights: str | None = "imagenet",
) -> nn.Module:
    """SMP re-initializes the stem conv when in_channels != 3 while keeping
    the deeper pretrained encoder weights intact."""
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
