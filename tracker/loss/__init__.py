"""Pérdidas para la segmentación de puntos."""
from .seg_loss import (
    SegLoss,
    feature_contrast_loss,
    focal_loss,
    motion_seg_loss_baseline,
    soft_dice_loss,
)

__all__ = [
    "SegLoss",
    "feature_contrast_loss",
    "focal_loss",
    "motion_seg_loss_baseline",
    "soft_dice_loss",
]
