"""Detección: propuesta de cajas 3D desde la segmentación + métricas (mAP)."""
from .box_proposal import propose_boxes, fit_oriented_box

__all__ = ["propose_boxes", "fit_oriented_box"]
