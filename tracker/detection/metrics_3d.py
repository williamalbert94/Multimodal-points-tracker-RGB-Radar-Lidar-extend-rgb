"""
Detection Metrics for Object Detection and Segmentation.

Metrics:
  - mAP (mean Average Precision) for 3D box detection
  - mIoU (mean Intersection over Union) for segmentation
  - Per-class metrics

Uses ROTATED BEV IoU (polygon intersection) for correct handling of
oriented bounding boxes. Axis-aligned IoU severely underestimates
overlap for rotated objects (cars at 45 deg -> ~50% IoU penalty).
"""

import numpy as np
import torch
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# Rotated 2D BEV IoU (polygon intersection)
# ─────────────────────────────────────────────────────────────────────────────

def _box_corners_2d(box):
    """Get 4 BEV corners of a rotated box. box: [7] (x,y,z,l,w,h,yaw)."""
    x, y = float(box[0]), float(box[1])
    l, w = float(box[3]), float(box[4])
    yaw = float(box[6]) if len(box) > 6 else 0.0

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    hl, hw = l / 2.0, w / 2.0

    dx = np.array([-hl, hl, hl, -hl])
    dy = np.array([-hw, -hw, hw, hw])

    corners_x = x + dx * cos_yaw - dy * sin_yaw
    corners_y = y + dx * sin_yaw + dy * cos_yaw

    return np.stack([corners_x, corners_y], axis=1)  # [4, 2]


def _polygon_area(vertices):
    """Shoelace formula for polygon area."""
    n = len(vertices)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0


def _line_segment_intersection(p1, p2, p3, p4):
    """Intersection of segments p1-p2 and p3-p4. Returns (x,y) or None."""
    d1x = p2[0] - p1[0]
    d1y = p2[1] - p1[1]
    d2x = p4[0] - p3[0]
    d2y = p4[1] - p3[1]
    cross = d1x * d2y - d1y * d2x
    if abs(cross) < 1e-10:
        return None
    t = ((p3[0] - p1[0]) * d2y - (p3[1] - p1[1]) * d2x) / cross
    u = ((p3[0] - p1[0]) * d1y - (p3[1] - p1[1]) * d1x) / cross
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (p1[0] + t * d1x, p1[1] + t * d1y)
    return None


def _point_in_polygon(point, polygon):
    """Ray casting point-in-polygon test."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _polygon_intersection_area(poly1, poly2):
    """Intersection area of two convex polygons via point collection."""
    intersection_points = []
    n1, n2 = len(poly1), len(poly2)

    for p in poly1:
        if _point_in_polygon(p, poly2):
            intersection_points.append(tuple(p))
    for p in poly2:
        if _point_in_polygon(p, poly1):
            intersection_points.append(tuple(p))

    for i in range(n1):
        for j in range(n2):
            pt = _line_segment_intersection(
                poly1[i], poly1[(i + 1) % n1],
                poly2[j], poly2[(j + 1) % n2]
            )
            if pt is not None:
                intersection_points.append(pt)

    if len(intersection_points) < 3:
        return 0.0

    # Remove near-duplicates
    unique = []
    for p in intersection_points:
        if not any(abs(p[0] - q[0]) < 1e-8 and abs(p[1] - q[1]) < 1e-8 for q in unique):
            unique.append(p)

    if len(unique) < 3:
        return 0.0

    # Sort by angle for convex hull ordering
    cx = sum(p[0] for p in unique) / len(unique)
    cy = sum(p[1] for p in unique) / len(unique)
    unique.sort(key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))

    return _polygon_area(unique)


def compute_rotated_iou_2d(box1, box2):
    """
    Compute rotated 2D BEV IoU between two boxes.
    Properly handles yaw orientation via polygon intersection.

    Args:
        box1, box2: [7] (x, y, z, l, w, h, yaw)
    Returns:
        iou: float
    """
    corners1 = _box_corners_2d(box1)
    corners2 = _box_corners_2d(box2)
    inter_area = _polygon_intersection_area(corners1, corners2)
    area1 = float(box1[3]) * float(box1[4])
    area2 = float(box2[3]) * float(box2[4])
    union_area = area1 + area2 - inter_area
    return inter_area / max(union_area, 1e-6)


def compute_iou_3d_boxes(box1, box2):
    """Compute rotated BEV IoU (backward-compatible name)."""
    return compute_rotated_iou_2d(box1, box2)


def compute_iou_matrix(boxes_pred, boxes_gt):
    """
    Compute ROTATED IoU matrix between predicted and GT boxes.

    Args:
        boxes_pred: [M, 7] predicted boxes
        boxes_gt: [N, 7] GT boxes
    Returns:
        iou_matrix: [M, N]
    """
    if isinstance(boxes_pred, torch.Tensor):
        boxes_pred = boxes_pred.detach().cpu().numpy()
    if isinstance(boxes_gt, torch.Tensor):
        boxes_gt = boxes_gt.detach().cpu().numpy()

    M = len(boxes_pred)
    N = len(boxes_gt)
    iou_matrix = np.zeros((M, N))

    for i in range(M):
        for j in range(N):
            iou_matrix[i, j] = compute_rotated_iou_2d(boxes_pred[i], boxes_gt[j])

    return iou_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Average Precision
# ─────────────────────────────────────────────────────────────────────────────

def compute_average_precision(recalls, precisions):
    """11-point interpolation AP. Kept for backward compatibility."""
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        mask = recalls >= t
        if np.any(mask):
            p = np.max(precisions[mask])
        else:
            p = 0
        ap += p / 11.0
    return ap


def compute_average_precision_allpoint(recalls, precisions):
    """All-point interpolation AP (PASCAL VOC 2010+). More accurate."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


# ─────────────────────────────────────────────────────────────────────────────
# mAP computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_map_3d(boxes_pred_list, boxes_gt_list, scores_pred_list=None,
                   class_pred_list=None, class_gt_list=None,
                   iou_thresholds=[0.3, 0.5, 0.7], per_class=False):
    """
    Compute mAP for 3D box detection using ROTATED BEV IoU.

    Args:
        boxes_pred_list: List[Tensor/ndarray] predicted boxes per sample [M_i, 7]
        boxes_gt_list: List[Tensor/ndarray] GT boxes per sample [N_i, 7]
        scores_pred_list: List[Tensor/ndarray] confidence scores per sample [M_i]
        class_pred_list: List[Tensor/ndarray] predicted classes (optional)
        class_gt_list: List[Tensor/ndarray] GT classes (optional)
        iou_thresholds: List[float] IoU thresholds
        per_class: bool, compute per-class AP
    Returns:
        metrics: dict
    """
    metrics = {}

    # Compute IoU matrix once per sample (reuse across IoU thresholds)
    iou_cache = [None] * len(boxes_pred_list)
    np_pred = [None] * len(boxes_pred_list)
    np_gt = [None] * len(boxes_pred_list)
    np_scores = [None] * len(boxes_pred_list)
    np_cls_pred = [None] * len(boxes_pred_list)
    np_cls_gt = [None] * len(boxes_pred_list)

    for i in range(len(boxes_pred_list)):
        bp = boxes_pred_list[i]
        bg = boxes_gt_list[i]
        if isinstance(bp, torch.Tensor): bp = bp.detach().cpu().numpy()
        if isinstance(bg, torch.Tensor): bg = bg.detach().cpu().numpy()
        np_pred[i] = bp
        np_gt[i] = bg
        if scores_pred_list is not None and i < len(scores_pred_list):
            s = scores_pred_list[i]
            if isinstance(s, torch.Tensor): s = s.detach().cpu().numpy()
            np_scores[i] = s
        else:
            np_scores[i] = np.ones(len(bp))
        if class_pred_list is not None and i < len(class_pred_list):
            c = class_pred_list[i]
            if isinstance(c, torch.Tensor): c = c.detach().cpu().numpy()
            np_cls_pred[i] = c
        if class_gt_list is not None and i < len(class_gt_list):
            c = class_gt_list[i]
            if isinstance(c, torch.Tensor): c = c.detach().cpu().numpy()
            np_cls_gt[i] = c
        if len(bp) > 0 and len(bg) > 0:
            iou_cache[i] = compute_iou_matrix(bp, bg)

    for iou_thresh in iou_thresholds:
        all_tp = []
        all_fp = []
        all_scores = []
        all_num_gt = 0

        for i in range(len(boxes_pred_list)):
            boxes_pred = np_pred[i]
            boxes_gt = np_gt[i]
            scores_i = np_scores[i]
            cls_pred_i = np_cls_pred[i]
            cls_gt_i = np_cls_gt[i]

            M, N = len(boxes_pred), len(boxes_gt)
            all_num_gt += N

            if M == 0:
                continue
            if N == 0:
                all_fp.extend([1] * M)
                all_tp.extend([0] * M)
                all_scores.extend(scores_i.tolist())
                continue

            iou_matrix = iou_cache[i]

            sorted_pred_idx = np.argsort(-scores_i)
            matched_gt = set()

            for pred_idx in sorted_pred_idx:
                ious = iou_matrix[pred_idx].copy()
                # Class-aware: zero out IoU against GTs of different class
                if cls_pred_i is not None and cls_gt_i is not None:
                    mismatch = cls_gt_i != cls_pred_i[pred_idx]
                    ious[mismatch] = 0.0

                max_iou = np.max(ious) if len(ious) > 0 else 0.0
                max_gt_idx = int(np.argmax(ious)) if len(ious) > 0 else -1

                if max_iou >= iou_thresh and max_gt_idx not in matched_gt:
                    all_tp.append(1)
                    all_fp.append(0)
                    matched_gt.add(max_gt_idx)
                else:
                    all_tp.append(0)
                    all_fp.append(1)

                all_scores.append(float(scores_i[pred_idx]))

        if len(all_scores) == 0:
            metrics[f'mAP@{iou_thresh}'] = 0.0
            metrics[f'F1@{iou_thresh}'] = 0.0
            continue

        sorted_indices = np.argsort(-np.array(all_scores))
        tp_sorted = np.array(all_tp)[sorted_indices]
        fp_sorted = np.array(all_fp)[sorted_indices]

        tp_cumsum = np.cumsum(tp_sorted)
        fp_cumsum = np.cumsum(fp_sorted)

        recalls = tp_cumsum / max(all_num_gt, 1)
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1)

        ap = compute_average_precision_allpoint(recalls, precisions)
        metrics[f'mAP@{iou_thresh}'] = ap

        if len(precisions) > 0 and len(recalls) > 0:
            f1_scores = 2 * (precisions * recalls) / np.maximum(precisions + recalls, 1e-6)
            metrics[f'F1@{iou_thresh}'] = float(np.max(f1_scores))

    metrics['mAP'] = float(np.mean([metrics[f'mAP@{t}'] for t in iou_thresholds]))
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_miou_segmentation(seg_pred_list, seg_gt_list, num_classes=2):
    """Compute mIoU for segmentation."""
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(seg_pred_list)):
        seg_pred = seg_pred_list[i]
        seg_gt = seg_gt_list[i]
        if isinstance(seg_pred, torch.Tensor):
            seg_pred = seg_pred.cpu().numpy()
        if isinstance(seg_gt, torch.Tensor):
            seg_gt = seg_gt.cpu().numpy()
        seg_pred = seg_pred.flatten()
        seg_gt = seg_gt.flatten()
        for c_pred in range(num_classes):
            for c_gt in range(num_classes):
                confusion_matrix[c_pred, c_gt] += np.sum(
                    (seg_pred == c_pred) & (seg_gt == c_gt))

    iou_per_class = []
    for c in range(num_classes):
        intersection = confusion_matrix[c, c]
        union = np.sum(confusion_matrix[c, :]) + np.sum(confusion_matrix[:, c]) - intersection
        iou_per_class.append(intersection / max(union, 1))

    return {
        'mIoU': np.mean(iou_per_class),
        'IoU_background': iou_per_class[0] if num_classes > 0 else 0,
        'IoU_foreground': iou_per_class[1] if num_classes > 1 else 0,
    }


def compute_detection_metrics_epoch(boxes_pred_list, boxes_gt_list,
                                    seg_pred_list=None, seg_gt_list=None):
    """Compute detection metrics for an entire epoch."""
    metrics = {}
    if boxes_pred_list and boxes_gt_list:
        metrics.update(compute_map_3d(
            boxes_pred_list, boxes_gt_list,
            iou_thresholds=[0.3, 0.5, 0.7]))
    if seg_pred_list and seg_gt_list:
        metrics.update(compute_miou_segmentation(seg_pred_list, seg_gt_list))
    return metrics


def print_detection_metrics(metrics, prefix=''):
    """Print detection metrics."""
    print(f"\n{'='*60}")
    print(f"  {prefix} Detection Metrics (Rotated BEV IoU)")
    print(f"{'='*60}")
    if 'mAP' in metrics:
        print(f"  Box Detection:")
        print(f"     mAP (avg):     {metrics['mAP']:.4f}")
        for t in [0.25, 0.3, 0.5, 0.7]:
            k = f'mAP@{t}'
            if k in metrics:
                print(f"     {k}:       {metrics[k]:.4f}")
        if 'F1@0.5' in metrics:
            print(f"     F1@0.5:        {metrics['F1@0.5']:.4f}")
    if 'mIoU' in metrics:
        print(f"\n  Segmentation:")
        print(f"     mIoU:          {metrics['mIoU']:.4f}")
    print(f"{'='*60}\n")
