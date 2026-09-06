"""Motion-segmentation mIoU, based on RaTrack's own `eval_motion_seg`.

RaTrack's mIoU (main_utils.py:377) is a per-radar-point binary
moving-vs-static IoU, averaged over the two classes:

    tp = (pre==1 & gt==1); tn = (pre==0 & gt==0)
    fp = (pre==1 & gt==0); fn = (pre==0 & gt==1)
    mIoU = 0.5 * ( tp/(tp+fp+fn) + tn/(tn+fp+fn) )

where `pre` is the segmentation head's per-point mask (`cls > 0.5`) and `gt` is
the per-point moving label (1 if the point falls in a *moving* GT box).

IMPORTANT — export gap
----------------------
RaTrack's exported `.txt` files contain ONLY the clustered *moving objects*, not
the per-point `cls` mask. So the exact RaTrack mIoU CANNOT be recomputed from the
current exports alone. Two paths:

  (A) EXACT — re-run RaTrack inference and additionally dump the `cls` mask.
      One extra block in RaTrack's `main_utils.epoch()` writer does it, e.g.:

          np.save(f"./results_seg/{seq[0]}/{index:05d}.npy",
                  (cls.squeeze() > 0.5).cpu().numpy().astype(np.uint8))

      then feed that mask as `pre` to `eval_motion_seg` here -> identical number.

  (B) APPROX — reconstruct a moving mask from the exported clusters (the union
      of all tracked cluster points). This is a *subset* of the true `cls==1`
      points (points not clustered into an object are dropped), so it is a
      LOWER BOUND on IoU_moving. Use only as a rough estimate.

`eval_motion_seg` here is pure numpy and byte-for-byte matches RaTrack's formula,
so path (A) reproduces their reported mIoU exactly.
"""
import numpy as np

import paths


def eval_motion_seg(pre: np.ndarray, gt: np.ndarray) -> dict:
    """Exact replica of RaTrack's eval_motion_seg (binary moving/static)."""
    pre = np.asarray(pre).astype(bool)
    gt = np.asarray(gt).astype(bool)
    tp = np.logical_and(pre, gt).sum() + 1e-20
    tn = np.logical_and(~pre, ~gt).sum() + 1e-20
    fp = np.logical_and(pre, ~gt).sum() + 1e-20
    fn = np.logical_and(~pre, gt).sum() + 1e-20
    acc = (tp + tn) / (tp + tn + fp + fn)
    sen = tp / (tp + fn)
    miou = 0.5 * (tp / (tp + fp + fn + 1e-4) + tn / (tn + fp + fn + 1e-4))
    return {"acc": float(acc), "miou": float(miou), "sen": float(sen)}


def predicted_moving_mask_from_clusters(pred_objs, radar_cloud, tol=1e-3):
    """APPROX path (B): union of tracked cluster points -> boolean moving mask
    over the frame's radar cloud."""
    mask = np.zeros(radar_cloud.shape[0], dtype=bool)
    for p in pred_objs:
        for pt in p.points:
            d = np.linalg.norm(radar_cloud - pt, axis=1)
            mask[int(np.argmin(d))] = True
    return mask


def gt_moving_mask(moving_gt_objs, frame_id):
    """Boolean mask of radar points inside *moving* GT boxes (radar frame),
    plus the frame's radar cloud.

    `moving_gt_objs` should already be filtered to moving objects (ideally via
    RaTrack's `filter_moving_boxes_det` for an exact match)."""
    from point_iou import radar_indices_in_boxes
    return radar_indices_in_boxes(moving_gt_objs, frame_id)
