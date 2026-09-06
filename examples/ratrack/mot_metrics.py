"""Point-based CLEAR-MOT + integral (AMOTA) metrics.

The accumulation logic mirrors AB3DMOT's `scripts/KITTI/evaluate.py`
(see /home/williamramirez/multimodal_reid/AB3DMOT), the difference being that
the per-frame prediction<->GT affinity is a *point-based IoU* (built in
`point_iou.py`) rather than a 3D box IoU, because RaTrack outputs point clusters
instead of boxes.

Definitions reproduced here (all in [0,1] unless noted):
  MOTA  = 1 - (FN + FP + IDS) / n_gt
  MODA  = 1 - (FN + FP)       / n_gt          # detection only, ignores IDS
  MOTP  = mean association IoU over true positives
  MT/ML = fraction of GT trajectories tracked > 80% / < 20% of their length
  IDS   = identity switches, FRAG = fragmentations
  sMOTA = clip(1 - (FN + FP + IDS - (1-recall)*n_gt) / (recall*n_gt), 0, 1)
Integral metrics average MOTA / MOTP / sMOTA over `num_sample_pts-1` recall
points, giving AMOTA / AMOTP / sAMOTA.
"""
from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


# A "frame" of matching input: parallel lists describing one time step.
#   gt_ids   : list[int]        track ids of GT objects present
#   pred_ids : list[int]        track ids of predicted objects present
#   pred_conf: list[float]      confidence of each predicted object
#   iou      : np.ndarray (G,P) point-based IoU between GT g and pred p
FrameMatch = Tuple[List[int], List[int], List[float], np.ndarray]


def _match_frame(gt_ids, pred_ids, iou, iou_thr):
    """Hungarian match GT<->pred at a given IoU threshold.

    Returns dict gt_index -> (pred_index, iou) for matched pairs only.
    """
    matches = {}
    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return matches
    cost = 1.0 - iou
    cost[iou < iou_thr] = 1e6  # forbid sub-threshold pairs
    rows, cols = linear_sum_assignment(cost)
    for r, c in zip(rows, cols):
        if iou[r, c] >= iou_thr:
            matches[r] = (c, iou[r, c])
    return matches


def compute_clearmot(frames: List[FrameMatch], conf_thr: float, iou_thr: float):
    """Single-threshold CLEAR-MOT pass over all frames.

    Predictions with confidence < conf_thr are dropped before matching (this is
    how the recall sweep for AMOTA is produced).
    """
    n_gt = 0
    fp = fn = idsw = frag = tp = 0
    motp_sum = 0.0

    # per GT trajectory bookkeeping for MT/ML + IDS + FRAG
    # traj[gt_id] = list over frames of matched-pred-id (or -1 if untracked this frame)
    traj: Dict[int, List[int]] = {}

    for gt_ids, pred_ids, pred_conf, iou in frames:
        # filter predictions by confidence
        keep = [i for i, c in enumerate(pred_conf) if c >= conf_thr]
        pred_ids_k = [pred_ids[i] for i in keep]
        iou_k = iou[:, keep] if (iou.size and keep) else np.zeros((len(gt_ids), 0))

        n_gt += len(gt_ids)
        matches = _match_frame(gt_ids, pred_ids_k, iou_k.copy(), iou_thr)

        matched_pred = set()
        # register per-GT presence this frame
        seen_gt = set()
        for g_idx, gid in enumerate(gt_ids):
            traj.setdefault(gid, [])
            seen_gt.add(gid)
            if g_idx in matches:
                p_idx, ov = matches[g_idx]
                tp += 1
                motp_sum += ov
                matched_pred.add(p_idx)
                traj[gid].append(pred_ids_k[p_idx])
            else:
                fn += 1
                traj[gid].append(-1)
        # NOTE: only frames where the GT object actually EXISTS are appended.
        # Padding absent objects with -1 would make MT/ML divide by the whole
        # sequence length (1289 frames here) instead of the object's lifespan,
        # pushing essentially every trajectory into "mostly lost".
        fp += len(pred_ids_k) - len(matched_pred)

    # IDS + FRAG from trajectories
    for gid, seq in traj.items():
        last = -1
        prev_present = False
        for cur in seq:
            if cur != -1:
                if last != -1 and cur != last:
                    idsw += 1
                last = cur
                if not prev_present and any(x != -1 for x in seq[:seq.index(cur)]):
                    frag += 1
                prev_present = True
            else:
                prev_present = False

    # MT / ML
    n_traj = len(traj)
    mt = ml = 0
    for seq in traj.values():
        length = sum(1 for _ in seq)
        tracked = sum(1 for x in seq if x != -1)
        ratio = tracked / float(length) if length else 0.0
        if ratio > 0.8:
            mt += 1
        elif ratio < 0.2:
            ml += 1

    recall = tp / float(tp + fn) if (tp + fn) else 0.0
    mota = 1.0 - (fn + fp + idsw) / float(n_gt) if n_gt else 0.0
    moda = 1.0 - (fn + fp) / float(n_gt) if n_gt else 0.0
    motp = motp_sum / float(tp) if tp else 0.0
    if n_gt and recall > 0:
        smota = min(1.0, max(0.0,
                    1.0 - (fn + fp + idsw - (1 - recall) * n_gt) / float(recall * n_gt)))
    else:
        smota = 0.0

    return {
        "n_gt": n_gt, "tp": tp, "fp": fp, "fn": fn, "idsw": idsw, "frag": frag,
        "recall": recall, "mota": mota, "moda": moda, "motp": motp, "smota": smota,
        "mt": mt / n_traj if n_traj else 0.0, "ml": ml / n_traj if n_traj else 0.0,
    }


def compute_integral(frames: List[FrameMatch], iou_thr: float, num_sample_pts: int = 41):
    """Sweep confidence thresholds to get `num_sample_pts-1` recall points and
    average MOTA/MOTP/sMOTA -> AMOTA/AMOTP/sAMOTA (AB3DMOT-style)."""
    all_conf = sorted({c for _, _, cs, _ in frames for c in cs}, reverse=True)
    if not all_conf:
        return {}

    # discretize confidences into recall-uniform thresholds
    thresholds = np.quantile(all_conf, np.linspace(0, 1, num_sample_pts - 1))
    mota_l, motp_l, smota_l = [], [], []
    for thr in thresholds:
        m = compute_clearmot(frames, conf_thr=float(thr), iou_thr=iou_thr)
        mota_l.append(m["mota"]); motp_l.append(m["motp"]); smota_l.append(m["smota"])

    return {
        "AMOTA": float(np.mean(mota_l)),
        "AMOTP": float(np.mean(motp_l)),
        "sAMOTA": float(np.mean(smota_l)),
        # best-threshold single-point metrics, reported alongside like AB3DMOT
        "best_MOTA": float(np.max(mota_l)),
    }
