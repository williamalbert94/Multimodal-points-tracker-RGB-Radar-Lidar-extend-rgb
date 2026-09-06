"""Focused ID-switch (IDSW) evaluation from RaTrack predictions vs VoD GT.

Definition used (standard CLEAR-MOT, matching what the user asked for):

    An ID switch is counted when a GT track that is **already established**
    (i.e. was matched to some prediction id at least once) is now matched to a
    *different* prediction id.

  * The first assignment to a GT track is NOT a switch (the track is only just
    being "defined").
  * Gaps do NOT reset the reference id: if a GT is lost for a few frames and
    re-acquired with the SAME prediction id, that is recovery, NOT a switch.
    Only a change to a different id counts. (This is the "cambios de id después
    de definir el track, no recuperar" the user described.)

IDSW therefore measures tracking-identity *granularity/instability*, not
re-identification recall.

Per-frame GT<->prediction correspondence comes from the point-based IoU in
`point_iou.py` (needs the dataset + open3d + RaTrack src). Everything else here
is pure Python.
"""
from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


def _match(gt_ids, pred_ids, iou, iou_thr):
    out = {}
    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return out
    cost = 1.0 - iou
    cost[iou < iou_thr] = 1e6
    for r, c in zip(*linear_sum_assignment(cost)):
        if iou[r, c] >= iou_thr:
            out[gt_ids[r]] = pred_ids[c]
    return out


def idsw_report(frames, iou_thr=0.25):
    """frames: list of (gt_ids, pred_ids, pred_conf, iou[G,P]) in time order.

    Returns (total_idsw, per_track) where per_track[gt_id] = list of
    (from_pred_id, to_pred_id) switch events.
    """
    # reference prediction id currently associated with each established GT track
    ref: Dict[int, int] = {}
    per_track: Dict[int, List[Tuple[int, int]]] = {}
    total = 0

    for gt_ids, pred_ids, _conf, iou in frames:
        matches = _match(list(gt_ids), list(pred_ids), np.asarray(iou, float), iou_thr)
        for gid, pid in matches.items():
            if gid not in ref:
                ref[gid] = pid            # first assignment: track defined, no switch
                per_track.setdefault(gid, [])
            elif pid != ref[gid]:
                per_track[gid].append((ref[gid], pid))
                total += 1
                ref[gid] = pid            # follow the new id
    return total, per_track


def summarize(total, per_track):
    n_tracks = len(per_track)
    switched = sum(1 for v in per_track.values() if v)
    print(f"  established GT tracks : {n_tracks}")
    print(f"  tracks with >=1 switch: {switched}")
    print(f"  total ID switches     : {total}")
    if switched:
        print("  per-track switch events (gt_id: from->to ...):")
        for gid, evs in per_track.items():
            if evs:
                chain = " ".join(f"{a}->{b}" for a, b in evs)
                print(f"    gt {gid}: {chain}")
