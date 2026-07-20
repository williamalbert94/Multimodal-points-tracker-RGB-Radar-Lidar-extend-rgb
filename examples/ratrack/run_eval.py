"""Recover RaTrack tracking metrics from its exported per-frame predictions.

Two modes:

  python run_eval.py --selfcheck
      Parses the in-repo sample predictions (clip delft_10) and prints cluster
      statistics + a synthetic CLEAR-MOT self-test. Needs nothing but numpy/scipy
      -> use this to confirm the parsing + metric core work.

  python run_eval.py --eval [--clips delft_1 delft_22 ...]
      Full evaluation: matches RaTrack predictions against VoD GT tracking labels
      with point-based IoU and reports MOTA/MODA/MT/ML/IDS + sAMOTA/AMOTA/AMOTP.
      Requires the VoD dataset, open3d, and RaTrack's src on disk (see paths.py).
"""
import argparse
import os
import numpy as np

import paths
from ratrack_io import (parse_prediction_frame, parse_gt_frame,
                        parse_gt_frame_moving, frame_ids_for_clip)
from mot_metrics import compute_clearmot, compute_integral


# RaTrack tracks MOVING objects only, so GT is filtered the same way by default
# (see ratrack_io.parse_gt_frame_moving). Set MOVING_ONLY=False to score against
# every annotated object instead.
MOVING_ONLY = True


def _load_gt(frame_id):
    trk = os.path.join(paths.GT_TRACKING_LABEL_DIR, f"{frame_id}.txt")
    if not MOVING_ONLY:
        return parse_gt_frame(trk)
    det = os.path.join(paths.VOD_ROOT, "lidar", "training", "label_2", f"{frame_id}.txt")
    return parse_gt_frame_moving(trk, det)


def _prediction_dir(base, clip):
    """Find a clip's prediction folder under `base` (handles the nested
    4dmot_runthis/<clip> layout as well as <base>/<clip>)."""
    for cand in (os.path.join(base, clip),
                 os.path.join(base, "4dmot_runthis", clip)):
        if os.path.isdir(cand):
            return cand
    return None


def selfcheck():
    print("== sample prediction parse (clip delft_10) ==")
    clip_dir = _prediction_dir(paths.SAMPLE_PREDICTION_DIR, "delft_10")
    assert clip_dir, f"sample not found under {paths.SAMPLE_PREDICTION_DIR}"
    files = sorted(f for f in os.listdir(clip_dir) if f.endswith(".txt"))
    n_obj = 0
    ids = set()
    for fn in files:
        for o in parse_prediction_frame(os.path.join(clip_dir, fn)):
            n_obj += 1
            ids.add(o.track_id)
    print(f"  {len(files)} frames, {n_obj} object-instances, {len(ids)} unique track ids")

    print("== synthetic CLEAR-MOT self-test ==")
    # GT id 7 tracked as 100,100,MISS,200  -> expect IDS=1, FN=1, TP=3
    def fr(gt, pid, iou=0.9, conf=0.9):
        g = [7] if gt else []
        if pid is None:
            return (g, [], [], np.zeros((len(g), 0)))
        return (g, [pid], [conf], np.array([[iou]]) if gt else np.zeros((0, 1)))
    frames = [fr(True, 100), fr(True, 100), fr(True, None), fr(True, 200)]
    m = compute_clearmot(frames, conf_thr=0.0, iou_thr=paths.POINT_IOU_THRESHOLD)
    print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
    assert (m["idsw"], m["tp"], m["fn"]) == (1, 3, 1), m
    print("  OK core metrics")


def _build_frames_for_clip(clip, pred_base):
    """Yield per-frame (gt_ids, pred_ids, pred_conf, iou) for a clip."""
    from point_iou import point_iou_matrix  # heavy import, only when evaluating
    clip_dir = _prediction_dir(pred_base, clip)
    if not clip_dir:
        print(f"  [skip] no predictions for {clip}")
        return []
    frames = []
    for fn in sorted(f for f in os.listdir(clip_dir) if f.endswith(".txt")):
        frame_id = fn[:-4]
        preds = parse_prediction_frame(os.path.join(clip_dir, fn))
        gts = _load_gt(frame_id)
        iou = point_iou_matrix(gts, preds, frame_id)
        frames.append(([g.track_id for g in gts],
                       [p.track_id for p in preds],
                       [p.conf for p in preds], iou))
    return frames


def idsw_only(clips, pred_base):
    """Recover just the ID-switch count (primary ask). Needs dataset + open3d."""
    from idsw_eval import idsw_report, summarize
    all_frames = []
    for clip in clips:
        fr = _build_frames_for_clip(clip, pred_base)
        print(f"  {clip}: {len(fr)} frames")
        all_frames.extend(fr)
    if not all_frames:
        print("No frames evaluated. Check paths.py / dataset availability.")
        return
    print("\n=== ID switches (established-track id changes, recovery excluded) ===")
    total, per_track = idsw_report(all_frames, iou_thr=paths.POINT_IOU_THRESHOLD)
    summarize(total, per_track)


def seg_eval(clips, pred_base):
    """Motion-segmentation mIoU using RaTrack's eval_motion_seg formula.

    IMPORTANT: this measures a DIFFERENT PIPELINE STAGE than RaTrack's published
    mIoU. RaTrack scores its segmentation *head* (`cls > 0.5`); that mask is not
    exported, so here the predicted moving mask is the union of the *final
    tracked clusters* (post motion-seg + DBSCAN + tracking). Clustering discards
    spurious moving points, so this number is typically HIGHER than the seg-head
    mIoU and is not a like-for-like substitute for it. To reproduce RaTrack's
    number exactly, export the `cls` mask (see seg_miou.py header)."""
    import numpy as np
    from seg_miou import eval_motion_seg, predicted_moving_mask_from_clusters
    from point_iou import radar_indices_in_boxes

    mious, accs, sens = [], [], []
    for clip in clips:
        clip_dir = _prediction_dir(pred_base, clip)
        if not clip_dir:
            print(f"  [skip] no predictions for {clip}")
            continue
        n = 0
        for fn in sorted(f for f in os.listdir(clip_dir) if f.endswith(".txt")):
            frame_id = fn[:-4]
            preds = parse_prediction_frame(os.path.join(clip_dir, fn))
            gts = _load_gt(frame_id)
            gt_mask, radar = radar_indices_in_boxes(gts, frame_id)
            pred_mask = predicted_moving_mask_from_clusters(preds, radar)
            m = eval_motion_seg(pred_mask, gt_mask)
            mious.append(m["miou"]); accs.append(m["acc"]); sens.append(m["sen"])
            n += 1
        print(f"  {clip}: {n} frames")

    if mious:
        scope = "moving-only GT" if MOVING_ONLY else "all GT"
        print(f"\n=== Motion-segmentation, tracked-cluster stage ({scope}) ===")
        print(f"  mIoU       : {100*np.mean(mious):.2f}")
        print(f"  acc        : {100*np.mean(accs):.2f}")
        print(f"  sensitivity: {100*np.mean(sens):.2f}")
        print("  RaTrack published (seg-HEAD stage): 57.0")
        print("  NOT like-for-like: this scores the final tracked clusters, not the\n"
              "  seg head. Export the cls mask (seg_miou.py header) to match exactly.")


def full_eval(clips, pred_base):
    all_frames = []
    for clip in clips:
        fr = _build_frames_for_clip(clip, pred_base)
        print(f"  {clip}: {len(fr)} frames")
        all_frames.extend(fr)
    if not all_frames:
        print("No frames evaluated. Check paths.py / dataset availability.")
        return

    thr = paths.POINT_IOU_THRESHOLD
    single = compute_clearmot(all_frames, conf_thr=0.0, iou_thr=thr)
    integ = compute_integral(all_frames, iou_thr=thr, num_sample_pts=paths.NUM_SAMPLE_PTS)

    print("\n=== RaTrack metrics (recovered) ===")
    for k in ("sAMOTA", "AMOTA", "AMOTP"):
        print(f"  {k:8s}: {100*integ.get(k, float('nan')):.2f}")
    for k in ("mota", "moda", "mt", "ml"):
        print(f"  {k.upper():8s}: {100*single[k]:.2f}")
    print(f"  IDS     : {single['idsw']}")
    print(f"  FRAG    : {single['frag']}")
    print("\nCompare against RaTrack's published table (README):")
    print("  RaTrack: sAMOTA 74.16 | AMOTA 31.50 | AMOTP 60.17 | MOTA 67.27 | "
          "MODA 77.83 | MT 42.65 | ML 14.71")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--idsw", action="store_true", help="recover ID switches only")
    ap.add_argument("--seg", action="store_true", help="motion-seg mIoU (approx)")
    ap.add_argument("--eval", action="store_true", help="full CLEAR-MOT + AMOTA")
    ap.add_argument("--clips", nargs="*", default=paths.VAL_CLIPS)
    ap.add_argument("--pred-base", default=paths.RATRACK_RESULT_DIR)
    args = ap.parse_args()

    if args.selfcheck or not (args.eval or args.idsw or args.seg):
        selfcheck()
    if args.idsw:
        idsw_only(args.clips, args.pred_base)
    if args.seg:
        seg_eval(args.clips, args.pred_base)
    if args.eval:
        full_eval(args.clips, args.pred_base)
