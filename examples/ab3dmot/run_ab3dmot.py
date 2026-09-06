"""Run the real AB3DMOT tracker on View-of-Delft, on RaTrack's split, and
recover ID switches (IDSW).

AB3DMOT is tracking-by-DETECTION: it never detects anything itself, it consumes
3D boxes and associates them over time with a 3D Kalman filter + Hungarian
matching. So the number you get depends entirely on which detections you feed it:

  --dets gt        Ground-truth boxes as "perfect" detections. This yields the
                   IDSW *floor* of pure motion-based association: how many
                   identities Kalman+Hungarian loses even when detection is
                   flawless. Runnable today; it is NOT "AB3DMOT-PP".

  --dets <dir>     Real detection files (KITTI label format, one .txt per frame:
                   `cls -1 -1 -1 x1 y1 x2 y2 h w l x y z ry score`). Point this
                   at PointPillars detections on VoD radar to obtain the actual
                   AB3DMOT-PP figure RaTrack reports.

Scope matches the RaTrack example: moving objects only, RaTrack's validation
clips, and the SAME `idsw_eval` so the numbers are directly comparable.
"""
import argparse
import os
import sys
import numpy as np
from easydict import EasyDict

# repo helpers (parsers + the shared IDSW definition)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "ratrack"))
from ratrack_io import parse_gt_frame_moving, GtObject          # noqa: E402
from idsw_eval import idsw_report, summarize                    # noqa: E402

AB3DMOT_SRC = os.environ.get("AB3DMOT_SRC", "/ab3dmot")
sys.path.insert(0, AB3DMOT_SRC)
from AB3DMOT_libs.model import AB3DMOT                          # noqa: E402
from AB3DMOT_libs.dist_metrics import iou                       # noqa: E402
from AB3DMOT_libs.box import Box3D                              # noqa: E402

VOD_ROOT = os.environ.get("VOD_ROOT", "/project/view_of_delft_PUBLIC")
GT_TRK = os.path.join(VOD_ROOT, "lidar", "training", "label_2_tracking")
GT_DET = os.path.join(VOD_ROOT, "lidar", "training", "label_2")
CLIPS_DIR = os.path.join(_HERE, "..", "ratrack", "clips")
VAL_CLIPS = ["delft_1", "delft_10", "delft_14", "delft_22"]


def make_tracker(max_age=2, min_hits=1):
    """AB3DMOT configured CLASS-AGNOSTICALLY, to mirror RaTrack's setup.

    VoD is not one of AB3DMOT's built-in datasets, so we instantiate it via a
    known branch and then override the association parameters explicitly (they
    are plain attributes set by get_param). Values follow AB3DMOT's own KITTI
    defaults for a generic object.
    """
    cfg = EasyDict({
        "dataset": "KITTI",
        "det_name": "pointrcnn",
        "vis": False,
        "affi_pro": False,
        "ego_com": False,
    })
    # AB3DMOT logs unconditionally via print_log, which needs a real file handle
    # (it does log.write even when display=False), so send it to /dev/null.
    trk = AB3DMOT(cfg, cat="Car", log=open(os.devnull, "w"))

    # Class-agnostic association parameters, set explicitly rather than
    # inherited from a KITTI branch that does not fit VoD.
    #
    # Metric choice matters a lot here. AB3DMOT's default `giou_3d` FAILS on VoD:
    # most moving objects are VRUs whose boxes are only ~0.5-0.7 m wide, and with
    # a moving ego vehicle they shift ~1.4 m per frame in camera coordinates, so
    # consecutive boxes do not overlap at all -> affinity below threshold -> the
    # tracker re-births a new ID every frame. (AB3DMOT normally counters this
    # with `ego_com` ego-motion compensation, which needs KITTI-format oxts that
    # VoD does not ship - it stores poses as pose/*.json.)
    # Measured on delft_1 (first 200 frames): giou_3d spawns 56 spurious births
    # vs 13 for dist_3d, which is stable for thresholds in [-2, -10].
    trk.algm, trk.metric, trk.thres = "hungar", "dist_3d", -4
    trk.min_hits, trk.max_age = min_hits, max_age
    trk.max_sim, trk.min_sim = 0.0, -100.0
    return trk


def check_dataset():
    """Fail loudly if the dataset is not actually mounted.

    Without this, a wrong/missing volume mount silently yields zero GT objects,
    which the evaluator happily reports as "0 ID switches" — a wrong result that
    looks like a good one.
    """
    for name, d in (("tracking labels", GT_TRK), ("detection labels", GT_DET)):
        if not os.path.isdir(d):
            raise SystemExit(
                f"ERROR: {name} directory not found: {d}\n"
                f"       VOD_ROOT={VOD_ROOT}\n"
                f"       Mount the View-of-Delft dataset there "
                f"(see docker-compose.yml / VOD_HOST).")
        if not any(f.endswith(".txt") for f in os.listdir(d)):
            raise SystemExit(f"ERROR: {name} directory is empty: {d}")


def frame_ids(clip):
    with open(os.path.join(CLIPS_DIR, f"{clip}.txt")) as f:
        return [ln.strip() for ln in f if ln.strip()]


def gt_boxes(frame_id):
    """Moving GT objects for a frame (same filter as the RaTrack example)."""
    return parse_gt_frame_moving(
        os.path.join(GT_TRK, f"{frame_id}.txt"),
        os.path.join(GT_DET, f"{frame_id}.txt"))


def load_dets(frame_id, dets_dir):
    """Real detections in KITTI label format -> (Nx7 boxes, scores)."""
    path = os.path.join(dets_dir, f"{frame_id}.txt")
    boxes, scores = [], []
    if not os.path.exists(path):
        return np.zeros((0, 7)), np.zeros((0,))
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 15:
                continue
            h, w, l = float(p[8]), float(p[9]), float(p[10])
            x, y, z = float(p[11]), float(p[12]), float(p[13])
            ry = float(p[14])
            boxes.append([h, w, l, x, y, z, ry])
            scores.append(float(p[15]) if len(p) > 15 else 1.0)
    return np.asarray(boxes).reshape(-1, 7), np.asarray(scores)


def _bev_polygon(b7):
    """Bottom rectangle of a [h,w,l,x,y,z,ry] KITTI-camera box as a polygon
    in the ground (x,z) plane, using AB3DMOT's roty convention."""
    from shapely.geometry import Polygon
    h, w, l, x, y, z, ry = b7
    xs = np.array([l / 2, l / 2, -l / 2, -l / 2])
    zs = np.array([w / 2, -w / 2, -w / 2, w / 2])
    c, s = np.cos(ry), np.sin(ry)
    return Polygon(zip(c * xs + s * zs + x, -s * xs + c * zs + z))


def box_iou3d(a7, b7):
    """3D IoU between two [h,w,l,x,y,z,ry] boxes.

    Deliberately NOT AB3DMOT's own `dist_metrics.iou`: its convex-hull polygon
    clipping divides by zero when two box edges are exactly parallel, which is
    precisely what happens when GT boxes are fed as detections (tracker output
    coincides with the GT box) and yields NaN -> qhull crash. Shapely's
    intersection is numerically robust for that degenerate case.
    """
    pa, pb = _bev_polygon(a7), _bev_polygon(b7)
    if not (pa.is_valid and pb.is_valid):
        return 0.0
    inter_area = pa.intersection(pb).area
    if inter_area <= 0:
        return 0.0
    ha, hb = a7[0], b7[0]
    ya, yb = a7[4], b7[4]                      # y is the box bottom (camera coords)
    overlap_h = max(0.0, min(ya, yb) - max(ya - ha, yb - hb))
    inter = inter_area * overlap_h
    union = pa.area * ha + pb.area * hb - inter
    return float(inter / union) if union > 0 else 0.0


def run_clip(clip, dets_source, max_age=2):
    """Track one clip; return per-frame (gt_ids, pred_ids, conf, iou matrix)."""
    tracker = make_tracker(max_age=max_age)
    frames = []
    for fi, fid in enumerate(frame_ids(clip)):
        gts = gt_boxes(fid)
        gt_ids = [g.track_id for g in gts]
        gt_arr = np.asarray([[g.hwl[0], g.hwl[1], g.hwl[2],
                              g.center_cam[0], g.center_cam[1], g.center_cam[2],
                              g.ry] for g in gts]).reshape(-1, 7)

        if dets_source == "gt":
            dets, scores = gt_arr, np.ones(len(gt_arr))
        else:
            dets, scores = load_dets(fid, dets_source)

        info = np.zeros((len(dets), 1)) if len(dets) else np.zeros((0, 1))
        results, _ = tracker.track({"dets": dets, "info": info}, fi, clip)

        # track() returns [array] where each row is [h,w,l,x,y,z,ry, ID, info...]
        out = np.asarray(results[0]).reshape(-1, 9) if len(results) else np.zeros((0, 9))
        pred_boxes = out[:, :7]
        pred_ids = [int(v) for v in out[:, 7]] if len(out) else []

        m = np.zeros((len(gt_arr), len(pred_boxes)))
        for gi in range(len(gt_arr)):
            for pi in range(len(pred_boxes)):
                m[gi, pi] = box_iou3d(gt_arr[gi], pred_boxes[pi])
        frames.append((gt_ids, pred_ids, [1.0] * len(pred_ids), m))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", default="gt",
                    help="'gt' for perfect detections, or a directory of KITTI-format detections")
    ap.add_argument("--clips", nargs="*", default=VAL_CLIPS)
    ap.add_argument("--iou-thr", type=float, default=0.25)
    ap.add_argument("--max-age", type=int, default=2,
                    help="frames a track coasts without a match before it is killed")
    args = ap.parse_args()

    check_dataset()

    label = "GT boxes (perfect detection)" if args.dets == "gt" else args.dets
    print(f"AB3DMOT on VoD | detections: {label} | max_age={args.max_age}")
    all_frames = []
    for clip in args.clips:
        fr = run_clip(clip, args.dets, max_age=args.max_age)
        print(f"  {clip}: {len(fr)} frames")
        all_frames.extend(fr)

    total, per_track = idsw_report(all_frames, iou_thr=args.iou_thr)
    print(f"\n=== AB3DMOT ID switches (IoU {args.iou_thr}, moving-only GT) ===")
    summarize(total, per_track)
    if args.dets == "gt":
        print("\nNOTE: this is the IDSW FLOOR of motion-only association with\n"
              "perfect detections — NOT the AB3DMOT-PP figure (that needs\n"
              "PointPillars detections on VoD radar; pass them via --dets).")


if __name__ == "__main__":
    main()
