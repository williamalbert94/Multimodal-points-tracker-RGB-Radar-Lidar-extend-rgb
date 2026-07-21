"""Ablation: AB3DMOT's association on RaTrack's detections.

  "AB3DMOT + RaTrack detector/segmentor"

RaTrack and AB3DMOT differ in TWO places at once: the detector (per-frame DBSCAN
clustering vs PointPillars boxes) and the associator (learned affinity + Sinkhorn
vs 3D Kalman + Hungarian). Comparing them end-to-end therefore cannot attribute
RaTrack's high ID-switch count to either part.

This script holds the detector fixed — both arms consume RaTrack's own exported
clusters — and swaps ONLY the association algorithm:

    arm A  = RaTrack detections + RaTrack association   (the ids in its .txt)
    arm B  = RaTrack detections + AB3DMOT association   (Kalman + Hungarian)

Both arms are scored against the same GT with the same point-based IoU and the
same IDSW definition, so any difference is attributable to association alone.

Detections are fed to AB3DMOT as cluster centroids with the cluster's own extent.
Association uses `dist_3d` (centre distance), so the box dimensions never enter
the matching — which matters because RaTrack deliberately avoids box regression
on sparse radar, and fitting boxes to ~6-point clusters would be meaningless.
Everything stays in the radar frame; AB3DMOT is frame-agnostic.
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "ratrack"))
from ratrack_io import parse_prediction_frame, parse_gt_frame_moving   # noqa: E402
from idsw_eval import idsw_report                                      # noqa: E402
from point_iou import (load_radar_cloud, frame_transforms, _gt_obb,    # noqa: E402
                       _cluster_indices)
from run_ab3dmot import make_tracker                                   # noqa: E402

VOD_ROOT = os.environ.get("VOD_ROOT", "/project/view_of_delft_PUBLIC")
GT_TRK = os.path.join(VOD_ROOT, "lidar", "training", "label_2_tracking")
GT_DET = os.path.join(VOD_ROOT, "lidar", "training", "label_2")
RATRACK_RESULTS = os.environ.get(
    "RATRACK_RESULT_DIR", "/ratrack_results")
CLIPS_DIR = os.path.join(_HERE, "..", "ratrack", "clips")
VAL_CLIPS = ["delft_1", "delft_10", "delft_14", "delft_22"]
MIN_OBJ_POINTS = 2          # RaTrack's default validity rule
IOU_THR = 0.25


def cluster_to_det(points):
    """Cluster points (N,3, radar frame) -> AB3DMOT box [h,w,l,x,y,z,ry].

    Centroid + axis-aligned extent, with a floor on the size so degenerate
    (1-2 point) clusters stay valid boxes. Only the centre is used for
    association under dist_3d.
    """
    c = points.mean(axis=0)
    ext = points.max(axis=0) - points.min(axis=0)
    l, w, h = np.maximum(ext, 0.5)
    return [h, w, l, c[0], c[1], c[2], 0.0]


def run_clip(clip, max_age=2):
    """Return per-frame records with both arms' id assignments."""
    import open3d as o3d

    clip_dir = os.path.join(RATRACK_RESULTS, clip)
    tracker = make_tracker(max_age=max_age)
    frames = []

    for fi, fn in enumerate(sorted(f for f in os.listdir(clip_dir)
                                   if f.endswith(".txt"))):
        fid = fn[:-4]
        clusters = [p for p in parse_prediction_frame(os.path.join(clip_dir, fn))
                    if p.points.shape[0] >= MIN_OBJ_POINTS]

        # ---- shared GT matching (identical for both arms) ----
        gts = parse_gt_frame_moving(os.path.join(GT_TRK, f"{fid}.txt"),
                                    os.path.join(GT_DET, f"{fid}.txt"))
        radar = load_radar_cloud(fid)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(radar)
        t_rc, t_rl = frame_transforms(fid)
        gt_keep, gt_sets = [], []
        for g in gts:
            idx = set(_gt_obb(g, t_rc, t_rl)
                      .get_point_indices_within_bounding_box(cloud.points))
            if len(idx) >= MIN_OBJ_POINTS:
                gt_keep.append(g)
                gt_sets.append(idx)
        cl_sets = [_cluster_indices(c.points, radar) for c in clusters]
        iou = np.zeros((len(gt_keep), len(clusters)))
        for gi, A in enumerate(gt_sets):
            for ci, B in enumerate(cl_sets):
                u = len(A | B)
                iou[gi, ci] = len(A & B) / u if u else 0.0

        # ---- arm B: AB3DMOT association over the same clusters ----
        dets = np.asarray([cluster_to_det(c.points) for c in clusters]).reshape(-1, 7)
        info = np.zeros((len(dets), 1))
        results, _ = tracker.track({"dets": dets, "info": info}, fi, clip)
        out = np.asarray(results[0]).reshape(-1, 9) if len(results) else np.zeros((0, 9))

        # map AB3DMOT tracks back onto the detections that produced them
        ab_ids = [-1] * len(clusters)
        if len(out):
            for row in out:
                centre, tid = row[3:6], int(row[7])
                if len(dets):
                    d = np.linalg.norm(dets[:, 3:6] - centre, axis=1)
                    j = int(np.argmin(d))
                    if d[j] < 2.0:            # Kalman output sits on its detection
                        ab_ids[j] = tid

        frames.append({
            "gt_ids": [g.track_id for g in gt_keep],
            "iou": iou,
            "ratrack_ids": [c.track_id for c in clusters],
            "ab3dmot_ids": ab_ids,
            # real per-cluster confidences: needed for the AMOTA recall sweep
            "conf": [float(c.conf) for c in clusters],
        })
    return frames


def to_eval(frames, key):
    """Build evaluator input, dropping detections the tracker produced no track
    for (id == -1).

    These must be REMOVED, not passed through: leaving -1 in the prediction list
    makes the evaluator treat "no track" as a legitimate id, so a GT object
    repeatedly matched to -1 looks perfectly stable and its real switches vanish.
    That silently understates IDSW — badly so for configs with min_hits > 1,
    where many detections never surface as tracks.
    """
    out = []
    for f in frames:
        keep = [i for i, tid in enumerate(f[key]) if tid != -1]
        ids = [f[key][i] for i in keep]
        conf = [f["conf"][i] for i in keep]
        iou = f["iou"][:, keep] if f["iou"].size else np.zeros((len(f["gt_ids"]), 0))
        out.append((f["gt_ids"], ids, conf, iou))
    return out


def main():
    all_frames = []
    for clip in VAL_CLIPS:
        fr = run_clip(clip)
        print(f"  {clip}: {len(fr)} frames")
        all_frames.extend(fr)

    print("\n=== Ablation: same detections, different association ===")
    print(f"    (RaTrack clusters, min_obj_points={MIN_OBJ_POINTS}, "
          f"point-IoU {IOU_THR}, moving-only GT)\n")
    for label, key in (("RaTrack association (learned affinity + Sinkhorn)", "ratrack_ids"),
                       ("AB3DMOT association (3D Kalman + Hungarian)", "ab3dmot_ids")):
        total, per_track = idsw_report(to_eval(all_frames, key), iou_thr=IOU_THR)
        est = len(per_track)
        sw = sum(1 for v in per_track.values() if v)
        print(f"  {label}")
        print(f"      established GT tracks : {est}")
        print(f"      tracks with >=1 switch: {sw}")
        print(f"      total ID switches     : {total}\n")


if __name__ == "__main__":
    main()
