"""Measure what each fusion mode does to per-object feature discriminability.

The tracker's re-identification stage works by comparing object descriptors
across frames, so the question that decides whether a fusion mode is worth its
cost is: **do two observations of the same object look more alike to each other
than to other objects?** This script answers it directly, without training
anything.

Method
------
For every ground-truth *moving* object in a frame we gather the points that fall
inside its 3D box under a given fusion mode and reduce them to a fixed-length,
position-invariant descriptor:

* ``log(1 + n_points)`` — how much evidence the sensor returned
* extent along each axis, in the box's own frame
* the three eigenvalues of the point covariance, normalised — the standard
  linearity / planarity / sphericity shape signature
* mean and standard deviation of every feature channel (RCS, Doppler, and the
  LiDAR-derived channels when the mode provides them)

Position is deliberately excluded: a descriptor that encodes where the object is
would "re-identify" by location and tell us nothing about appearance.

Descriptors are z-scored across the corpus, then scored two ways:

* **Rank-1** — for each observation, is its nearest neighbour *in a different
  frame* the same track id? This is the standard re-identification protocol.
* **Silhouette** — how tightly observations of one identity group relative to
  other identities.

A mode only earns its complexity if it moves these numbers.

Usage
-----
    python examples/feature_analysis/analyze_features.py
    python examples/feature_analysis/analyze_features.py --modes none radar_base
    python examples/feature_analysis/analyze_features.py --stride 5 --max-frames 300
    python examples/feature_analysis/analyze_features.py --fiftyone   # + visual session

Results are written to `examples/feature_analysis/results/` (git-ignored).
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "examples", "ratrack"))

from external.vod.configuration import VodTrackLocations              # noqa: E402
from external.vod.frame import FrameDataLoader, FrameTransformMatrix   # noqa: E402
from external.vod.frame.transformations import homogeneous_transformation  # noqa: E402
from ratrack_io import parse_gt_frame_moving                          # noqa: E402
from point_iou import _gt_obb                                          # noqa: E402
from tracker.dataset.fusion_vod import fuse, MODES                     # noqa: E402

VOD_ROOT = os.environ.get("VOD_ROOT", "/project/view_of_delft_PUBLIC")
CLIPS_DIR = os.path.join(_REPO, "tracker", "dataset", "clips")
OUT_DIR = os.path.join(_HERE, "results")

TRAIN = ['delft_2', 'delft_3', 'delft_4', 'delft_6', 'delft_9', 'delft_11',
         'delft_12', 'delft_13', 'delft_19', 'delft_23', 'delft_24',
         'delft_26', 'delft_27']
VAL = ['delft_1', 'delft_10', 'delft_14', 'delft_22']

MIN_POINTS = 2          # RaTrack's own validity rule


def descriptor(points, feats, box_centre):
    """Fixed-length, position-invariant descriptor for one object observation."""
    p = points - box_centre                      # translate to the box, keep scale
    n = len(p)
    out = [np.log1p(n)]
    out += list(p.max(axis=0) - p.min(axis=0)) if n > 1 else [0.0, 0.0, 0.0]

    # shape signature: normalised covariance eigenvalues
    if n >= 3:
        ev = np.linalg.eigvalsh(np.cov(p.T) + 1e-9 * np.eye(3))
        ev = np.sort(np.abs(ev))[::-1]
        ev = ev / (ev.sum() + 1e-9)
    else:
        ev = np.zeros(3)
    out += list(ev)

    # per-channel statistics
    if feats is not None and feats.size:
        out += list(feats.mean(axis=0)) + list(feats.std(axis=0))
    return np.asarray(out, dtype=np.float64)


def collect(mode, clips, stride, max_frames, radius):
    """Descriptors + labels for one fusion mode."""
    loc = VodTrackLocations(root_dir=VOD_ROOT, output_dir=VOD_ROOT,
                            frame_set_path="", pred_dir="")
    import open3d as o3d

    X, ids, frames, classes = [], [], [], []
    seen = 0
    for clip in clips:
        path = os.path.join(CLIPS_DIR, f"{clip}.txt")
        for fid in [l.strip() for l in open(path) if l.strip()][::stride]:
            if max_frames and seen >= max_frames:
                break
            try:
                gts = parse_gt_frame_moving(
                    os.path.join(VOD_ROOT, "lidar", "training",
                                 "label_2_tracking", f"{fid}.txt"),
                    os.path.join(VOD_ROOT, "lidar", "training",
                                 "label_2", f"{fid}.txt"))
                if not gts:
                    continue
                fd = FrameDataLoader(kitti_locations=loc, frame_number=fid)
                tf = FrameTransformMatrix(fd)
                radar, lidar = fd.radar_data, fd.lidar_data
                if radar is None or lidar is None:
                    continue

                lid = lidar[:, :3]
                hom = np.hstack([lid, np.ones((len(lid), 1))])
                lid_r = homogeneous_transformation(hom, tf.t_radar_lidar)[:, :3]

                pts, feats = fuse(radar[:, :3], radar[:, 3:6], lid_r,
                                  mode=mode, radius=radius, max_points=None)

                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(pts)
                for g in gts:
                    idx = _gt_obb(g, tf.t_radar_camera, tf.t_radar_lidar) \
                        .get_point_indices_within_bounding_box(cloud.points)
                    if len(idx) < MIN_POINTS:
                        continue
                    idx = np.asarray(idx)
                    centre = pts[idx].mean(axis=0)
                    X.append(descriptor(pts[idx], feats[idx], centre))
                    ids.append(g.track_id)
                    frames.append(fid)
                    classes.append(g.cls)
                seen += 1
            except Exception:
                continue
        if max_frames and seen >= max_frames:
            break
    return np.asarray(X), np.asarray(ids), np.asarray(frames), np.asarray(classes)


def rank1(X, ids, frames):
    """Fraction of observations whose nearest cross-frame neighbour shares the id."""
    if len(X) < 2:
        return float("nan")
    # z-score so channels with large units do not dominate the distance
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    d = ((Xz[:, None, :] - Xz[None, :, :]) ** 2).sum(-1)
    same_frame = frames[:, None] == frames[None, :]
    np.fill_diagonal(d, np.inf)
    d[same_frame] = np.inf                 # a match in the same frame is trivial
    nn = d.argmin(1)
    valid = np.isfinite(d.min(1))
    if valid.sum() == 0:
        return float("nan")
    return float((ids[nn][valid] == ids[valid]).mean())


def silhouette(X, ids):
    from sklearn.metrics import silhouette_score
    keep = np.isin(ids, [i for i in set(ids) if (ids == i).sum() >= 2])
    if keep.sum() < 3 or len(set(ids[keep])) < 2:
        return float("nan")
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return float(silhouette_score(Xz[keep], ids[keep]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="*", default=list(MODES))
    ap.add_argument("--split", choices=["train", "val", "both"], default="val")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--radius", type=float, default=0.5)
    ap.add_argument("--no-common", action="store_true",
                    help="score each mode on its own corpus instead of the "
                         "observations shared by all modes (not comparable)")
    ap.add_argument("--fiftyone", action="store_true",
                    help="also build a FiftyOne dataset with the embeddings")
    a = ap.parse_args()

    if not os.path.isdir(VOD_ROOT):
        raise SystemExit(f"ERROR: dataset not found at {VOD_ROOT}. "
                         f"Set VOD_ROOT or mount View-of-Delft there.")

    clips = {"train": TRAIN, "val": VAL, "both": TRAIN + VAL}[a.split]
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"split={a.split}  stride={a.stride}  max_frames={a.max_frames}  "
          f"radius={a.radius}m\n")
    header = f"  {'mode':12s} {'observations':>13s} {'identities':>11s} " \
             f"{'dim':>4s} {'rank-1':>8s} {'silhouette':>11s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Collect every mode first, then score on the SAME observations.
    # Modes differ in how many objects clear the MIN_POINTS threshold — the
    # LiDAR-based ones see more — and rank-1 gets harder as identities are
    # added, so scoring each on its own corpus would compare different tasks.
    raw = {}
    for mode in a.modes:
        raw[mode] = collect(mode, clips, a.stride, a.max_frames, a.radius)

    common = None
    if not a.no_common:
        for mode, (X, ids, frames, _) in raw.items():
            keys = set(zip(frames.tolist(), ids.tolist())) if len(X) else set()
            common = keys if common is None else (common & keys)
        print(f"  common observations across modes: {len(common or [])}\n")

    results = {}
    for mode in a.modes:
        X, ids, frames, classes = raw[mode]
        if len(X) and common is not None:
            keep = np.array([(f, i) in common
                             for f, i in zip(frames.tolist(), ids.tolist())])
            X, ids, frames, classes = X[keep], ids[keep], frames[keep], classes[keep]
        if len(X) == 0:
            print(f"  {mode:12s} {'no data':>13s}")
            continue
        r1, sil = rank1(X, ids, frames), silhouette(X, ids)
        results[mode] = {"observations": int(len(X)),
                         "identities": int(len(set(ids))),
                         "dim": int(X.shape[1]),
                         "rank1": r1, "silhouette": sil}
        print(f"  {mode:12s} {len(X):13d} {len(set(ids)):11d} "
              f"{X.shape[1]:4d} {100*r1:7.1f}% {sil:11.3f}")

        if a.fiftyone:
            export_fiftyone(mode, X, ids, frames, classes)

    with open(os.path.join(OUT_DIR, "feature_analysis.json"), "w") as f:
        json.dump({"config": vars(a), "results": results}, f, indent=2)
    print(f"\n  -> {os.path.join(OUT_DIR, 'feature_analysis.json')}")

    if results:
        base = results.get("none", {}).get("rank1")
        if base is not None and np.isfinite(base):
            print("\n  rank-1 relative to the radar-only baseline:")
            for m, r in results.items():
                if m == "none" or not np.isfinite(r["rank1"]):
                    continue
                print(f"    {m:12s} {100*(r['rank1'] - base):+6.1f} points")


def export_fiftyone(mode, X, ids, frames, classes):
    """Optional: a FiftyOne dataset with a 2D projection of the descriptors."""
    try:
        import fiftyone as fo
        import fiftyone.brain as fob
    except ImportError:
        print(f"    [fiftyone] not installed — skipping export for {mode}")
        return

    name = f"vod-features-{mode}"
    if name in fo.list_datasets():
        fo.delete_dataset(name)
    ds = fo.Dataset(name, persistent=True)
    samples = []
    for i in range(len(X)):
        img = os.path.join(VOD_ROOT, "lidar", "training", "image_2",
                           f"{frames[i]}.jpg")
        s = fo.Sample(filepath=img if os.path.exists(img) else __file__)
        s["track_id"] = int(ids[i])
        s["frame"] = str(frames[i])
        s["cls"] = str(classes[i])
        s["fusion_mode"] = mode
        samples.append(s)
    ds.add_samples(samples)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    fob.compute_visualization(ds, embeddings=Xz, brain_key="descriptors",
                              method="umap", verbose=False)
    print(f"    [fiftyone] dataset '{name}' ready — `fiftyone app launch {name}`")


if __name__ == "__main__":
    main()
