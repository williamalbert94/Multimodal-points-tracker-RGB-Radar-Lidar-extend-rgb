"""Project 3D ground-truth boxes into the camera image.

Complements `check_dataloader.py`: instead of verifying the BEV geometry, this
verifies the **camera projection chain** — box (camera frame) -> LiDAR frame ->
camera frame -> image plane, using the frame's own calibration.

Two panels per frame:

1. **3D boxes on RGB** — every annotated object drawn as a projected wireframe.
   Moving objects (the ones the tracker is scored on) are green with their track
   id; static objects are grey. If the calibration or the rotation convention is
   wrong the wireframes drift off the objects, which is immediately visible.
2. **Radar points on RGB** — the radar sweep projected into the same image,
   coloured by range. This cross-checks the radar->camera extrinsic that the BEV
   panels cannot show.

The projection math mirrors the VoD devkit's `get_2d_label_corners`, extended to
carry the track id and moving flag through (the devkit sorts boxes by range and
drops those fields).

Usage (inside the container):

    python examples/dataloader/check_projection.py --split both
    python examples/dataloader/check_projection.py --split val --limit 20

Figures go to `examples/dataloader/check_projection/<split>/`, which is
git-ignored.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from external.vod.configuration import VodTrackLocations          # noqa: E402
from external.vod.frame import FrameDataLoader, FrameTransformMatrix  # noqa: E402
from external.vod.frame.transformations import (                  # noqa: E402
    homogeneous_transformation, project_pcl_to_image)

DEFAULT_ROOT = os.environ.get("VOD_ROOT", "/project/view_of_delft_PUBLIC")
OUT_DIR = os.path.join(_HERE, "check_projection")

TRAIN = ['delft_2', 'delft_3', 'delft_4', 'delft_6', 'delft_9', 'delft_11',
         'delft_12', 'delft_13', 'delft_19', 'delft_23', 'delft_24',
         'delft_26', 'delft_27']
VAL = ['delft_1', 'delft_10', 'delft_14', 'delft_22']

# corner order: 0-3 bottom face, 4-7 top face
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),        # bottom
         (4, 5), (5, 6), (6, 7), (7, 4),        # top
         (0, 4), (1, 5), (2, 6), (3, 7)]        # verticals


def parse_labels(tracking_lines, detection_lines):
    """Geometry + track id + moving flag for one frame.

    VoD ships `label_2` and `label_2_tracking` row-aligned; column 1 is the
    moving flag in the former and the track id in the latter, so the two must be
    zipped positionally to recover both.
    """
    out = []
    det = detection_lines or []
    for i, line in enumerate(tracking_lines or []):
        f = line.split()
        if len(f) < 15:
            continue
        moving = False
        if i < len(det):
            df = det[i].split()
            moving = len(df) > 1 and int(float(df[1])) == 1
        out.append({
            "cls": f[0],
            "track_id": int(float(f[1])),
            "h": float(f[8]), "w": float(f[9]), "l": float(f[10]),
            "x": float(f[11]), "y": float(f[12]), "z": float(f[13]),
            "ry": float(f[14]),
            "moving": moving,
        })
    return out


def project_box(label, tf):
    """8 box corners projected to image pixels; None if fully behind the camera.

    Mirrors the devkit's `get_2d_label_corners`: build corners in the box frame,
    rotate by -(ry + pi/2), place at the centre expressed in the LiDAR frame,
    transform back to camera, then apply the projection matrix.
    """
    l, w, h = label["l"], label["w"], label["h"]
    corners = np.array([
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
        [0, 0, 0, 0, h, h, h, h],
    ])
    rot = -(label["ry"] + np.pi / 2)
    R = np.array([[np.cos(rot), -np.sin(rot), 0],
                  [np.sin(rot), np.cos(rot), 0],
                  [0, 0, 1]])
    centre = (tf.t_lidar_camera @ np.array([label["x"], label["y"],
                                            label["z"], 1.0]))[:3]
    pts = (R @ corners).T + centre
    hom = np.concatenate((pts, np.ones((8, 1))), axis=1)
    hom = homogeneous_transformation(hom, tf.t_camera_lidar)

    depth = hom[:, 2]
    if np.all(depth <= 0.1):                 # entirely behind the camera
        return None, depth
    img = hom @ tf.camera_projection_matrix.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (img[:, :2].T / img[:, 2]).T
    return uv, depth


def draw_boxes(ax, labels, tf, img_shape):
    h_img, w_img = img_shape[:2]
    drawn = 0
    for lab in sorted(labels, key=lambda d: -(d["x"] ** 2 + d["z"] ** 2)):
        uv, depth = project_box(lab, tf)
        if uv is None:
            continue
        # skip boxes whose pixels land far outside the canvas
        if uv[:, 0].max() < -w_img or uv[:, 0].min() > 2 * w_img:
            continue
        colour = "#2ecc71" if lab["moving"] else "#7f8c8d"
        lw = 1.6 if lab["moving"] else 0.9
        for a, b in EDGES:
            if depth[a] <= 0.1 or depth[b] <= 0.1:
                continue                       # edge crosses the image plane
            ax.plot([uv[a, 0], uv[b, 0]], [uv[a, 1], uv[b, 1]],
                    color=colour, linewidth=lw, zorder=4)
        if lab["moving"]:
            top = uv[4:][np.argmin(uv[4:, 1])]
            if 0 <= top[0] <= w_img and -50 <= top[1] <= h_img:
                ax.text(top[0], top[1] - 6, f"{lab['cls']} #{lab['track_id']}",
                        color="#2ecc71", fontsize=6, zorder=6,
                        bbox=dict(facecolor="black", alpha=0.45,
                                  pad=0.8, edgecolor="none"))
        drawn += 1
    return drawn


def render(frame_id, clip, fd, tf, out_path):
    img = fd.image
    if img is None:
        return False
    labels = parse_labels(fd.raw_tracking_labels, fd.raw_detection_labels)
    n_mov = sum(1 for lab in labels if lab["moving"])

    # Dedicated narrow column for the colourbar, so it does not shrink the
    # second image and leave the two panels misaligned.
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[40, 1], hspace=0.12, wspace=0.02)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
    cax = fig.add_subplot(gs[1, 1])

    ax = axes[0]
    ax.imshow(img)
    drawn = draw_boxes(ax, labels, tf, img.shape)
    ax.set_title(f"3D GT boxes projected to camera — "
                 f"{n_mov} moving (green) / {len(labels) - n_mov} static (grey)",
                 fontsize=10)
    ax.axis("off")

    ax = axes[1]
    ax.imshow(img)
    radar = fd.radar_data
    if radar is not None and len(radar):
        uvs, depth = project_pcl_to_image(
            point_cloud=radar,
            t_camera_pcl=tf.t_camera_radar,
            camera_projection_matrix=tf.camera_projection_matrix,
            image_shape=img.shape)
        if len(uvs):
            s = ax.scatter(uvs[:, 0], uvs[:, 1], c=depth, s=9,
                           cmap="jet_r", alpha=0.85, zorder=3)
            fig.colorbar(s, cax=cax, label="range (m)")
        else:
            cax.axis("off")
    else:
        cax.axis("off")
    ax.set_title("Radar points projected to camera (colour = range)", fontsize=10)
    ax.axis("off")

    fig.suptitle(f"{clip} — frame {frame_id} — {drawn} boxes projected",
                 fontsize=12)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return True


def frames_of(clip):
    path = os.path.join(_REPO, "tracker", "dataset", "clips", f"{clip}.txt")
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def run_split(split, root, limit, stride):
    clips = TRAIN if split == "train" else VAL
    out = os.path.join(OUT_DIR, split)
    os.makedirs(out, exist_ok=True)
    loc = VodTrackLocations(root_dir=root, output_dir=root,
                            frame_set_path="", pred_dir="")

    done = failed = 0
    for clip in clips:
        ids = frames_of(clip)[::stride]
        for fid in ids:
            if limit and done >= limit:
                break
            try:
                fd = FrameDataLoader(kitti_locations=loc, frame_number=fid)
                tf = FrameTransformMatrix(fd)
                if render(fid, clip, fd, tf,
                          os.path.join(out, f"{clip}_{fid}.png")):
                    done += 1
                    if done % 25 == 0:
                        print(f"  [{split}] {done} figuras", flush=True)
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"  [{split}] {clip}/{fid} falló: "
                          f"{type(e).__name__}: {str(e)[:80]}", flush=True)
        if limit and done >= limit:
            break
    print(f"  [{split}] listo: {done} figuras en {out} ({failed} fallos)")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "both"], default="both")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()

    if not os.path.isdir(a.root):
        raise SystemExit(f"ERROR: dataset not found at {a.root}. "
                         f"Mount View-of-Delft there or pass --root.")

    total = 0
    for s in (["train", "val"] if a.split == "both" else [a.split]):
        total += run_split(s, a.root, a.limit, a.stride)
    print(f"\nTotal: {total} figuras bajo {OUT_DIR}/")


if __name__ == "__main__":
    main()
