"""3D box projection with near-plane clipping, constrained to the image canvas.

Same output as `check_projection.py`, with the reprojection blow-up fixed. See
`docs/pages/reprojection.md` for the failure it addresses: when an object passes
within a couple of metres of the camera some corners sit at near-zero depth, and
the perspective division `u = f_x X_c / Z_c + c_x` throws them tens of thousands
of pixels off-canvas — matplotlib then autoscales and squeezes the photo into a
thumbnail.

Two corrections, applied in this order:

1. **Near-plane clipping (in 3D, before projecting).** Each box edge is
   intersected with the plane `Z_c = NEAR` and only the visible part is kept:

       P(t) = P1 + t (P2 - P1),   t = (NEAR - z1) / (z2 - z1)

   Clipping *before* the division is what makes this correct — clamping pixel
   coordinates afterwards would bend edges towards the wrong place, because the
   projection of a point behind the camera is not a point in front of it.
   This is what a rasteriser's near plane does.

2. **Canvas clamp.** Axes limits are pinned to the image extent and artists are
   drawn with `clip_on=True`, so a surviving stray coordinate can no longer
   rescale the photo out of view.

Usage:

    python examples/dataloader/check_projection_adjusted.py --split both
    python examples/dataloader/check_projection_adjusted.py --split val --limit 20
    python examples/dataloader/check_projection_adjusted.py --split train --near 1.0

Figures go to `examples/dataloader/check_projection_adjusted/<split>/`
(git-ignored).
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

from external.vod.configuration import VodTrackLocations              # noqa: E402
from external.vod.frame import FrameDataLoader, FrameTransformMatrix  # noqa: E402
from external.vod.frame.transformations import (                      # noqa: E402
    homogeneous_transformation, project_pcl_to_image)

sys.path.insert(0, _HERE)
from check_projection import parse_labels, EDGES, frames_of, TRAIN, VAL  # noqa: E402

DEFAULT_ROOT = os.environ.get("VOD_ROOT", "/project/view_of_delft_PUBLIC")
OUT_DIR = os.path.join(_HERE, "check_projection_adjusted")

# Near plane in metres. Anything closer is clipped away; nothing useful is
# visible within half a metre of the lens anyway.
NEAR_DEFAULT = 0.5


def box_corners_camera(label, tf):
    """The box's 8 corners expressed in the CAMERA frame, before projection.

    Splitting this out (rather than going straight to pixels as
    `check_projection.py` does) is what allows clipping in 3D, where it is
    geometrically meaningful.
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
    return homogeneous_transformation(hom, tf.t_camera_lidar)[:, :3]


def clip_segment_near(p1, p2, near):
    """Clip a 3D segment against the half-space Z_c >= near.

    Returns the visible sub-segment, or None when the edge lies entirely behind
    the near plane.
    """
    z1, z2 = p1[2], p2[2]
    if z1 >= near and z2 >= near:
        return p1, p2
    if z1 < near and z2 < near:
        return None
    t = (near - z1) / (z2 - z1)          # z1 != z2 here, they straddle `near`
    q = p1 + t * (p2 - p1)
    return (q, p2) if z1 < near else (p1, q)


def project(points, P):
    """Perspective projection of camera-frame points already known to be in front."""
    hom = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    img = hom @ P.T
    return (img[:, :2].T / img[:, 2]).T


def draw_boxes(ax, labels, tf, img_shape, near):
    h_img, w_img = img_shape[:2]
    drawn = clipped = 0
    for lab in sorted(labels, key=lambda d: -(d["x"] ** 2 + d["z"] ** 2)):
        cam = box_corners_camera(lab, tf)
        colour = "#2ecc71" if lab["moving"] else "#7f8c8d"
        lw = 1.6 if lab["moving"] else 0.9

        any_edge = was_clipped = False
        for a, b in EDGES:
            seg = clip_segment_near(cam[a], cam[b], near)
            if seg is None:
                was_clipped = True
                continue
            if not np.allclose(seg[0], cam[a]) or not np.allclose(seg[1], cam[b]):
                was_clipped = True
            uv = project(np.vstack(seg), tf.camera_projection_matrix)
            ax.plot(uv[:, 0], uv[:, 1], color=colour, linewidth=lw,
                    zorder=4, clip_on=True)
            any_edge = True

        if any_edge:
            drawn += 1
            clipped += int(was_clipped)
            if lab["moving"]:
                front = cam[cam[:, 2] >= near]
                if len(front):
                    uv = project(front, tf.camera_projection_matrix)
                    top = uv[np.argmin(uv[:, 1])]
                    if 0 <= top[0] <= w_img and 0 <= top[1] <= h_img:
                        ax.text(top[0], top[1] - 6,
                                f"{lab['cls']} #{lab['track_id']}",
                                color="#2ecc71", fontsize=6, zorder=6,
                                clip_on=True,
                                bbox=dict(facecolor="black", alpha=0.45,
                                          pad=0.8, edgecolor="none"))
    return drawn, clipped


def lock_to_canvas(ax, img_shape):
    """Pin the axes to the image extent so no artist can rescale it away."""
    h, w = img_shape[:2]
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)          # image convention: y grows downward
    ax.set_aspect("equal")
    ax.axis("off")


def render(frame_id, clip, fd, tf, out_path, near):
    img = fd.image
    if img is None:
        return False
    labels = parse_labels(fd.raw_tracking_labels, fd.raw_detection_labels)
    n_mov = sum(1 for lab in labels if lab["moving"])

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[40, 1], hspace=0.12, wspace=0.02)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
    cax = fig.add_subplot(gs[1, 1])

    ax = axes[0]
    ax.imshow(img)
    drawn, clipped = draw_boxes(ax, labels, tf, img.shape, near)
    lock_to_canvas(ax, img.shape)
    ax.set_title(f"3D GT boxes, near-plane clipped at {near} m — "
                 f"{n_mov} moving (green) / {len(labels) - n_mov} static (grey)"
                 + (f" — {clipped} clipped" if clipped else ""), fontsize=10)

    ax = axes[1]
    ax.imshow(img)
    radar = fd.radar_data
    if radar is not None and len(radar):
        uvs, depth = project_pcl_to_image(
            point_cloud=radar, t_camera_pcl=tf.t_camera_radar,
            camera_projection_matrix=tf.camera_projection_matrix,
            image_shape=img.shape)
        if len(uvs):
            s = ax.scatter(uvs[:, 0], uvs[:, 1], c=depth, s=9, cmap="jet_r",
                           alpha=0.85, zorder=3, clip_on=True)
            fig.colorbar(s, cax=cax, label="range (m)")
        else:
            cax.axis("off")
    else:
        cax.axis("off")
    lock_to_canvas(ax, img.shape)
    ax.set_title("Radar points projected to camera (colour = range)", fontsize=10)

    fig.suptitle(f"{clip} — frame {frame_id} — {drawn} boxes projected",
                 fontsize=12)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return True


def run_split(split, root, limit, stride, near):
    clips = TRAIN if split == "train" else VAL
    out = os.path.join(OUT_DIR, split)
    os.makedirs(out, exist_ok=True)
    loc = VodTrackLocations(root_dir=root, output_dir=root,
                            frame_set_path="", pred_dir="")
    done = failed = 0
    for clip in clips:
        for fid in frames_of(clip)[::stride]:
            if limit and done >= limit:
                break
            try:
                fd = FrameDataLoader(kitti_locations=loc, frame_number=fid)
                tf = FrameTransformMatrix(fd)
                if render(fid, clip, fd, tf,
                          os.path.join(out, f"{clip}_{fid}.png"), near):
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
    ap.add_argument("--near", type=float, default=NEAR_DEFAULT,
                    help="near-plane distance in metres (default 0.5)")
    a = ap.parse_args()

    if not os.path.isdir(a.root):
        raise SystemExit(f"ERROR: dataset not found at {a.root}. "
                         f"Mount View-of-Delft there or pass --root.")

    total = 0
    for s in (["train", "val"] if a.split == "both" else [a.split]):
        total += run_split(s, a.root, a.limit, a.stride, a.near)
    print(f"\nTotal: {total} figuras bajo {OUT_DIR}/")


if __name__ == "__main__":
    main()
