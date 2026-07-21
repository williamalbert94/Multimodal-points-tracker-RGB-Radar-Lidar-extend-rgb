"""Visual sanity check for the View-of-Delft dataloader.

For every frame of the train and validation splits this renders one figure with
three panels:

1. **Radar/LiDAR alignment (BEV)** — radar and LiDAR points overlaid in the radar
   frame. If the extrinsic transform is right the two modalities land on the same
   structures; a constant offset between them means the calibration is wrong.
2. **Ego-motion compensation (BEV)** — the current radar sweep against the same
   sweep re-expressed in the next frame's pose. Static structure should collapse
   onto itself; residual drift means `T_ego` is wrong.
3. **LiDAR BEV height map + RGB** — the LiDAR sweep rasterised to a top-down
   height image, next to the camera frame for reference.

Ground-truth moving boxes are drawn on the BEV panels.

Usage (inside the container):

    python examples/dataloader/check_dataloader.py --split val
    python examples/dataloader/check_dataloader.py --split train --limit 50
    python examples/dataloader/check_dataloader.py --split both

Figures go to `examples/dataloader/check/<split>/` which is git-ignored.
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tracker.dataset import TrackingDataVOD                      # noqa: E402
from external.vod.configuration import VodTrackLocations         # noqa: E402
from external.vod.frame import FrameDataLoader                   # noqa: E402

DEFAULT_ROOT = os.environ.get("VOD_ROOT", "/project/view_of_delft_PUBLIC")
CHECK_DIR = os.path.join(_HERE, "check")

# BEV raster settings (metres)
BEV_SIDE = (-25.0, 25.0)     # lateral
BEV_FWD = (0.0, 50.0)        # forward
BEV_RES = 0.15               # metres per pixel


def box_corners_bev(label, transforms):
    """GT box footprint in the radar frame, as a (4,2) array of (x, y)."""
    from scipy.spatial.transform import Rotation as R
    c = (transforms.t_radar_camera @ np.array([label.x, label.y, label.z, 1.0]))[:3]
    rot = R.from_euler("XYZ", [0, 0, -(label.ry + np.pi / 2)]).as_matrix()
    rot = transforms.t_radar_lidar[:3, :3] @ rot
    l, w = label.l, label.w
    local = np.array([[l / 2, w / 2, 0], [l / 2, -w / 2, 0],
                      [-l / 2, -w / 2, 0], [-l / 2, w / 2, 0]])
    return (rot @ local.T).T[:, :2] + c[:2]


def draw_boxes(ax, labels, transforms, color="lime"):
    for lbl in labels.values():
        try:
            corners = box_corners_bev(lbl, transforms)
        except Exception:
            continue
        ax.add_patch(MplPolygon(corners, closed=True, fill=False,
                                edgecolor=color, linewidth=1.2, zorder=5))


def lidar_bev_image(pts):
    """Rasterise a LiDAR sweep to a top-down height map (uint8)."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    keep = ((x > BEV_FWD[0]) & (x < BEV_FWD[1]) &
            (y > BEV_SIDE[0]) & (y < BEV_SIDE[1]))
    x, y, z = x[keep], y[keep], z[keep]
    w = int((BEV_SIDE[1] - BEV_SIDE[0]) / BEV_RES)
    h = int((BEV_FWD[1] - BEV_FWD[0]) / BEV_RES)
    img = np.zeros((h, w), dtype=np.float32)
    if len(x) == 0:
        return img
    col = ((y - BEV_SIDE[0]) / BEV_RES).astype(np.int32).clip(0, w - 1)
    row = ((x - BEV_FWD[0]) / BEV_RES).astype(np.int32).clip(0, h - 1)
    zc = np.clip(z, -3.0, 2.0)
    np.maximum.at(img, (row, col), (zc + 3.0) / 5.0)
    return img


def style_bev(ax, title):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X — forward (m)")
    ax.set_ylabel("Y — lateral (m)")
    ax.set_xlim(BEV_FWD)
    ax.set_ylim(BEV_SIDE)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)


def render(sample, rgb, out_path):
    (pc0, pc1, ft0, ft1, pc0_comp, curr_idx, clip, ego,
     pc_last_l, pc0_l, pc1_l, new_seq, lbl1, lbl2, tf1, tf2, motion) = sample

    fig = plt.figure(figsize=(17, 5.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.15], wspace=0.28)

    # 1) radar vs lidar alignment
    ax = fig.add_subplot(gs[0, 0])
    sub = pc0_l[::40]
    ax.scatter(sub[:, 0], sub[:, 1], s=0.4, c="#9bb8d3", label="LiDAR", zorder=1)
    ax.scatter(pc0[:, 0], pc0[:, 1], s=14, c="#d62728", marker="^",
               edgecolor="black", linewidth=0.3, label="Radar", zorder=4)
    draw_boxes(ax, lbl1, tf1)
    style_bev(ax, "Radar / LiDAR alignment (radar frame)")

    # 2) ego-motion compensation
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(pc0[:, 0], pc0[:, 1], s=16, c="#1f77b4", marker="^",
               edgecolor="black", linewidth=0.3, label="raw sweep t", zorder=3)
    ax.scatter(pc0_comp[:, 0], pc0_comp[:, 1], s=16, c="#d62728", marker="v",
               edgecolor="black", linewidth=0.3, label="ego-compensated", zorder=4)
    style_bev(ax, "Ego-motion compensation")

    # 3) LiDAR BEV height map
    ax = fig.add_subplot(gs[0, 2])
    img = lidar_bev_image(pc0_l)
    ax.imshow(img.T, origin="lower", cmap="viridis", aspect="equal",
              extent=[BEV_FWD[0], BEV_FWD[1], BEV_SIDE[0], BEV_SIDE[1]])
    draw_boxes(ax, lbl1, tf1, color="white")
    ax.set_title("LiDAR BEV height map", fontsize=10)
    ax.set_xlabel("X — forward (m)")
    ax.set_ylabel("Y — lateral (m)")

    # 4) RGB reference
    ax = fig.add_subplot(gs[0, 3])
    if rgb is not None:
        ax.imshow(rgb)
    else:
        ax.text(0.5, 0.5, "RGB unavailable", ha="center", va="center")
    ax.set_title("Camera (reference)", fontsize=10)
    ax.axis("off")

    fig.suptitle(f"{clip} — frame {curr_idx:05d} — {len(lbl1)} moving GT objects",
                 fontsize=12)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def run_split(split, root, limit, stride):
    args = SimpleNamespace(eval=(split == "val"), dataset_path=root,
                           aug=False, num_workers=0, batch_size=1)
    ds = TrackingDataVOD(args, root)
    out_dir = os.path.join(CHECK_DIR, split)
    os.makedirs(out_dir, exist_ok=True)

    loc = VodTrackLocations(root_dir=root, output_dir=root,
                            frame_set_path="", pred_dir="")

    total = len(ds)
    idxs = range(0, total, stride)
    if limit:
        idxs = list(idxs)[:limit]

    done = failed = 0
    for i in idxs:
        try:
            sample = ds[i]
            frame_id = str(sample[5]).zfill(5)
            try:
                rgb = FrameDataLoader(kitti_locations=loc,
                                      frame_number=frame_id).image
            except Exception:
                rgb = None
            out = os.path.join(out_dir, f"{sample[6]}_{frame_id}.png")
            render(sample, rgb, out)
            done += 1
            if done % 25 == 0:
                print(f"  [{split}] {done}/{len(list(idxs))} figuras", flush=True)
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  [{split}] frame idx {i} falló: "
                      f"{type(e).__name__}: {str(e)[:90]}", flush=True)
    print(f"  [{split}] listo: {done} figuras en {out_dir} ({failed} fallos)")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "both"], default="both")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--limit", type=int, default=0,
                    help="max figures per split (0 = all)")
    ap.add_argument("--stride", type=int, default=1,
                    help="render every Nth frame")
    a = ap.parse_args()

    if not os.path.isdir(a.root):
        raise SystemExit(f"ERROR: dataset not found at {a.root}. "
                         f"Mount View-of-Delft there or pass --root.")

    splits = ["train", "val"] if a.split == "both" else [a.split]
    total = 0
    for s in splits:
        total += run_split(s, a.root, a.limit, a.stride)
    print(f"\nTotal: {total} figuras bajo {CHECK_DIR}/")


if __name__ == "__main__":
    main()
