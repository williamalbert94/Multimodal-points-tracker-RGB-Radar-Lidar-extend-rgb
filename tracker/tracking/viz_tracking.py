"""Figuras de TRACKING por frame, mismo estilo que la viz de segmentación:

    Ground Truth (BEV)  |  Proyección en cámara RGB  |  Predicción (BEV)

pero a nivel de CAJA + ID de track (en vez de puntos móvil/estático). Cada caja
se colorea por su track ID y se rotula con el número de ID; así se sigue la
identidad entre frames (un cambio de color/ID en la predicción = ID switch).

Corre el GalleryTracker sobre un clip (detecciones GT-seg) y guarda un PNG por
frame en <out>/<clip>/<frame>.png.

Uso (contenedor):
    /opt/conda/envs/mira/bin/python -u -m tracker.tracking.viz_tracking \
        --detections tracker/results/detections_gtseg_val.pkl \
        --clip delft_1 --out tracker/results/track_exp_Q/vis --cada 1
    (opcional: --reid-head tracker/results/reid_head.pth  para usar apariencia)
"""
import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from external.vod.configuration import VodTrackLocations
from external.vod.frame import FrameDataLoader, FrameTransformMatrix
from tracker.tracking.gt_tracks import GtTrackLoader, read_clip_frames
from tracker.tracking.gallery_tracker import GalleryTracker

CLIPS_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "clips")

LIM_X = (0, 75)          # metros hacia adelante (eje vertical del BEV)
LIM_Y = (-30, 30)        # metros a los lados   (eje horizontal del BEV)
BG = "#37474F"           # fondo oscuro (igual que la viz de segmentación)

# aristas de la caja (0-3 cara superior, 4-7 base)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
         (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def color_de_id(tid):
    return plt.cm.tab20(int(tid) % 20)


def corners_radar(box):
    """8 esquinas 3D (frame radar). z es BOTTOM-center (KITTI) -> caja de z a z+h."""
    x, y, z, l, w, h, yaw = box
    dx = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
    dy = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])
    dz = np.array([h, h, h, h, 0, 0, 0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([dx*c - dy*s + x, dx*s + dy*c + y, dz + z], axis=1)


def proyectar(pts_xyz, t_camera_radar, P):
    n = len(pts_xyz)
    homo = np.hstack([pts_xyz[:, :3], np.ones((n, 1), np.float32)])
    cam = (t_camera_radar @ homo.T).T
    depth = cam[:, 2].copy()
    uvw = P @ cam.T
    uvw = uvw / np.where(np.abs(uvw[2]) < 1e-6, 1e-6, uvw[2])
    return uvw[:2].T, depth


def _panel_bev(ax, boxes, ids, titulo):
    """BEV: cajas (footprint) + ID. Eje horizontal = Y, vertical = X."""
    ax.set_facecolor(BG)
    for b, t in zip(boxes, ids):
        cor = corners_radar(b)[:4]            # cara superior
        col = color_de_id(t)
        poly = np.vstack([cor, cor[0]])
        ax.plot(poly[:, 1], poly[:, 0], color=col, lw=1.8)   # (Y, X)
        ax.text(b[1], b[0], str(int(t)), color="white", fontsize=7,
                ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc=col, ec="none", alpha=0.9))
    ax.set_xlim(LIM_Y); ax.set_ylim(LIM_X)
    ax.set_xlabel("Y (m) - Izquierda/Derecha", fontsize=9)
    ax.set_ylabel("X (m) - Adelante", fontsize=9)
    ax.set_title(titulo, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.15)


def _panel_rgb(ax, imagen, boxes, ids, transforms):
    if imagen is None:
        ax.text(0.5, 0.5, "sin imagen", ha="center", va="center"); ax.axis("off"); return
    ax.imshow(imagen)
    H, W = imagen.shape[:2]
    P = transforms.camera_projection_matrix
    tcr = transforms.t_camera_radar
    for b, t in zip(boxes, ids):
        uvs, depth = proyectar(corners_radar(b), tcr, P)
        if np.any(depth <= 0):
            continue
        col = color_de_id(t)
        for a, bb in EDGES:
            ax.plot([uvs[a, 0], uvs[bb, 0]], [uvs[a, 1], uvs[bb, 1]], color=col, lw=1.6)
        u, v = uvs[:, 0].mean(), uvs[:, 1].min()
        if 0 <= u < W and 0 <= v < H:
            ax.text(u, v - 5, str(int(t)), color="white", fontsize=8, ha="center",
                    va="bottom", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", fc=col, ec="none", alpha=0.9))
    ax.set_title("Proyección en cámara RGB", fontsize=11, fontweight="bold")
    ax.axis("off")


def figura(ruta, imagen, transforms, gt_boxes, gt_ids, pred_boxes, pred_ids, clip, frame):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5),
                             gridspec_kw={"width_ratios": [1, 1.5, 1]})
    _panel_bev(axes[0], gt_boxes, gt_ids, f"Ground Truth (BEV) — {len(gt_boxes)} objetos")
    _panel_rgb(axes[1], imagen, pred_boxes, pred_ids, transforms)
    _panel_bev(axes[2], pred_boxes, pred_ids, f"Predicción (BEV) — {len(pred_boxes)} tracks")
    fig.suptitle(f"Tracking (caja + ID) — {clip} / frame {frame}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/project/view_of_delft_PUBLIC")
    ap.add_argument("--detections", required=True)
    ap.add_argument("--reid-head", default=None)
    ap.add_argument("--clip", default="delft_1")
    ap.add_argument("--out", default="tracker/results/track_exp_Q/vis")
    ap.add_argument("--cada", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-age", type=int, default=10)
    ap.add_argument("--match-threshold", type=float, default=0.3)
    args = ap.parse_args()

    dets = pickle.load(open(args.detections, "rb"))
    gl = GtTrackLoader(args.dataset)
    loc = VodTrackLocations(root_dir=args.dataset, output_dir=args.dataset,
                            frame_set_path="", pred_dir="")

    head = feat_fn = None
    if args.reid_head:
        import torch
        from tracker.tracking.reid_head import ReIDHead, features_a_tensor
        ck = torch.load(args.reid_head, map_location="cuda")
        head = ReIDHead(appear_dim=ck["appear_dim"], embedding_dim=ck["embedding_dim"]).to("cuda")
        head.load_state_dict(ck["model"]); head.eval()
        feat_fn = (features_a_tensor, torch)

    tracker = GalleryTracker(max_age=args.max_age,
                             matching_threshold=args.match_threshold,
                             use_appearance=bool(args.reid_head))

    frames = read_clip_frames(os.path.join(CLIPS_DIR, f"{args.clip}.txt"))
    if args.limit:
        frames = frames[:args.limit]
    out_dir = os.path.join(args.out, args.clip)

    for k, f in enumerate(frames):
        d = dets.get(int(f), {"boxes": np.zeros((0, 7), np.float32)})
        det_boxes = np.asarray(d["boxes"], np.float32)
        npts = np.asarray(d.get("num_points", [])) if len(det_boxes) else None
        emb = None
        if head is not None and len(det_boxes):
            feats_a, torch = feat_fn
            appear, box = feats_a(d, "cuda")
            with torch.no_grad():
                emb = head(appear, box).cpu().numpy()
        pred_boxes, pred_ids = tracker.update(det_boxes, emb, npts)

        if k % max(args.cada, 1) != 0:
            continue
        gt_boxes, gt_ids, _ = gl.load_frame(f)
        try:
            fd = FrameDataLoader(kitti_locations=loc, frame_number=f"{int(f):05d}")
            image = fd.image
            transforms = FrameTransformMatrix(fd)
        except Exception:
            continue
        figura(os.path.join(out_dir, f"{int(f):05d}.png"), image, transforms,
               gt_boxes, gt_ids, pred_boxes, pred_ids, args.clip, f"{int(f):05d}")
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(frames)} renderizados", flush=True)

    print(f"[viz] figuras -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
