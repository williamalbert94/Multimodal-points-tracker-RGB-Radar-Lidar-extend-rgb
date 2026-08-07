"""Detector multimodal RGB 2D -> 3D (frustum) para instancias MÓVILES.

Por frame:
  1. Faster R-CNN (COCO) sobre la imagen -> cajas 2D de clases móviles.
  2. LiDAR denso (en frame radar) proyectado a la imagen -> para cada caja 2D se
     toman los puntos LiDAR dentro (frustum), se agrupan por profundidad (DBSCAN)
     y se ajusta una caja 3D orientada (PCA), con prior de tamaño por clase.
  3. Filtro de MOVIMIENTO con la segmentación Q (0.60 IoU_moving): se conserva la
     caja solo si contiene ≥1 punto de radar MÓVIL proyectado dentro de la caja 2D
     (descarta autos/ciclistas estacionados, que no están en el GT móvil).

Guarda un pickle {frame_num: {'boxes': [M,7], 'scores': [M]}} en frame radar,
compatible con `eval_tracking.py --mode gallery-det`.

Uso (contenedor):
    /opt/conda/envs/mira/bin/python -u -m tracker.tracking.precompute_detections_rgb \
        --config tracker/config/seg_exp_Q_lidarflow.yaml \
        --checkpoint tracker/checkpoints/seg_exp_Q_lidarflow/best_miou_model.pth \
        --umbral 0.5 --score-2d 0.5 --out tracker/results/detections_rgb_val.pkl
"""
import argparse
import os
import pickle
import warnings

import numpy as np
import torch

from external.vod.configuration import VodTrackLocations
from external.vod.frame import FrameDataLoader
from tracker.config import load_config
from tracker.dataset import TrackingDataVOD
from tracker.model import build_model
from tracker.detection.box_proposal import fit_oriented_box, PRIOR_SIZES
from tracker.runner.inference_seg import predecir_frame

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None

# COCO (labels de torchvision) -> prior de tamaño por clase móvil.
COCO_MOVING = {
    1: "pedestrian",   # person
    2: "cyclist",      # bicycle
    3: "car",          # car
    4: "cyclist",      # motorcycle
    6: "car",          # bus  (usa prior grande de car)
    8: "car",          # truck
}


def proyectar(pts_xyz, t_camera_pcl, P):
    """Proyección MANUAL a la imagen preservando la correspondencia de índices.

    (project_pcl_to_image de VoD recorta los puntos fuera de imagen y rompe el
    alineamiento con el array original, así que se hace acá a mano.)

    Returns: uvs [N,2] (float), depth [N] (en frame cámara). Todos alineados con
    el input; la validez se decide con depth>0 y estar dentro de la imagen.
    """
    n = len(pts_xyz)
    homo = np.hstack([pts_xyz[:, :3], np.ones((n, 1), np.float32)])
    cam = (t_camera_pcl @ homo.T).T                 # [N,4] frame cámara
    depth = cam[:, 2].copy()
    uvw = P @ cam.T                                  # [3,N]
    uvw = uvw / np.where(np.abs(uvw[2]) < 1e-6, 1e-6, uvw[2])
    uvs = uvw[:2].T                                  # [N,2]
    return uvs, depth


def cargar_frcnn(device):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.eval().to(device)
    return model


@torch.no_grad()
def detectar_2d(frcnn, image, device, score_thr):
    """Imagen HxWx3 uint8 -> lista de (x1,y1,x2,y2, coco_label, score) móviles."""
    t = torch.from_numpy(image).float().permute(2, 0, 1).div(255.0).to(device)
    out = frcnn([t])[0]
    dets = []
    for box, lab, sc in zip(out["boxes"].cpu().numpy(),
                            out["labels"].cpu().numpy(),
                            out["scores"].cpu().numpy()):
        if sc < score_thr or int(lab) not in COCO_MOVING:
            continue
        dets.append((box[0], box[1], box[2], box[3], int(lab), float(sc)))
    return dets


def caja_frustum(lidar_frustum, clase, min_pts=8):
    """Ajusta una caja 3D orientada a los puntos LiDAR del frustum (frame radar).

    Agrupa por DBSCAN y se queda con el cluster MÁS DENSO (más puntos): el objeto
    detectado en 2D suele dominar los returns LiDAR de su caja, mientras el fondo
    y el clutter de primer plano quedan como clusters chicos. (Antes se tomaba el
    más cercano al sensor, que agarraba clutter de primer plano -> cajas corridas.)
    Prior de tamaño por clase si hay pocos puntos.
    """
    if len(lidar_frustum) < 3:
        return None
    pts = lidar_frustum
    if DBSCAN is not None and len(pts) >= min_pts:
        labels = DBSCAN(eps=1.0, min_samples=3).fit_predict(pts)
        mejor, mejor_n = None, 0
        for c in np.unique(labels):
            if c == -1:
                continue
            sel = pts[labels == c]
            if len(sel) > mejor_n:                 # cluster más denso
                mejor_n, mejor = len(sel), sel
        if mejor is not None:
            pts = mejor
    box = fit_oriented_box(pts)
    # tamaño por prior de clase (LiDAR mide bien, pero el frustum puede recortar)
    if clase in PRIOR_SIZES and len(pts) < 40:
        l, w, h = PRIOR_SIZES[clase]
        box[3], box[4], box[5] = l, w, h
    return box


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--umbral", type=float, default=0.5, help="corte segmentación móvil")
    ap.add_argument("--score-2d", type=float, default=0.5, help="corte score Faster R-CNN")
    ap.add_argument("--min-frustum", type=int, default=5, help="mín puntos LiDAR en frustum")
    ap.add_argument("--sin-filtro-movil", action="store_true",
                    help="no exige punto radar móvil dentro (mantiene estáticos)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = "cuda"
    cfg = load_config(args.config)
    cfg.num_workers = 0
    cfg.eval = True
    cfg.aug = False

    net = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location=dev)
    net.load_state_dict(ckpt.get("model", ckpt))
    net.eval()
    frcnn = cargar_frcnn(dev)
    print(f"[rgb-det] Q mIoU {ckpt.get('miou', float('nan')):.4f} | "
          f"Faster R-CNN COCO | score2d={args.score_2d} umbral={args.umbral}", flush=True)

    ds = TrackingDataVOD(cfg, cfg.dataset_path)
    total = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    in_ch = int(getattr(cfg, "in_channels", 2))
    kitti = VodTrackLocations(root_dir=cfg.dataset_path, output_dir=cfg.dataset_path,
                              frame_set_path="", pred_dir="")

    dets = {}
    n_box = n_2d = 0
    for i in range(total):
        try:
            m = ds[i]
        except Exception:
            continue
        puntos, feats = m[0], m[2]
        lidar_radar = m[9]                      # LiDAR denso en frame radar
        transforms = m[14]
        num_frame = int(m[5])
        if len(puntos) == 0 or lidar_radar is None or len(lidar_radar) == 0:
            dets[num_frame] = {"boxes": np.zeros((0, 7), np.float32),
                               "scores": np.zeros(0, np.float32)}
            continue

        try:
            image = FrameDataLoader(kitti_locations=kitti,
                                    frame_number=f"{num_frame:05d}").image
        except Exception:
            continue
        H, W = image.shape[:2]

        dets2d = detectar_2d(frcnn, image, dev, args.score_2d)
        n_2d += len(dets2d)
        if not dets2d:
            dets[num_frame] = {"boxes": np.zeros((0, 7), np.float32),
                               "scores": np.zeros(0, np.float32)}
            continue

        P = transforms.camera_projection_matrix
        t_cr = transforms.t_camera_radar

        # proyección LiDAR (radar frame) -> imagen, alineada con lidar_radar
        uvs_l, depth_l = proyectar(lidar_radar[:, :3], t_cr, P)
        val_l = (depth_l > 0) & (uvs_l[:, 0] >= 0) & (uvs_l[:, 0] < W) & \
                (uvs_l[:, 1] >= 0) & (uvs_l[:, 1] < H)

        # segmentación Q -> puntos radar MÓVILES (3D + proyección alineada).
        # El radar ANCLA el centro de la caja (localiza bien el centro); el LiDAR
        # recortado alrededor de ese centro refina yaw/extensión (evita el fondo).
        prob = predecir_frame(
            net, puntos, feats, cfg.num_points, in_ch,
            puntos_prev=m[1], feats_prev=m[3], puntos_comp=m[4],
            puntos_ref2=m[17], feats_ref2=m[18], puntos_comp2=m[19])
        mov = puntos[prob > args.umbral, :3]
        uvs_mov = None
        if len(mov):
            uvs_mov, dm = proyectar(mov, t_cr, P)
            vm = dm > 0
            mov, uvs_mov = mov[vm], uvs_mov[vm]

        boxes, scores = [], []
        for (x1, y1, x2, y2, lab, sc) in dets2d:
            clase = COCO_MOVING.get(lab, "car")
            # anclaje radar: puntos móviles dentro de la caja 2D
            if uvs_mov is None or len(uvs_mov) == 0:
                continue
            inb = ((uvs_mov[:, 0] >= x1) & (uvs_mov[:, 0] <= x2) &
                   (uvs_mov[:, 1] >= y1) & (uvs_mov[:, 1] <= y2))
            if inb.sum() < 1:            # sin radar móvil: no se ancla (descarta)
                continue
            mov_in = mov[inb]
            center = mov_in.mean(axis=0)
            l, w, h = PRIOR_SIZES.get(clase, PRIOR_SIZES["car"])
            yaw = 0.0

            # LiDAR dentro de la caja 2D y CERCA del centro radar (refina yaw/size)
            dl = (val_l & (uvs_l[:, 0] >= x1) & (uvs_l[:, 0] <= x2) &
                  (uvs_l[:, 1] >= y1) & (uvs_l[:, 1] <= y2))
            lid = lidar_radar[dl, :3]
            if len(lid):
                near = np.linalg.norm(lid[:, :2] - center[:2], axis=1) < 3.0
                lid_obj = lid[near]
                if len(lid_obj) >= args.min_frustum:
                    fb = fit_oriented_box(lid_obj)
                    yaw = float(fb[6])
                    if len(lid_obj) >= 40:      # suficiente LiDAR: extensión medida
                        l, w, h = float(fb[3]), float(fb[4]), float(fb[5])
            if yaw == 0.0 and len(mov_in) >= 3:  # si no hubo LiDAR, yaw por radar
                yaw = float(fit_oriented_box(mov_in)[6])

            boxes.append(np.array([center[0], center[1], center[2], l, w, h, yaw],
                                  np.float32))
            scores.append(sc)

        if boxes:
            dets[num_frame] = {"boxes": np.stack(boxes).astype(np.float32),
                               "scores": np.asarray(scores, np.float32)}
            n_box += len(boxes)
        else:
            dets[num_frame] = {"boxes": np.zeros((0, 7), np.float32),
                               "scores": np.zeros(0, np.float32)}

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{total}  (frames={len(dets)}, 2D={n_2d}, 3D={n_box})",
                  flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(dets, f)
    print(f"\n[rgb-det] {len(dets)} frames | 2D móviles={n_2d} | cajas 3D={n_box} "
          f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
