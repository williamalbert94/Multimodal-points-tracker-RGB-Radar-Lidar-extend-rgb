"""Detector REALISTA (DBSCAN, sin cajas GT) usando TODA la evidencia disponible.

`precompute_detections.py` ya hace segmentación -> DBSCAN -> caja orientada,
pero (a) agrupa solo por XY, (b) para clusters chicos inventa el tamaño con un
prior de clase y (c) guarda solo {boxes, scores}, sin el descriptor que la
cabeza Re-ID necesita. Este script cierra las tres cosas, sin cambiar el
contrato de salida que consume `track_inference.py`.

Qué se usa, y para qué:

  * AGRUPAMIENTO — DBSCAN sobre [x, y, alfa * v_comp] en vez de solo [x, y].
    El radar mide velocidad radial compensada por ego (`v_r_comp`, canal 2 de
    las features). Dos objetos que se rozan en el espacio pero se mueven
    distinto (un coche que adelanta a un ciclista) caen en el mismo cluster si
    solo se mira XY; con el Doppler como tercera dimensión se separan. `alfa`
    convierte m/s en "metros equivalentes": con alfa=0.5, 2 m/s de diferencia
    pesan lo mismo que 1 m de separación.

  * CAJA 3D — el LiDAR denso del propio frame (raw_pc0_lidar, ya alineado al
    frame radar) define extensión y yaw, igual que `box_proposal.propose_boxes_lidar`:
    se recorta alrededor del centro del cluster, se quita el suelo por altura y
    se ajusta la caja sobre esos puntos. El radar localiza, el LiDAR mide. Solo
    cuando no hay LiDAR suficiente se cae al prior de clase.

  * APARIENCIA — se poolean (max + avg) las features Q de los puntos de cada
    cluster, que es lo que `reid_head` espera como `feat_max`/`feat_avg`.

A diferencia de `precompute_detections_gtseg.py`, acá NO interviene ninguna caja
de la anotación: centro, tamaño, yaw y cantidad de objetos salen enteros del
detector. Es la vía honesta de punta a punta.

Salida: pickle {frame: {boxes[M,7], num_points[M], feat_max[M,F], feat_avg[M,F]}}
en frame radar. Sin `track_ids`: un cluster no tiene identidad propia, se la
asigna el tracker en el paso siguiente.

Uso (dentro del contenedor):
    python -u -m tracker.tracking.precompute_detections_dbscan_reid \
        --config tracker/config/seg_exp_Q_lidarflow.yaml \
        --checkpoint tracker/checkpoints/seg_exp_Q_lidarflow/best_miou_model.pth \
        --umbral 0.5 --out tracker/results/detections_dbscan_reid_val.pkl

Para reproducir el detector viejo (solo XY, sin LiDAR): --peso-doppler 0 --sin-lidar
"""
import argparse
import os
import pickle
import warnings

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

from tracker.config import load_config
from tracker.dataset import TrackingDataVOD
from tracker.model import build_model
from tracker.detection.box_proposal import (fit_oriented_box, _prior_por_span,
                                            _quitar_suelo)
from tracker.runner.inference_seg import predecir_frame

CANAL_V_COMP = 2        # features: [RCS, v_r, v_r_comp, n_vec, alt_media, rango_alt]


def caja_del_cluster(cluster_pts, arbol_lidar, lidar_xyz, radio_crop, min_lidar):
    """Caja del cluster: LiDAR si alcanza, si no radar + prior. (caja, uso_lidar)"""
    if arbol_lidar is not None:
        centro_xy = cluster_pts[:, :2].mean(axis=0)
        vecinos = arbol_lidar.query_ball_point(centro_xy, radio_crop)
        if len(vecinos) >= min_lidar:
            crop = lidar_xyz[vecinos]
            crop = _quitar_suelo(crop, base_z=float(crop[:, 2].min()))
            if len(crop) >= min_lidar:
                return fit_oriented_box(crop), True

    caja = fit_oriented_box(cluster_pts)
    l, w, h = _prior_por_span(float(caja[3]))
    caja[3], caja[4], caja[5] = l, w, h
    return caja, False


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--umbral", type=float, default=0.5,
                    help="corte de segmentación para 'móvil'")
    ap.add_argument("--eps", type=float, default=2.0, help="radio de DBSCAN (m)")
    ap.add_argument("--min-samples", type=int, default=1)
    ap.add_argument("--min-points", type=int, default=2,
                    help="descarta clusters con menos puntos que esto")
    ap.add_argument("--peso-doppler", type=float, default=0.5,
                    help="metros equivalentes por m/s de v_r_comp en el DBSCAN; "
                         "0 = agrupar solo por XY, como el detector viejo")
    ap.add_argument("--radio-crop", type=float, default=2.5,
                    help="radio XY (m) del recorte LiDAR alrededor del cluster")
    ap.add_argument("--min-lidar", type=int, default=5,
                    help="puntos LiDAR mínimos para confiar en la caja del LiDAR")
    ap.add_argument("--sin-lidar", action="store_true",
                    help="no usar LiDAR para la caja (radar + prior de clase)")
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.num_workers = 0
    cfg.eval = (args.split == "val")
    cfg.aug = False

    net = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cuda")
    net.load_state_dict(ckpt.get("model", ckpt))
    net.eval()
    print(f"[dbscan] checkpoint: {args.checkpoint} "
          f"(mIoU {ckpt.get('miou', float('nan')):.4f})", flush=True)
    print(f"[dbscan] umbral={args.umbral} eps={args.eps} "
          f"min_points={args.min_points} peso_doppler={args.peso_doppler} "
          f"caja={'radar+prior' if args.sin_lidar else 'LiDAR'}", flush=True)

    ds = TrackingDataVOD(cfg, cfg.dataset_path)
    total = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    in_ch = int(getattr(cfg, "in_channels", 2))

    dets = {}
    n_box = n_lidar_ok = 0
    for i in range(total):
        try:
            m = ds[i]
        except Exception:
            continue
        puntos, feats = m[0], m[2]
        num_frame = int(m[5])
        lidar_xyz = np.asarray(m[9])[:, :3] if m[9] is not None else None
        if len(puntos) == 0:
            continue

        prob, feat_pp = predecir_frame(
            net, puntos, feats, cfg.num_points, in_ch,
            puntos_prev=m[1], feats_prev=m[3], puntos_comp=m[4],
            puntos_ref2=m[17], feats_ref2=m[18], puntos_comp2=m[19],
            devolver_feats=True)

        mask_mov = prob > args.umbral
        idx_mov = np.where(mask_mov)[0]
        keep_boxes, keep_np, keep_fmax, keep_favg = [], [], [], []

        if len(idx_mov) >= max(1, args.min_points):
            pts_mov = np.asarray(puntos)[idx_mov, :3]

            # Espacio de agrupamiento: XY, más el Doppler compensado escalado a
            # metros. Sin esta tercera dimensión, dos objetos pegados pero con
            # velocidades distintas se funden en un solo cluster.
            rasgos = [pts_mov[:, 0], pts_mov[:, 1]]
            if args.peso_doppler > 0 and np.asarray(feats).shape[1] > CANAL_V_COMP:
                v_comp = np.asarray(feats)[idx_mov, CANAL_V_COMP]
                rasgos.append(args.peso_doppler * v_comp)
            espacio = np.stack(rasgos, axis=1)

            labels = DBSCAN(eps=args.eps,
                            min_samples=args.min_samples).fit_predict(espacio)

            arbol = None
            if not args.sin_lidar and lidar_xyz is not None and len(lidar_xyz):
                arbol = cKDTree(lidar_xyz[:, :2])

            for c in np.unique(labels):
                if c == -1:                               # ruido de DBSCAN
                    continue
                sel = labels == c
                npts = int(sel.sum())
                if npts < args.min_points:
                    continue
                caja, uso_lidar = caja_del_cluster(
                    pts_mov[sel], arbol, lidar_xyz, args.radio_crop, args.min_lidar)
                fin = feat_pp[idx_mov[sel]]               # (npts, F) del cluster
                keep_boxes.append(caja)
                keep_np.append(npts)
                keep_fmax.append(fin.max(axis=0))
                keep_favg.append(fin.mean(axis=0))
                n_lidar_ok += int(uso_lidar)

        if keep_boxes:
            dets[num_frame] = {
                "boxes": np.stack(keep_boxes).astype(np.float32),
                "num_points": np.array(keep_np, int),
                "feat_max": np.stack(keep_fmax).astype(np.float32),
                "feat_avg": np.stack(keep_favg).astype(np.float32),
            }
            n_box += len(keep_boxes)
        else:
            dets[num_frame] = _vacio()

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{total}  (frames={len(dets)}, cajas={n_box}, "
                  f"con caja LiDAR={n_lidar_ok})", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(dets, f)
    pct = 100 * n_lidar_ok / max(n_box, 1)
    print(f"\n[dbscan] {len(dets)} frames, {n_box} cajas "
          f"({n_lidar_ok} = {pct:.1f}% con extensión medida por LiDAR) -> {args.out}",
          flush=True)


def _vacio():
    return {"boxes": np.zeros((0, 7), np.float32),
            "num_points": np.zeros(0, int),
            "feat_max": np.zeros((0, 1), np.float32),
            "feat_avg": np.zeros((0, 1), np.float32)}


if __name__ == "__main__":
    main()
