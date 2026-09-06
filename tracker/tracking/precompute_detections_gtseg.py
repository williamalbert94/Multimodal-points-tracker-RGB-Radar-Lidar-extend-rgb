"""Detección "GT-box + filtro por segmentación" (proxy realista de detector).

Por frame:
  1. Se toman las cajas GT móviles (localización perfecta -> esquiva el muro de
     tamaño/ubicación del detector).
  2. Se corre el backbone Q -> puntos radar MÓVILES (prob > umbral).
  3. Se MANTIENE una caja GT solo si contiene ≥ `min_pts` puntos móviles dentro
     (recall REALISTA: solo lo que el radar de verdad percibe; las cajas
     invisibles se remueven y cuentan como FN, como en la realidad).
  4. Para cada caja mantenida se guardan los puntos móviles internos y sus
     features Q pooleadas (max+avg) -> insumo del embedding Re-ID.

Guarda pickle {frame: {boxes[M,7], num_points[M], track_ids[M],
                        feat_max[M,F], feat_avg[M,F]}} en frame radar.

Uso (contenedor):
    /opt/conda/envs/mira/bin/python -u -m tracker.tracking.precompute_detections_gtseg \
        --config tracker/config/seg_exp_Q_lidarflow.yaml \
        --checkpoint tracker/checkpoints/seg_exp_Q_lidarflow/best_miou_model.pth \
        --umbral 0.5 --min-pts 1 --out tracker/results/detections_gtseg_val.pkl
"""
import argparse
import os
import pickle
import warnings

import numpy as np
import torch

from tracker.config import load_config
from tracker.dataset import TrackingDataVOD
from tracker.model import build_model
from tracker.runner.inference_seg import predecir_frame
from tracker.tracking.gt_tracks import GtTrackLoader


def puntos_en_caja(pts_xyz, box, margen=0.25):
    """Máscara BEV de puntos dentro de la caja rotada. box=[x,y,z,l,w,h,yaw]."""
    rel = pts_xyz[:, :2] - box[:2]
    yaw = box[6]
    c, s = np.cos(-yaw), np.sin(-yaw)
    rx = rel[:, 0] * c - rel[:, 1] * s
    ry = rel[:, 0] * s + rel[:, 1] * c
    return (np.abs(rx) <= box[3] / 2 + margen) & (np.abs(ry) <= box[4] / 2 + margen)


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--umbral", type=float, default=0.5, help="corte segmentación móvil")
    ap.add_argument("--min-pts", type=int, default=1,
                    help="mín puntos móviles dentro para MANTENER la caja GT")
    ap.add_argument("--min-radar-pts", type=int, default=0,
                    help="además, exige ≥N puntos de RADAR en la caja (universo "
                         "RaTrack: objetos observables). 0 = sin este filtro")
    ap.add_argument("--moving-only", action="store_true",
                    help="protocolo RaTrack: solo objetos GT en movimiento")
    ap.add_argument("--todas-las-clases", action="store_true",
                    help="no excluye peatón ni las demás clases de EXCLUDE_TYPES; "
                         "solo descarta DontCare, como hace RaTrack")
    ap.add_argument("--clases", nargs="+", default=None, metavar="TIPO",
                    help="restringe el GT a estos tipos (ej: Car truck moped_scooter)")
    ap.add_argument("--margen", type=float, default=0.25,
                    help="margen de la caja para el test de contención (radar impreciso)")
    ap.add_argument("--split", choices=["train", "val"], default="val",
                    help="val = 4 clips de validación; train = resto (para ReID)")
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
    print(f"[gtseg] Q mIoU {ckpt.get('miou', float('nan')):.4f} | "
          f"umbral={args.umbral} min_pts={args.min_pts}", flush=True)

    ds = TrackingDataVOD(cfg, cfg.dataset_path)
    total = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    in_ch = int(getattr(cfg, "in_channels", 2))
    gl = GtTrackLoader(cfg.dataset_path, moving_only=args.moving_only,
                       keep_types=args.clases,
                       exclude_types={"DontCare"} if args.todas_las_clases else None)
    if args.todas_las_clases:
        print("[gtseg] sin exclusiones de clase: solo se descarta DontCare", flush=True)
    if args.clases:
        print(f"[gtseg] GT restringido a las clases: {sorted(args.clases)}", flush=True)

    dets = {}
    n_keep = n_gt = 0
    for i in range(total):
        try:
            m = ds[i]
        except Exception:
            continue
        puntos, feats = m[0], m[2]
        num_frame = int(m[5])
        if len(puntos) == 0:
            continue

        gt_boxes, gt_ids, _ = gl.load_frame(num_frame)
        n_gt += len(gt_boxes)
        if len(gt_boxes) == 0:
            dets[num_frame] = _vacio()
            continue

        prob, feat_pp = predecir_frame(
            net, puntos, feats, cfg.num_points, in_ch,
            puntos_prev=m[1], feats_prev=m[3], puntos_comp=m[4],
            puntos_ref2=m[17], feats_ref2=m[18], puntos_comp2=m[19],
            devolver_feats=True)
        mov_mask = prob > args.umbral
        mov_xyz = puntos[mov_mask, :3]
        mov_feat = feat_pp[mov_mask]                       # (Nmov, F)

        # Para el filtro de observabilidad se usa el RADAR CRUDO (no la nube
        # fusionada con LiDAR de m[0]), consistente con el conteo del GtTrackLoader.
        radar_xyz = None
        if args.min_radar_pts > 0:
            try:
                from external.vod.frame import FrameDataLoader
                fd = FrameDataLoader(kitti_locations=gl.loc,
                                     frame_number=f"{num_frame:05d}")
                radar_xyz = fd.radar_data[:, :3]
            except Exception:
                radar_xyz = puntos[:, :3]                  # fallback
        keep_boxes, keep_ids, keep_np, keep_fmax, keep_favg = [], [], [], [], []
        for b in range(len(gt_boxes)):
            if len(mov_xyz) == 0:
                continue
            # universo RaTrack: la caja debe tener ≥N puntos de radar CRUDO
            # (margen 0.0, idéntico a GtTrackLoader -> mismo conteo que el GT)
            if args.min_radar_pts > 0:
                nrad = int(puntos_en_caja(radar_xyz, gt_boxes[b], margen=0.0).sum())
                if nrad < args.min_radar_pts:
                    continue
            inside = puntos_en_caja(mov_xyz, gt_boxes[b], margen=args.margen)
            npt = int(inside.sum())
            if npt < args.min_pts:
                continue
            fin = mov_feat[inside]                          # (npt, F)
            keep_boxes.append(gt_boxes[b])
            keep_ids.append(int(gt_ids[b]))
            keep_np.append(npt)
            keep_fmax.append(fin.max(axis=0))
            keep_favg.append(fin.mean(axis=0))

        if keep_boxes:
            dets[num_frame] = {
                "boxes": np.stack(keep_boxes).astype(np.float32),
                "track_ids": np.array(keep_ids, int),
                "num_points": np.array(keep_np, int),
                "feat_max": np.stack(keep_fmax).astype(np.float32),
                "feat_avg": np.stack(keep_favg).astype(np.float32),
            }
            n_keep += len(keep_boxes)
        else:
            dets[num_frame] = _vacio()

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{total}  (mantenidas={n_keep} / GT={n_gt}, "
                  f"recall={100*n_keep/max(n_gt,1):.1f}%)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(dets, f)
    print(f"\n[gtseg] {len(dets)} frames | cajas mantenidas={n_keep} / GT={n_gt} "
          f"(recall {100*n_keep/max(n_gt,1):.1f}%) -> {args.out}", flush=True)


def _vacio():
    return {"boxes": np.zeros((0, 7), np.float32), "track_ids": np.zeros(0, int),
            "num_points": np.zeros(0, int),
            "feat_max": np.zeros((0, 1), np.float32),
            "feat_avg": np.zeros((0, 1), np.float32)}


if __name__ == "__main__":
    main()
