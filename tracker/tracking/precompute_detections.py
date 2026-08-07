"""Precomputa las detecciones (cajas móviles) por frame y las guarda en disco.

Corre el backbone Q (segmentación) sobre el split de validación una sola vez;
por cada frame:
    segmentación -> puntos móviles -> DBSCAN + caja orientada (box_proposal)
y guarda un pickle {frame_num: {'boxes': [M,7], 'scores': [M]}} en frame radar.

Así el tracking se puede iterar rápido leyendo este archivo, sin re-correr el
modelo. (La apariencia/embeddings se agrega después en un segundo pase.)

Uso (dentro del contenedor):
    /opt/conda/envs/mira/bin/python -u -m tracker.tracking.precompute_detections \
        --config tracker/config/seg_exp_Q_lidarflow.yaml \
        --checkpoint tracker/checkpoints/seg_exp_Q_lidarflow/best_miou_model.pth \
        --umbral 0.5 --out tracker/results/detections_val.pkl
"""
import argparse
import pickle
import warnings

import numpy as np
import torch

from tracker.config import load_config
from tracker.dataset import TrackingDataVOD
from tracker.model import build_model
from tracker.detection.box_proposal import propose_boxes
from tracker.runner.inference_seg import predecir_frame


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--umbral", type=float, default=0.5,
                    help="Corte de segmentación para 'móvil'.")
    ap.add_argument("--eps", type=float, default=2.0, help="Radio DBSCAN (m).")
    ap.add_argument("--min-samples", type=int, default=1)
    ap.add_argument("--min-points", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True, help="Ruta del pickle de salida.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.num_workers = 0
    cfg.eval = True
    cfg.aug = False

    net = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cuda")
    net.load_state_dict(ckpt.get("model", ckpt))
    net.eval()
    print(f"[precompute] checkpoint: {args.checkpoint} "
          f"(mIoU {ckpt.get('miou', float('nan')):.4f})", flush=True)

    ds = TrackingDataVOD(cfg, cfg.dataset_path)
    total = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    in_ch = int(getattr(cfg, "in_channels", 2))
    print(f"[precompute] frames: {total} | umbral={args.umbral} eps={args.eps} "
          f"min_points={args.min_points}", flush=True)

    dets = {}
    n_box = 0
    for i in range(total):
        try:
            m = ds[i]
        except Exception:
            continue
        puntos, feats = m[0], m[2]
        num_frame = int(m[5])
        if len(puntos) == 0:
            continue

        prob = predecir_frame(
            net, puntos, feats, cfg.num_points, in_ch,
            puntos_prev=m[1], feats_prev=m[3], puntos_comp=m[4],
            puntos_ref2=m[17], feats_ref2=m[18], puntos_comp2=m[19])

        mask_mov = prob > args.umbral
        boxes, scores = propose_boxes(
            puntos[:, :3], mask_mov, prob=prob,
            eps=args.eps, min_samples=args.min_samples, min_points=args.min_points)

        dets[num_frame] = {"boxes": boxes.astype(np.float32),
                           "scores": scores.astype(np.float32)}
        n_box += len(boxes)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{total}  (frames={len(dets)}, cajas={n_box})", flush=True)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(dets, f)
    print(f"\n[precompute] guardado {len(dets)} frames, {n_box} cajas -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
