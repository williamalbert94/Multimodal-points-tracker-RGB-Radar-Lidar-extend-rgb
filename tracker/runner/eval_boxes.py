"""Evaluación de DETECCIÓN (Etapa 1): propone cajas desde la segmentación y mide mAP.

Pipeline por frame del split de validación:
    segmentación (mejor checkpoint) -> puntos móviles
      -> DBSCAN + caja orientada (PCA)        [box_proposal.propose_boxes]
      -> match con las cajas GT móviles, IoU-BEV rotado
      -> mAP @ {0.3, 0.5, 0.7}                 [detection.metrics_3d]

Uso (dentro del contenedor):
    python -m tracker.runner.eval_boxes \
        --config tracker/config/seg_exp_Q_lidarflow.yaml \
        --checkpoint tracker/checkpoints/seg_exp_Q_lidarflow/best_miou_model.pth \
        --umbral 0.5 --eps 2.0 --min-points 2
"""
import argparse
import warnings

import numpy as np
import torch

from tracker.config import load_config
from tracker.dataset import TrackingDataVOD
from tracker.dataset.gt_labels import get_bbx_param
from tracker.model import build_model
from tracker.detection.box_proposal import propose_boxes
from tracker.detection.metrics_3d import compute_map_3d
from tracker.runner.inference_seg import predecir_frame


def gt_boxes_from_labels(lbl1, transforms, margen=0.0):
    """Cajas GT móviles del frame como [N, 7] = (x, y, z, l, w, h, yaw) en radar."""
    if not lbl1:
        return np.zeros((0, 7), dtype=np.float32)
    cajas = []
    for label in lbl1.values():
        obb = get_bbx_param(
            [label.h, label.w, label.l, label.x, label.y, label.z, label.ry],
            transforms, "radar", margen=margen)
        c = np.asarray(obb.center).reshape(3)
        e = np.asarray(obb.extent).reshape(3)                 # [l, w, h]
        Rm = np.asarray(obb.R).reshape(3, 3)
        yaw = float(np.arctan2(Rm[1, 0], Rm[0, 0]))
        cajas.append([c[0], c[1], c[2], e[0], e[1], e[2], yaw])
    return np.asarray(cajas, dtype=np.float32)


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="Evaluación de detección (mAP) desde segmentación.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--umbral", type=float, default=0.5, help="Corte de segmentación para móvil.")
    ap.add_argument("--limit", type=int, default=0, help="Máx frames (0 = todos).")
    ap.add_argument("--eps", type=float, default=2.0, help="Radio DBSCAN (m).")
    ap.add_argument("--min-samples", type=int, default=1, help="Núcleo DBSCAN.")
    ap.add_argument("--min-points", type=int, default=2, help="Puntos mínimos por caja.")
    ap.add_argument("--gt-margin", type=float, default=0.0,
                    help="Margen de la GT de detección (0 = extensión real anotada).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.num_workers = 0
    cfg.eval = True
    cfg.aug = False

    net = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cuda")
    net.load_state_dict(ckpt.get("model", ckpt))
    net.eval()
    print(f"[eval-boxes] checkpoint: {args.checkpoint} "
          f"(época {ckpt.get('epoch', '?')}, mIoU {ckpt.get('miou', float('nan')):.4f})")

    ds = TrackingDataVOD(cfg, cfg.dataset_path)
    total = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    margen = float(args.gt_margin)                        # GT de detección = extensión real
    in_ch = int(getattr(cfg, "in_channels", 2))
    print(f"[eval-boxes] frames: {total} | umbral={args.umbral} eps={args.eps} "
          f"min_points={args.min_points}")

    pred_boxes, gt_boxes, pred_scores = [], [], []
    n_pred = n_gt = 0
    for i in range(total):
        try:
            m = ds[i]
        except Exception:
            continue
        puntos, feats = m[0], m[2]
        lbl1, transforms = m[12], m[14]
        if len(puntos) == 0:
            continue

        prob = predecir_frame(
            net, puntos, feats, cfg.num_points, in_ch,
            puntos_prev=m[1], feats_prev=m[3], puntos_comp=m[4],
            puntos_ref2=m[17], feats_ref2=m[18], puntos_comp2=m[19])

        mask_mov = prob > args.umbral
        boxes_p, scores_p = propose_boxes(
            puntos[:, :3], mask_mov, prob=prob,
            eps=args.eps, min_samples=args.min_samples, min_points=args.min_points)
        boxes_g = gt_boxes_from_labels(lbl1, transforms, margen=margen)

        pred_boxes.append(boxes_p)
        pred_scores.append(scores_p)
        gt_boxes.append(boxes_g)
        n_pred += len(boxes_p)
        n_gt += len(boxes_g)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{total}  (cajas pred={n_pred}, gt={n_gt})", flush=True)

    print(f"\n[eval-boxes] total cajas propuestas={n_pred}  GT={n_gt}")
    met = compute_map_3d(pred_boxes, gt_boxes, scores_pred_list=pred_scores,
                         iou_thresholds=[0.3, 0.5, 0.7])
    print("\n================ DETECCIÓN (mAP) ================")
    for k in ["mAP@0.3", "mAP@0.5", "mAP@0.7", "F1@0.5", "mAP"]:
        if k in met:
            print(f"  {k:10s} = {met[k]:.4f}")


if __name__ == "__main__":
    main()
