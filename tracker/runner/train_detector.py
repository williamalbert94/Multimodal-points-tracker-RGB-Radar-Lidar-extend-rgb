"""Entrena la cabecera de detección (Fase 2) sobre el backbone de segmentación.

El backbone (mejor modelo de segmentación) se carga y se CONGELA; solo se entrena
la `PointDetectionHead` con las cajas GT móviles. Métrica: mAP con IoU-BEV rotado.

Uso (dentro del contenedor):
    python -m tracker.runner.train_detector \
        --config tracker/config/seg_exp_Q_lidarflow.yaml \
        --seg-checkpoint tracker/checkpoints/seg_exp_Q_lidarflow/best_miou_model.pth \
        --epochs 60 --lr 0.001
"""
import argparse
import logging
import os
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from tracker.config import load_config
from tracker.dataset import TrackingDataVOD
from tracker.model import build_model
from tracker.runner.train_utils import custom_collate_fn
from tracker.runner.train_utils.trainer import _prepare_batch
from tracker.runner.eval_boxes import gt_boxes_from_labels
from tracker.detection.detection_head import PointDetectionHead
from tracker.detection.detection_losses import asignar_targets, detection_loss
from tracker.detection.decode import decode_boxes
from tracker.detection.metrics_3d import compute_map_3d

DEVICE = "cuda"


def build_loader(cfg, train):
    cfg.eval = not train
    ds = TrackingDataVOD(cfg, cfg.dataset_path)
    cfg.eval = False
    return DataLoader(ds, batch_size=cfg.batch_size, num_workers=getattr(cfg, "num_workers", 4),
                      shuffle=bool(train and getattr(cfg, "shuffle", False)),
                      drop_last=train, collate_fn=custom_collate_fn)


def backbone_feats(net, d, multiframe):
    """Features por punto del backbone congelado. [B, F, N]."""
    with torch.no_grad():
        if multiframe:
            _, feats = net(d["pc1"], d["ft1"], d["pc2"], d["ft2"], d["pc1_comp"],
                           pc3=d["pc3"], feature3=d["ft3"], pc1_comp2=d["pc1_comp2"])
        else:
            _, feats = net(d["pc1"], d["ft1"], d["pc2"], d["ft2"], d["pc1_comp"])
    return feats


def gt_boxes_batch(batch):
    """Cajas GT móviles por muestra como tensores [M,7] (extensión real, margen 0)."""
    lbl1_b, tf_b = batch[12], batch[14]
    out = []
    for lbl, tf in zip(lbl1_b, tf_b):
        g = gt_boxes_from_labels(lbl, tf, margen=0.0)
        out.append(torch.from_numpy(g).float() if len(g) else torch.zeros(0, 7))
    return out


@torch.no_grad()
def evaluar_map(net, head, loader, cfg, multiframe, umbral=0.3, nms_iou=0.3):
    head.eval()
    P, S, G = [], [], []
    for batch in loader:
        d = _prepare_batch(batch, cfg.num_points, cfg.in_channels, DEVICE,
                           0.0, 0, multiframe=multiframe)
        feats = backbone_feats(net, d, multiframe)
        pred = head(feats)
        dec = decode_boxes(pred, d["pc1"], umbral=umbral, nms_iou=nms_iou)
        gts = gt_boxes_batch(batch)
        for (bx, sc), g in zip(dec, gts):
            P.append(bx); S.append(sc); G.append(g.numpy())
    return compute_map_3d(P, G, scores_pred_list=S, iou_thresholds=[0.3, 0.5, 0.7])


def main(args):
    cfg = load_config(args.config)
    cfg.num_workers = getattr(cfg, "num_workers", 8)
    multiframe = bool(getattr(cfg, "multiframe", False)) or bool(getattr(cfg, "lidar_temporal", False))

    log = logging.getLogger("det"); log.setLevel(logging.INFO)
    h = logging.StreamHandler(); h.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
    log.handlers.clear(); log.addHandler(h)

    # ── Backbone congelado ────────────────────────────────────────────────────
    net = build_model(cfg)
    ck = torch.load(args.seg_checkpoint, map_location=DEVICE)
    net.load_state_dict(ck.get("model", ck))
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    log.info(f"backbone congelado: {args.seg_checkpoint} (mIoU {ck.get('miou', float('nan')):.4f})")

    tr = build_loader(cfg, True)
    va = build_loader(cfg, False)

    # dimensión de features (forward dummy)
    d0 = _prepare_batch(next(iter(tr)), cfg.num_points, cfg.in_channels, DEVICE, 0.0, 0, multiframe=multiframe)
    fdim = backbone_feats(net, d0, multiframe).shape[1]
    head = PointDetectionHead(feature_dim=fdim).to(DEVICE)
    log.info(f"cabecera de detección: feature_dim={fdim}, params={sum(p.numel() for p in head.parameters())/1e6:.2f}M")

    opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=6)

    ck_dir = os.path.join(getattr(cfg, "checkpoint_dir", "./checkpoints"),
                          getattr(cfg, "exp_name", "seg") + "_detector")
    os.makedirs(ck_dir, exist_ok=True)
    best = -1.0
    sin_mejora = 0
    for ep in range(1, args.epochs + 1):
        head.train()
        ls = {"conf": 0, "off": 0, "size": 0, "yaw": 0}; nb = 0
        for batch in tr:
            d = _prepare_batch(batch, cfg.num_points, cfg.in_channels, DEVICE, 0.0, 0, multiframe=multiframe)
            feats = backbone_feats(net, d, multiframe)
            pred = head(feats)
            gts = [g.to(DEVICE) for g in gt_boxes_batch(batch)]
            tgt = asignar_targets(d["pc1"], gts)
            loss, ld = detection_loss(pred, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            for k in ls: ls[k] += ld[k]
            nb += 1
        log.info(f"época {ep}/{args.epochs}  conf={ls['conf']/nb:.3f} off={ls['off']/nb:.3f} "
                 f"size={ls['size']/nb:.3f} yaw={ls['yaw']/nb:.3f}")

        if ep % args.val_every == 0 or ep == args.epochs:
            met = evaluar_map(net, head, va, cfg, multiframe)
            log.info(f"  [val] mAP@0.3={met['mAP@0.3']:.4f} mAP@0.5={met['mAP@0.5']:.4f} "
                     f"mAP@0.7={met['mAP@0.7']:.4f} F1@0.5={met['F1@0.5']:.4f}")
            sched.step(met["mAP@0.5"])
            if met["mAP@0.5"] > best:
                best = met["mAP@0.5"]; sin_mejora = 0
                torch.save({"head": head.state_dict(), "epoch": ep,
                            "map50": best}, os.path.join(ck_dir, "best_map_head.pth"))
                log.info(f"  nuevo mejor mAP@0.5={best:.4f} -> guardado")
            else:
                sin_mejora += 1
                if sin_mejora >= args.early_stop:
                    log.info(f"early stopping (sin mejorar {args.early_stop} vals). Mejor mAP@0.5={best:.4f}")
                    break
    log.info(f"listo. Mejor mAP@0.5={best:.4f}")


if __name__ == "__main__":
    import torch.multiprocessing as _mp
    _mp.set_start_method("spawn", force=True)
    warnings.filterwarnings("ignore")
    logging.getLogger("numba").setLevel(logging.CRITICAL)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seg-checkpoint", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--early-stop", type=int, default=12)
    main(ap.parse_args())
