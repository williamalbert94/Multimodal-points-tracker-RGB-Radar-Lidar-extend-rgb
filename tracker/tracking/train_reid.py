"""Entrena la cabeza Re-ID con triplet loss sobre identidades de track (GT).

Datos: pickle de detecciones GT-seg del split de TRAIN (features Q pooleadas por
caja + track_id GT). Se arman pares de frames consecutivos (t, t+1); el triplet
con hard mining acerca embeddings del mismo track_id y aleja los distintos.

Se entrena en TRAIN y se evalúa en VAL (sin leakage de identidades).

Uso (contenedor):
    /opt/conda/envs/mira/bin/python -u -m tracker.tracking.train_reid \
        --train tracker/results/detections_gtseg_train.pkl \
        --out tracker/results/reid_head.pth --epochs 40
"""
import argparse
import pickle

import numpy as np
import torch

from tracker.tracking.reid_head import ReIDHead, features_a_tensor, batch_hard_triplet

DEV = "cuda"


def pares_consecutivos(dets):
    """Frames (f, f+1) ambos con ≥1 caja -> lista de pares ordenados."""
    fs = sorted(dets.keys())
    pares = []
    presentes = set(fs)
    for f in fs:
        if (f + 1) in presentes and len(dets[f]["boxes"]) and len(dets[f + 1]["boxes"]):
            pares.append((f, f + 1))
    return pares


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="pickle gtseg de TRAIN")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--embedding-dim", type=int, default=256)
    args = ap.parse_args()

    dets = pickle.load(open(args.train, "rb"))
    pares = pares_consecutivos(dets)
    ap_dim = np.asarray(dets[pares[0][0]]["feat_max"]).shape[1] * 2
    print(f"[train-reid] {len(dets)} frames, {len(pares)} pares consecutivos, "
          f"appear_dim={ap_dim}", flush=True)

    head = ReIDHead(appear_dim=ap_dim, embedding_dim=args.embedding_dim).to(DEV)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    for ep in range(1, args.epochs + 1):
        head.train()
        np.random.shuffle(pares)
        tot, ntr = 0.0, 0
        for fa, fb in pares:
            app_a, box_a = features_a_tensor(dets[fa], DEV)
            app_b, box_b = features_a_tensor(dets[fb], DEV)
            ids_a = torch.from_numpy(dets[fa]["track_ids"]).to(DEV)
            ids_b = torch.from_numpy(dets[fb]["track_ids"]).to(DEV)
            emb_a = head(app_a, box_a)
            emb_b = head(app_b, box_b)
            loss, n = batch_hard_triplet(emb_a, ids_a, emb_b, ids_b, margin=args.margin)
            if loss is None:
                continue
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * n; ntr += n
        sched.step()
        if ep % 2 == 0 or ep == 1:
            print(f"  época {ep}/{args.epochs}  triplet_loss={tot/max(ntr,1):.4f} "
                  f"({ntr} triplets)", flush=True)

    torch.save({"model": head.state_dict(), "appear_dim": ap_dim,
                "embedding_dim": args.embedding_dim}, args.out)
    print(f"[train-reid] cabeza guardada -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
