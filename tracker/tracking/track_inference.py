"""Inferencia de TRACKING: genera la misma estructura de resultados que la de
segmentación (`seg_exp_Q_lidarflow`), pero a nivel de tracks.

Usa el mejor checkpoint (vía las detecciones precomputadas con el filtro de boxes
GT+segmentación) y el GalleryTracker (movimiento; `--reid-head` para apariencia).

Produce en `<out>/`:
    vis/<clip>/<frame>.png    figura de 3 paneles: GT (BEV) | RGB | Predicción (BEV)
                              con cajas + track IDs (color = identidad)
    data/<clip>/<frame>.txt   tracks predichos, un renglón por track:
                              frame track_id x y z l w h yaw score   (frame radar)
    metrics.txt               resumen de métricas de tracking

Uso (contenedor):
    /opt/conda/envs/mira/bin/python -u -m tracker.tracking.track_inference \
        --detections tracker/results/detections_gtseg_val.pkl \
        --out tracker/results/track_exp_Q_lidarflow --cada 1
"""
import argparse
import os
import pickle

import numpy as np

from external.vod.configuration import VodTrackLocations
from external.vod.frame import FrameDataLoader, FrameTransformMatrix
from tracker.tracking.gt_tracks import GtTrackLoader, read_clip_frames
from tracker.tracking.gallery_tracker import GalleryTracker
from tracker.tracking.mot_metrics import MOTMetricsAccumulator
from tracker.tracking.viz_tracking import figura, CLIPS_DIR

VAL_CLIPS = ["delft_1", "delft_10", "delft_14", "delft_22"]


def escribir_data(ruta, frame, boxes, ids, scores):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w") as f:
        for b, t, s in zip(boxes, ids, scores):
            f.write(f"{frame} {int(t)} {b[0]:.4f} {b[1]:.4f} {b[2]:.4f} "
                    f"{b[3]:.4f} {b[4]:.4f} {b[5]:.4f} {b[6]:.4f} {s:.4f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/project/view_of_delft_PUBLIC")
    ap.add_argument("--detections", required=True)
    ap.add_argument("--reid-head", default=None)
    ap.add_argument("--out", default="tracker/results/track_exp_Q_lidarflow")
    ap.add_argument("--clips", nargs="+", default=VAL_CLIPS)
    ap.add_argument("--cada", type=int, default=1, help="guardar figura cada N frames")
    ap.add_argument("--gt-min-radar-points", type=int, default=0)
    ap.add_argument("--fov-only", action="store_true",
                    help="excluye GT y detecciones fuera del FOV de la cámara")
    ap.add_argument("--max-age", type=int, default=10)
    ap.add_argument("--match-threshold", type=float, default=0.3)
    args = ap.parse_args()

    dets = pickle.load(open(args.detections, "rb"))
    gl = GtTrackLoader(args.dataset, min_radar_points=args.gt_min_radar_points,
                       fov_only=args.fov_only)
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

    dir_vis = os.path.join(args.out, "vis")
    dir_data = os.path.join(args.out, "data")
    acc = MOTMetricsAccumulator()
    n_fig = 0

    for ci, clip in enumerate(args.clips):
        frames = read_clip_frames(os.path.join(CLIPS_DIR, f"{clip}.txt"))
        tracker = GalleryTracker(max_age=args.max_age,
                                 matching_threshold=args.match_threshold,
                                 use_appearance=bool(args.reid_head))
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
            # filtro FOV a las detecciones (consistente con el GT)
            if args.fov_only and len(det_boxes):
                fmask = gl.fov_mask(det_boxes, f)
                det_boxes = det_boxes[fmask]
                if npts is not None:
                    npts = npts[fmask]
                if emb is not None:
                    emb = emb[fmask]
            pred_boxes, pred_ids = tracker.update(det_boxes, emb, npts)

            # scores por track (nº de puntos de su detección, o 1)
            scores = (npts if npts is not None and len(npts) == len(pred_boxes)
                      else np.ones(len(pred_boxes)))

            # ── data/ ────────────────────────────────────────────────────────
            escribir_data(os.path.join(dir_data, clip, f"{int(f):05d}.txt"),
                          int(f), pred_boxes, pred_ids, scores)

            # ── métricas (ids desplazados por clip) ──────────────────────────
            gt_boxes, gt_ids, _ = gl.load_frame(f)
            oid = np.asarray(pred_ids, int)
            if len(oid):
                oid = oid + (ci + 1) * 10_000_000
            acc.update(frame_id=ci * 100000 + int(f),
                       gt_boxes=gt_boxes, gt_ids=gt_ids,
                       pred_boxes=pred_boxes, pred_ids=oid)

            # ── vis/ ─────────────────────────────────────────────────────────
            if k % max(args.cada, 1) == 0:
                try:
                    fd = FrameDataLoader(kitti_locations=loc, frame_number=f"{int(f):05d}")
                    image = fd.image
                    transforms = FrameTransformMatrix(fd)
                    figura(os.path.join(dir_vis, clip, f"{int(f):05d}.png"),
                           image, transforms, gt_boxes, gt_ids, pred_boxes, pred_ids,
                           clip, f"{int(f):05d}")
                    n_fig += 1
                except Exception:
                    pass
        print(f"  {clip}: {len(frames)} frames procesados", flush=True)

    # ── metrics.txt ──────────────────────────────────────────────────────────
    m = acc.compute_metrics()
    os.makedirs(args.out, exist_ok=True)
    filtro = (f"GT-box + segmentación (móvil)"
              + (f" + ≥{args.gt_min_radar_points} pts radar" if args.gt_min_radar_points else ""))
    apar = "con apariencia (ReID)" if args.reid_head else "sin apariencia (movimiento)"
    lineas = [
        "=" * 68,
        f"RESULTADOS DE TRACKING — {os.path.basename(args.out)}",
        f"detecciones : {args.detections}",
        f"filtro boxes: {filtro}",
        f"tracker     : GalleryTracker {apar}",
        "=" * 68,
        f"  MOTA        : {m['MOTA']:.2f}%",
        f"  sAMOTA      : {m['sAMOTA']:.2f}%   (AB3DMOT-scaled, prom. sobre IoU)",
        f"  AMOTA       : {m['AMOTA']:.2f}%",
        f"  IDF1        : {m['IDF1']:.2f}%",
        f"  MOTP        : {m['MOTP']:.2f}%",
        f"  MT/PT/ML    : {m['MT']:.1f}% / {m['PT']:.1f}% / {m['ML']:.1f}%",
        f"  ID-switches : {m['ID_switches']}",
        f"  Fragment.   : {m['Fragmentations']}",
        f"  TP/FP/FN    : {m['TP']} / {m['FP']} / {m['FN']}",
        "=" * 68,
        "sAMOTA/AMOTA oficial de AB3DMOT (barrido de confianza): "
        "ver tracker/tracking/amota_ab3dmot.py",
    ]
    texto = "\n".join(lineas)
    with open(os.path.join(args.out, "metrics.txt"), "w") as fo:
        fo.write(texto + "\n")
    print("\n" + texto)
    print(f"\n[track-inf] {n_fig} figuras -> {dir_vis}\n"
          f"            data -> {dir_data}\n"
          f"            métricas -> {os.path.join(args.out, 'metrics.txt')}", flush=True)


if __name__ == "__main__":
    main()
