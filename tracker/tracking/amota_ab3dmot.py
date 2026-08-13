"""sAMOTA / AMOTA en el protocolo OFICIAL de AB3DMOT (el que usa el paper de RaTrack).

Diferencia clave con `mot_metrics.py`: AB3DMOT barre la CONFIANZA de las
detecciones (41 puntos), re-trackea a cada nivel, y calcula una MOTA ESCALADA
por recall que NO castiga los objetos legítimamente no detectados:

    recall_r = TP_r / num_gt
    sMOTA_r  = max(0, 1 - (IDSW_r + FP_r + FN_r - (1-recall_r)*num_gt) / (recall_r*num_gt))
    MOTA_r   = 1 - (IDSW_r + FP_r + FN_r) / num_gt          (AMOTA, sin escalar)

    sAMOTA = mean_r sMOTA_r          AMOTA = mean_r max(0, MOTA_r)

Matching por IoU-BEV rotado a 0.25 (default de AB3DMOT 3D). Score de la detección
= nº de puntos móviles de la caja (proxy de confianza del GT-seg).

Uso (contenedor):
    /opt/conda/envs/mira/bin/python -u -m tracker.tracking.amota_ab3dmot \
        --detections tracker/results/detections_gtseg_val.pkl
"""
import argparse
import os
import pickle

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracker.tracking.gt_tracks import GtTrackLoader, read_clip_frames
from tracker.tracking.gallery_tracker import GalleryTracker
from tracker.detection.metrics_3d import compute_rotated_iou_2d

VAL_CLIPS = ["delft_1", "delft_10", "delft_14", "delft_22"]
CLIPS_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "clips")


def match_frame(gt_boxes, gt_ids, pr_boxes, pr_ids, last_match, iou_thr):
    """Matching húngaro por IoU-BEV a `iou_thr`. Devuelve tp, fp, fn, idsw, sum_iou."""
    ng, npd = len(gt_boxes), len(pr_boxes)
    if ng == 0:
        return 0, npd, 0, 0, 0.0
    if npd == 0:
        return 0, 0, ng, 0, 0.0
    iou = np.zeros((ng, npd))
    for i in range(ng):
        for j in range(npd):
            iou[i, j] = compute_rotated_iou_2d(gt_boxes[i], pr_boxes[j])
    ri, ci = linear_sum_assignment(-iou)
    tp = idsw = 0
    s_iou = 0.0
    matched_g, matched_p = set(), set()
    for i, j in zip(ri, ci):
        if iou[i, j] >= iou_thr:
            tp += 1; s_iou += iou[i, j]
            matched_g.add(i); matched_p.add(j)
            gid = int(gt_ids[i]); pid = int(pr_ids[j])
            if gid in last_match and last_match[gid] != pid:
                idsw += 1
            last_match[gid] = pid
    fp = npd - len(matched_p)
    fn = ng - len(matched_g)
    return tp, fp, fn, idsw, s_iou


def cargar_tracks_ratrack(result_dir, clips):
    """Carga los tracks de RaTrack (formato point-cloud, frame radar).
    Línea: NA flag -1 -1 score track_id  x1 y1 z1 x2 y2 z2 ...
    Estima una caja axis-aligned por track (como su estimate_3d_box_from_points).
    Devuelve {frame_global: [(tid, box[7], score)]}."""
    import glob
    out = {}
    for clip in clips:
        for txt in sorted(glob.glob(os.path.join(result_dir, clip, "*.txt"))):
            fid = int(os.path.splitext(os.path.basename(txt))[0])
            rows = []
            for line in open(txt):
                p = line.split()
                if len(p) < 6:
                    continue
                score = float(p[4]); tid = int(p[5])
                vals = np.array([float(x) for x in p[6:]], np.float32)
                k = (len(vals) // 3) * 3
                pts = vals[:k].reshape(-1, 3)
                if len(pts) < 1:
                    continue
                c = (pts.min(0) + pts.max(0)) / 2
                sz = np.maximum(pts.max(0) - pts.min(0), 0.3)
                box = np.array([c[0], c[1], c[2], sz[0], sz[1], sz[2], 0.0], np.float32)
                rows.append((tid, box, score))
            out[fid] = rows
    return out


def eval_ratrack_threshold(tracks, gt_cache, clips, score_thr, iou_thr):
    """Evalúa tracks YA producidos (RaTrack) filtrando por score. Sin re-trackear."""
    TP = FP = FN = IDSW = NGT = 0
    s_iou = 0.0
    for clip in clips:
        frames = read_clip_frames(os.path.join(CLIPS_DIR, f"{clip}.txt"))
        last = {}
        for f in frames:
            rows = [r for r in tracks.get(int(f), []) if r[2] >= score_thr]
            if rows:
                ob = np.stack([r[1] for r in rows]); oi = np.array([r[0] for r in rows], int)
            else:
                ob = np.zeros((0, 7), np.float32); oi = np.zeros(0, int)
            gb, gi = gt_cache[int(f)]
            NGT += len(gb)
            tp, fp, fn, idsw, si = match_frame(gb, gi, ob, oi, last, iou_thr)
            TP += tp; FP += fp; FN += fn; IDSW += idsw; s_iou += si
    return dict(TP=TP, FP=FP, FN=FN, IDSW=IDSW, NGT=NGT, sIoU=s_iou)


def eval_a_threshold(dets, gt_cache, clips, score_thr, iou_thr, inject_fp=0.0, rng=None,
                     fov_gl=None):
    """Filtra detecciones por score, trackea y acumula TP/FP/FN/IDSW/num_gt.
    `gt_cache`: {frame: (gt_boxes, gt_ids)} precargado.
    `inject_fp`: FPs sintéticos por frame (fracción de nº de GT).
    `fov_gl`: GtTrackLoader para filtrar detecciones fuera del FOV (o None)."""
    TP = FP = FN = IDSW = NGT = 0
    s_iou = 0.0
    for clip in clips:
        frames = read_clip_frames(os.path.join(CLIPS_DIR, f"{clip}.txt"))
        tk = GalleryTracker(max_age=10, matching_threshold=0.3, use_appearance=False)
        last = {}
        for f in frames:
            d = dets.get(int(f), None)
            if d is None or len(d["boxes"]) == 0:
                db = np.zeros((0, 7), np.float32); npts = None
            else:
                sc = np.asarray(d["num_points"], float)
                keep = sc >= score_thr
                db = np.asarray(d["boxes"], np.float32)[keep]
                npts = sc[keep]
                if fov_gl is not None and len(db):
                    fm = fov_gl.fov_mask(db, f)
                    db, npts = db[fm], npts[fm]
            gb, gi = gt_cache[int(f)]
            if inject_fp > 0 and rng is not None and len(gb):
                nfp = rng.poisson(inject_fp * len(gb))
                if nfp > 0:                       # cajas FP lejos del GT
                    fpb = np.zeros((nfp, 7), np.float32)
                    fpb[:, 0] = rng.uniform(0, 60, nfp)
                    fpb[:, 1] = rng.uniform(-25, 25, nfp)
                    fpb[:, 3:6] = np.array([2.0, 1.0, 1.5])
                    db = np.vstack([db, fpb]) if len(db) else fpb
                    npts = np.concatenate([npts, np.full(nfp, score_thr)]) if npts is not None else None
            ob, oi = tk.update(db, None, npts)
            NGT += len(gb)
            tp, fp, fn, idsw, si = match_frame(gb, gi, ob, oi, last, iou_thr)
            TP += tp; FP += fp; FN += fn; IDSW += idsw; s_iou += si
    return dict(TP=TP, FP=FP, FN=FN, IDSW=IDSW, NGT=NGT, sIoU=s_iou)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/project/view_of_delft_PUBLIC")
    ap.add_argument("--detections", default=None)
    ap.add_argument("--ratrack-results", default=None,
                    help="dir de results de RaTrack (evalúa SUS tracks en este protocolo)")
    ap.add_argument("--iou", type=float, default=0.25, help="umbral IoU (AB3DMOT 3D=0.25)")
    ap.add_argument("--recall-points", type=int, default=40)
    ap.add_argument("--explain", action="store_true",
                    help="imprime la descomposición término-por-término")
    ap.add_argument("--inject-fp", type=float, default=0.0,
                    help="FPs sintéticos por frame (fracción del nº de GT)")
    ap.add_argument("--gt-min-radar-points", type=int, default=0,
                    help="protocolo RaTrack: solo GT con ≥N puntos de radar")
    ap.add_argument("--gt-moving-only", action="store_true",
                    help="protocolo RaTrack: solo objetos GT en movimiento")
    ap.add_argument("--fov-only", action="store_true",
                    help="excluye GT y detecciones fuera del FOV de la cámara")
    ap.add_argument("--clips", nargs="+", default=VAL_CLIPS)
    args = ap.parse_args()

    es_ratrack = args.ratrack_results is not None
    gl = GtTrackLoader(args.dataset, min_radar_points=args.gt_min_radar_points,
                       fov_only=args.fov_only, moving_only=args.gt_moving_only)

    # cachear GT por frame una vez (se reusa en todo el barrido)
    gt_cache = {}
    for clip in args.clips:
        for f in read_clip_frames(os.path.join(CLIPS_DIR, f"{clip}.txt")):
            gb, gi, _ = gl.load_frame(f)
            gt_cache[int(f)] = (gb, gi)
    print(f"[amota] GT cacheado: {len(gt_cache)} frames", flush=True)

    if es_ratrack:
        rt_tracks = cargar_tracks_ratrack(args.ratrack_results, args.clips)
        all_sc = np.array([r[2] for v in rt_tracks.values() for r in v], float)
        lo, hi = (all_sc.min(), all_sc.max()) if len(all_sc) else (0, 1)
        thresholds = np.linspace(lo, hi, args.recall_points)
        eval_fn = lambda thr: eval_ratrack_threshold(rt_tracks, gt_cache, args.clips, thr, args.iou)
    else:
        dets = pickle.load(open(args.detections, "rb"))
        all_sc = np.array([n for v in dets.values() for n in v.get("num_points", [])], float)
        smax = all_sc.max() if len(all_sc) else 1
        thresholds = np.linspace(1, smax, args.recall_points)
        rng = np.random.default_rng(0) if args.inject_fp > 0 else None
        fov_gl = gl if args.fov_only else None
        eval_fn = lambda thr: eval_a_threshold(dets, gt_cache, args.clips, thr, args.iou,
                                               inject_fp=args.inject_fp, rng=rng,
                                               fov_gl=fov_gl)

    if args.explain:
        print(f"\n{'thr':>7} {'recall':>7} {'TP':>6} {'FP':>5} {'FN':>7} {'IDSW':>5} "
              f"{'(1-r)·ngt':>10} {'MOTA':>7} {'sMOTA':>7}")
        print("-" * 74)

    smota_list, mota_list, recalls = [], [], []
    for thr in thresholds:
        r = eval_fn(thr)
        ng = max(r["NGT"], 1)
        recall = r["TP"] / ng
        errors = r["IDSW"] + r["FP"] + r["FN"]
        mota = 1 - errors / ng
        if recall > 0:
            smota = max(0.0, 1 - (errors - (1 - recall) * ng) / (recall * ng))
        else:
            smota = 0.0
        smota_list.append(smota); mota_list.append(max(0.0, mota)); recalls.append(recall)
        if args.explain:
            print(f"{thr:7.3f} {recall:7.3f} {r['TP']:6d} {r['FP']:5d} {r['FN']:7d} "
                  f"{r['IDSW']:5d} {(1-recall)*ng:10.0f} {100*mota:7.1f} {100*smota:7.1f}")

    sAMOTA = float(np.mean(smota_list)) * 100
    AMOTA = float(np.mean(mota_list)) * 100
    r0 = eval_fn(thresholds[0])                       # con todas las detecciones
    ng = max(r0["NGT"], 1)
    fuente = "RaTrack (sus tracks)" if es_ratrack else "NUESTRO tracker"
    print(f"\n{'='*60}\nPROTOCOLO AB3DMOT  IoU={args.iou}  |  {fuente}\n{'='*60}")
    print(f"  sAMOTA (escalado por recall) : {sAMOTA:.2f}%")
    print(f"  AMOTA                        : {AMOTA:.2f}%")
    print(f"  recall máx (thr=1)           : {100*r0['TP']/ng:.1f}%")
    print(f"  MOTP (IoU medio de matches)  : {100*r0['sIoU']/max(r0['TP'],1):.1f}%")
    print(f"  TP/FP/FN (thr=1)             : {r0['TP']}/{r0['FP']}/{r0['FN']}  IDSW={r0['IDSW']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
