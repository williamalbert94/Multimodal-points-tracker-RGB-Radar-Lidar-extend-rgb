"""Evalúa las métricas de tracking (MOTA / sAMOTA / IDF1 / MOTP) sobre los clips
de validación de VoD.

Modos (`--mode`)
----------------
sanity          GT alimentado como predicción perfecta. Prueba de cordura del
                MÉTRICO: debe dar MOTA≈100, IDF1≈100, IDSW=0.
baseline-gtdet  Tracker IoU (solo geometría) sobre cajas GT. Piso de asociación.
gallery-gtdet   GalleryTracker multi-cue sobre cajas GT. Con `--no-appearance`
                mide el aporte de movimiento/galería; sin la bandera usaría
                apariencia (embeddings, aún no conectados sobre GT-det).

Contrato del tracker: `update(boxes, embeddings, extra) -> (out_boxes, out_ids)`.

Uso:
    python -m tracker.tracking.eval_tracking --mode baseline-gtdet
    python -m tracker.tracking.eval_tracking --mode gallery-gtdet --no-appearance
"""
import argparse
import os

import numpy as np

from tracker.tracking.gt_tracks import GtTrackLoader, read_clip_frames
from tracker.tracking.mot_metrics import MOTMetricsAccumulator
from tracker.tracking.simple_tracker import SimpleIoUTracker
from tracker.tracking.gallery_tracker import GalleryTracker

VAL_CLIPS = ["delft_1", "delft_10", "delft_14", "delft_22"]
CLIPS_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "clips")


def eval_clips(dataset_path, clips_dir, clips, gt_loader,
               detection_source, make_tracker, frame_offset_mult=100000,
               id_offset=10_000_000):
    """Recorre los clips y acumula métricas.

    `detection_source(frame_id, gt_boxes, gt_ids)` -> (det_boxes, embeddings, extra).
    `make_tracker()` -> tracker fresco por clip con
        `.update(det_boxes, embeddings, extra) -> (out_boxes, out_ids)`.

    `frame_offset_mult`: separa los frame_id entre clips para que las secuencias
    no se mezclen (el acumulador identifica tracks por (clip, frame)).
    Si `gt_loader.fov_only`, filtra también las detecciones fuera del FOV.
    """
    fov_only = getattr(gt_loader, "fov_only", False)
    acc = MOTMetricsAccumulator()
    for ci, clip in enumerate(clips):
        clip_path = os.path.join(clips_dir, f"{clip}.txt")
        if not os.path.exists(clip_path):
            print(f"  [skip] clip no encontrado: {clip_path}")
            continue
        frames = read_clip_frames(clip_path)
        tracker = make_tracker()
        n_gt = n_det = 0
        for f in frames:
            gt_boxes, gt_ids, _ = gt_loader.load_frame(f)
            det_boxes, embeddings, extra = detection_source(f, gt_boxes, gt_ids)
            if fov_only and len(det_boxes):
                fm = gt_loader.fov_mask(det_boxes, f)
                det_boxes = det_boxes[fm]
                if embeddings is not None:
                    embeddings = embeddings[fm]
                if extra is not None and hasattr(extra, "__len__") and len(extra) == len(fm):
                    extra = np.asarray(extra)[fm]
            out_boxes, out_ids = tracker.update(det_boxes, embeddings, extra)
            # Los ids de tracker reinician en 1 por clip: se desplazan a un rango
            # disjunto por clip para que no colisionen entre secuencias (si no,
            # el pred_id=1 de dos clips se fusionaría y arruinaría IDF1).
            out_ids = np.asarray(out_ids, int)
            if len(out_ids):
                out_ids = out_ids + (ci + 1) * id_offset
            n_gt += len(gt_ids)
            n_det += len(det_boxes)
            acc.update(
                frame_id=ci * frame_offset_mult + f,
                gt_boxes=gt_boxes, gt_ids=gt_ids,
                pred_boxes=out_boxes, pred_ids=out_ids,
            )
        print(f"  {clip}: {len(frames)} frames, {n_gt} GT / {n_det} detecciones")
    return acc.compute_metrics()


# ── fuentes de detección ─────────────────────────────────────────────────────
def gt_detection_source(frame_id, gt_boxes, gt_ids):
    """Cajas GT como detecciones (sin ids ni embeddings)."""
    return gt_boxes.copy(), None, None


def sanity_detection_source(frame_id, gt_boxes, gt_ids):
    """GT como detección perfecta; pasa los gt_ids por el slot `extra`."""
    return gt_boxes.copy(), None, gt_ids.copy()


class FileDetectionSource:
    """Lee detecciones reales precomputadas (pickle {frame: {boxes, scores}})."""
    def __init__(self, path):
        import pickle
        with open(path, "rb") as f:
            self.dets = pickle.load(f)
        n = sum(len(v["boxes"]) for v in self.dets.values())
        print(f"[det] {len(self.dets)} frames, {n} cajas desde {path}")

    def __call__(self, frame_id, gt_boxes, gt_ids):
        d = self.dets.get(int(frame_id))
        if d is None or len(d["boxes"]) == 0:
            return np.zeros((0, 7), np.float32), None, None
        npts = np.asarray(d["num_points"]) if "num_points" in d else None
        return np.asarray(d["boxes"], np.float32), None, npts


class AppearanceDetectionSource:
    """Detecciones GT-seg + embeddings Re-ID (cabeza entrenada sobre features Q)."""
    def __init__(self, path, head_path):
        import pickle
        import torch
        from tracker.tracking.reid_head import ReIDHead, features_a_tensor
        self.torch = torch
        self._feat = features_a_tensor
        with open(path, "rb") as f:
            self.dets = pickle.load(f)
        ck = torch.load(head_path, map_location="cuda")
        self.head = ReIDHead(appear_dim=ck["appear_dim"],
                             embedding_dim=ck["embedding_dim"]).to("cuda")
        self.head.load_state_dict(ck["model"])
        self.head.eval()
        n = sum(len(v["boxes"]) for v in self.dets.values())
        print(f"[det+app] {len(self.dets)} frames, {n} cajas | cabeza {head_path}")

    def __call__(self, frame_id, gt_boxes, gt_ids):
        d = self.dets.get(int(frame_id))
        if d is None or len(d["boxes"]) == 0:
            return np.zeros((0, 7), np.float32), None, None
        appear, box = self._feat(d, "cuda")
        with self.torch.no_grad():
            emb = self.head(appear, box).cpu().numpy()
        npts = np.asarray(d["num_points"]) if "num_points" in d else None
        return np.asarray(d["boxes"], np.float32), emb, npts


# ── tracker identidad (sanity) ────────────────────────────────────────────────
class _IdentityTracker:
    """Devuelve las cajas con los gt_ids tal cual (recibidos en `extra`)."""
    def update(self, boxes, embeddings=None, extra=None):
        return boxes, np.asarray(extra).astype(int)


def print_metrics(m, titulo):
    print(f"\n{'='*56}\n{titulo}\n{'='*56}")
    print(f"  MOTA   : {m['MOTA']:.2f}%")
    print(f"  sAMOTA : {m['sAMOTA']:.2f}%")
    print(f"  AMOTA  : {m['AMOTA']:.2f}%")
    print(f"  IDF1   : {m['IDF1']:.2f}%")
    print(f"  MOTP   : {m['MOTP']:.2f}%")
    print(f"  TP/FP/FN : {m['TP']} / {m['FP']} / {m['FN']}")
    print(f"  ID-switches : {m['ID_switches']}")
    print(f"  MT/PT/ML : {m['MT']:.1f}% / {m['PT']:.1f}% / {m['ML']:.1f}%")
    print(f"{'='*56}\n")


def main():
    ap = argparse.ArgumentParser()
    default_ds = "/project/view_of_delft_PUBLIC"
    if not os.path.exists(default_ds):
        default_ds = "/local2/local/williamramirez/view_of_delft_PUBLIC"
    ap.add_argument("--dataset", default=default_ds)
    ap.add_argument("--clips-dir", default=CLIPS_DIR)
    ap.add_argument("--mode",
                    choices=["sanity", "baseline-gtdet", "gallery-gtdet",
                             "gallery-det", "baseline-det", "gallery-det-app"],
                    default="sanity")
    ap.add_argument("--detections", default=None,
                    help="pickle de detecciones reales (modos *-det)")
    ap.add_argument("--reid-head", default=None,
                    help="cabeza Re-ID entrenada (.pth) para gallery-det-app")
    ap.add_argument("--iou-threshold", type=float, default=0.1,
                    help="baseline: umbral IoU de asociación")
    ap.add_argument("--max-age", type=int, default=3)
    ap.add_argument("--match-threshold", type=float, default=0.3,
                    help="gallery: umbral de score multi-cue")
    ap.add_argument("--no-appearance", action="store_true",
                    help="gallery: desactiva la señal de apariencia (ablation)")
    ap.add_argument("--clips", nargs="+", default=VAL_CLIPS)
    ap.add_argument("--gt-classes", nargs="+", default=None,
                    help="filtra el GT a estos tipos (ej: Car Cyclist). None=todos")
    ap.add_argument("--gt-min-radar-points", type=int, default=0,
                    help="protocolo RaTrack: solo GT con ≥N puntos de radar")
    ap.add_argument("--fov-only", action="store_true",
                    help="excluye GT y detecciones fuera del FOV de la cámara")
    args = ap.parse_args()

    gt_loader = GtTrackLoader(args.dataset, keep_types=args.gt_classes,
                              min_radar_points=args.gt_min_radar_points,
                              fov_only=args.fov_only)
    if args.fov_only:
        print("[gt] solo objetos dentro del FOV de la cámara")
    if args.gt_classes:
        print(f"[gt] filtrado a clases: {args.gt_classes}")
    if args.gt_min_radar_points:
        print(f"[gt] solo objetos con ≥{args.gt_min_radar_points} puntos de radar")

    if args.mode == "sanity":
        print("Prueba de cordura: GT alimentado como predicción perfecta.")
        m = eval_clips(args.dataset, args.clips_dir, args.clips, gt_loader,
                       sanity_detection_source, _IdentityTracker)
        print_metrics(m, "SANITY (GT como predicción) — esperado ~100% / IDSW=0")
        ok = (m["MOTA"] > 99.0 and m["IDF1"] > 99.0 and m["ID_switches"] == 0)
        print("[sanity] MÉTRICO VÁLIDO" if ok else "[sanity] REVISAR MÉTRICO/CARGADOR")
        return

    if args.mode == "baseline-gtdet":
        print(f"Baseline IoU sobre cajas GT. iou_thr={args.iou_threshold} "
              f"max_age={args.max_age}")
        make_trk = lambda: SimpleIoUTracker(iou_threshold=args.iou_threshold,
                                            max_age=args.max_age)
        m = eval_clips(args.dataset, args.clips_dir, args.clips, gt_loader,
                       gt_detection_source, make_trk)
        print_metrics(m, "BASELINE IoU (detección=GT) — piso de asociación")
        return

    if args.mode == "gallery-gtdet":
        use_app = not args.no_appearance
        etiqueta = "con apariencia" if use_app else "SIN apariencia (movimiento/galería)"
        print(f"GalleryTracker sobre cajas GT [{etiqueta}]. "
              f"max_age={args.max_age} match_thr={args.match_threshold}")
        make_trk = lambda: GalleryTracker(max_age=args.max_age,
                                          matching_threshold=args.match_threshold,
                                          use_appearance=use_app)
        m = eval_clips(args.dataset, args.clips_dir, args.clips, gt_loader,
                       gt_detection_source, make_trk)
        print_metrics(m, f"GALLERY (detección=GT, {etiqueta})")
        return

    if args.mode == "gallery-det-app":
        assert args.detections and args.reid_head, "usar --detections y --reid-head"
        det_src = AppearanceDetectionSource(args.detections, args.reid_head)
        print(f"GalleryTracker CON apariencia (ReID entrenado). "
              f"max_age={args.max_age} match_thr={args.match_threshold}")
        make_trk = lambda: GalleryTracker(max_age=args.max_age,
                                          matching_threshold=args.match_threshold,
                                          use_appearance=True)
        m = eval_clips(args.dataset, args.clips_dir, args.clips, gt_loader,
                       det_src, make_trk)
        print_metrics(m, "GALLERY (GT-seg det, CON apariencia ReID)")
        return

    # ── modos con DETECCIÓN REAL (pickle precomputado) ────────────────────────
    if args.mode in ("gallery-det", "baseline-det"):
        assert args.detections, "usar --detections <pickle>"
        det_src = FileDetectionSource(args.detections)
        if args.mode == "baseline-det":
            print(f"Baseline IoU sobre detección real. iou_thr={args.iou_threshold}")
            make_trk = lambda: SimpleIoUTracker(iou_threshold=args.iou_threshold,
                                                max_age=args.max_age)
            titulo = "BASELINE IoU (detección real)"
        else:
            use_app = not args.no_appearance
            etq = "con apariencia" if use_app else "SIN apariencia"
            print(f"GalleryTracker sobre detección real [{etq}]. "
                  f"max_age={args.max_age} match_thr={args.match_threshold}")
            make_trk = lambda: GalleryTracker(max_age=args.max_age,
                                              matching_threshold=args.match_threshold,
                                              use_appearance=use_app)
            titulo = f"GALLERY (detección real, {etq})"
        m = eval_clips(args.dataset, args.clips_dir, args.clips, gt_loader,
                       det_src, make_trk)
        print_metrics(m, titulo)
        return


if __name__ == "__main__":
    main()
