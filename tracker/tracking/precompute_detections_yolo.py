"""Proxy "detector 2D + caja GT": YOLO decide QUE se detecta, el GT pone la CAJA.

Es el analogo exacto del proxy de segmentacion (`precompute_detections_gtseg`),
que mantiene la caja GT y solo la filtra por evidencia del sensor. Aca la
evidencia es un detector 2D sobre la imagen:

  1. YOLO sobre la imagen -> cajas 2D de clases de trafico.
  2. Se emparejan por IoU en el plano de la IMAGEN contra las cajas 2D anotadas
     (`label_2_tracking_2dann`), que vienen alineadas fila a fila con los
     parametros 3D del objeto.
  3. Cada objeto GT emparejado se emite con su CAJA 3D del GT y el score de YOLO.

Mide el techo de un detector 2D real: recall de YOLO x localizacion perfecta.
NO es un detector 3D — no resuelve el paso 2D->3D, lo esquiva a proposito para
aislar cuanto aporta cada etapa.
"""
import argparse, os, pickle, sys
import numpy as np
sys.path.insert(0, "/project")
try:
    from ultralytics import YOLO
except ImportError:                                       # dependencia OPCIONAL
    raise SystemExit(
        "Esta via requer o pacote `ultralytics`, que NAO faz parte do ambiente\n"
        "base (nada do pipeline principal depende dele). Instale-o so se for\n"
        "usar o detector 2D:\n\n    pip install ultralytics\n")
from tracker.tracking.gt_tracks import GtTrackLoader

# clases COCO de trafico -> las que pueden ser objetos moviles en VoD
COCO_TRAFICO = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
                5: "bus", 7: "truck"}

def iou2d(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0

ap = argparse.ArgumentParser()
ap.add_argument("--modelo", default="yolov8x.pt")
ap.add_argument("--score-2d", type=float, default=0.25)
ap.add_argument("--iou-match", type=float, default=0.5)
ap.add_argument("--dataset", default="/project/view_of_delft_PUBLIC")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--union-con", default=None,
                help="pickle de otra via (p.ej. el proxy de segmentacion) para "
                     "emitir la UNION: la camara aporta lo que el radar no ve "
                     "y viceversa")
ap.add_argument("--out", required=True)
a = ap.parse_args()

IMG = os.path.join(a.dataset, "lidar/training/image_2")
ANN2D = os.path.join(a.dataset, "lidar/training/label_2_tracking_2dann")
RT = "/ratrack/src/result/4dmot_runthis"

modelo = YOLO(a.modelo)
gl = GtTrackLoader(a.dataset, moving_only=True)

frames = []
for c in ["delft_1", "delft_10", "delft_14", "delft_22"]:
    d = os.path.join(RT, c)
    if os.path.isdir(d):
        frames += [int(x[:-4]) for x in os.listdir(d) if x.endswith(".txt")]
frames = sorted(frames)
if a.limit:
    frames = frames[:a.limit]
print("[yolo] %s | %d cuadros | score2d=%.2f iou_match=%.2f"
      % (a.modelo, len(frames), a.score_2d, a.iou_match), flush=True)

def caja2d_gt(frame):
    """{obj_id: [x1,y1,x2,y2]} de las anotaciones 2D hechas a mano."""
    p = os.path.join(ANN2D, "%05d.txt" % frame)
    out = {}
    if not os.path.exists(p):
        return out
    for l in open(p):
        c = l.split()
        if len(c) >= 15 and c[0] != "DontCare":
            out[int(c[1])] = np.array([float(v) for v in c[4:8]])
    return out

dets = {}
n_2d = n_match = n_gt = 0
for i, f in enumerate(frames):
    img = os.path.join(IMG, "%05d.jpg" % f)
    gb, gi, _ = gl.load_frame(f)
    n_gt += len(gb)
    if not os.path.exists(img) or len(gb) == 0:
        dets[f] = {"boxes": np.zeros((0, 7), np.float32),
                   "scores": np.zeros(0, np.float32)}
        continue

    r = modelo.predict(img, conf=a.score_2d, verbose=False)[0]
    cajas2d = []
    for b, cls, sc in zip(r.boxes.xyxy.cpu().numpy(),
                          r.boxes.cls.cpu().numpy().astype(int),
                          r.boxes.conf.cpu().numpy()):
        if int(cls) in COCO_TRAFICO:
            cajas2d.append((b, float(sc)))
    n_2d += len(cajas2d)

    g2d = caja2d_gt(f)
    usados = set()
    keep_b, keep_s = [], []
    # greedy por score: cada deteccion 2D reclama el GT con mayor IoU en imagen
    for b, sc in sorted(cajas2d, key=lambda x: -x[1]):
        mejor, mejor_iou = None, a.iou_match
        for k, oid in enumerate(gi):
            if k in usados or int(oid) not in g2d:
                continue
            v = iou2d(b, g2d[int(oid)])
            if v >= mejor_iou:
                mejor, mejor_iou = k, v
        if mejor is not None:
            usados.add(mejor)
            keep_b.append(gb[mejor]); keep_s.append(sc)

    n_match += len(keep_b)
    dets[f] = {"boxes": np.asarray(keep_b, np.float32).reshape(-1, 7),
               "scores": np.asarray(keep_s, np.float32)}
    if (i + 1) % 200 == 0:
        print("  %d/%d  2D=%d  emparejadas=%d / gt=%d"
              % (i + 1, len(frames), n_2d, n_match, n_gt), flush=True)

if a.union_con:
    # Union con otra via que tambien emita cajas GT. Se deduplica por el centro
    # redondeado, que es identico cuando las dos aciertan el mismo objeto.
    otra = pickle.load(open(a.union_con, "rb"))
    n_antes = sum(len(v["boxes"]) for v in dets.values())
    for f in set(dets) | set(otra):
        cajas, scores, vistos = [], [], set()
        for src in (dets.get(f), otra.get(f)):
            if not src:
                continue
            bx = np.asarray(src.get("boxes", np.zeros((0, 7))), np.float32).reshape(-1, 7)
            sc = np.asarray(src.get("scores", src.get("num_points", np.ones(len(bx)))),
                            np.float32).reshape(-1)
            for k in range(len(bx)):
                clave = tuple(np.round(bx[k][:3], 2))
                if clave in vistos:
                    continue
                vistos.add(clave)
                cajas.append(bx[k]); scores.append(float(sc[k]) if k < len(sc) else 1.0)
        dets[f] = {"boxes": np.asarray(cajas, np.float32).reshape(-1, 7),
                   "scores": np.asarray(scores, np.float32)}
    n_union = sum(len(v["boxes"]) for v in dets.values())
    print("[yolo] union con %s: %d -> %d cajas" % (a.union_con, n_antes, n_union), flush=True)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
pickle.dump(dets, open(a.out, "wb"))
print("\n[yolo] %d cuadros | 2D trafico=%d | emparejadas=%d / GT=%d (recall %.4f) -> %s"
      % (len(dets), n_2d, n_match, n_gt, n_match / max(n_gt, 1), a.out), flush=True)
