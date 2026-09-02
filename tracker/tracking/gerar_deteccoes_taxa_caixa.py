"""GT de tracking + GT de segmentacion siempre a 10 Hz; la CAJA baja de tasa.

Escenario de sistema realista: un detector 3D de cajas es caro y corre a 5 Hz,
mientras que la segmentacion por punto es barata y corre a 10 Hz. En los cuadros
sin caja el objeto NO desaparece — sigue existiendo por su segmentacion — pero
su caja hay que reconstruirla de los puntos, que es peor.

  * cuadro con caja  (k % N == 0): se emite la caja GT (perfecta).
  * cuadro sin caja              : se emite una caja ajustada a los puntos de
    radar que la segmentacion GT asigna a ese objeto (PCA + prior de clase si
    hay pocos puntos), conservando su identidad.

Asi se mide cuanto cuesta refrescar la caja a la mitad de la tasa, con deteccion
e identidad perfectas.
"""
import os, pickle, sys
import numpy as np
sys.path.insert(0, "/project")
from tracker.tracking.gt_tracks import GtTrackLoader
from tracker.detection.box_proposal import fit_oriented_box, _prior_por_span
from external.vod.frame import FrameDataLoader

salida, modo = sys.argv[1], sys.argv[2]
# modo "N"    -> caja cada N cuadros (intervalo N*0.1 s)
# modo "0.15" -> patron alternado 2,1,2,1... (dos de cada tres): media 0.15 s,
#                porque 0.15 s no cae en la grilla de 0.1 s de VoD.
alterno = (modo == "0.15")
cada = 1 if alterno else int(modo)
CLIPS_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "clips")
VAL = ["delft_1", "delft_10", "delft_14", "delft_22"]
gl = GtTrackLoader("/project/view_of_delft_PUBLIC", moving_only=True)

def dentro(pts, b, margen=0.25):
    dx, dy = pts[:, 0] - b[0], pts[:, 1] - b[1]
    c, s = np.cos(-b[6]), np.sin(-b[6])
    rx, ry = dx * c - dy * s, dx * s + dy * c
    return (np.abs(rx) <= b[3] / 2 + margen) & (np.abs(ry) <= b[4] / 2 + margen)

frames = []
for c in VAL:                       # split de validacao, das listas do proprio repo
    p = os.path.join(CLIPS_DIR, c + ".txt")
    if os.path.exists(p):
        frames += [int(l.split()[0]) for l in open(p) if l.strip()]
frames = sorted(frames)

dets = {}
n_gtbox = n_seg = n_perdidos = 0
for k, f in enumerate(frames):
    b, i, _ = gl.load_frame(f)
    b = np.asarray(b, np.float32).reshape(-1, 7)
    con_caja = (k % 3 != 1) if alterno else (k % cada == 0)
    cajas, ids = [], []
    if con_caja:
        cajas = list(b); ids = list(np.asarray(i, int)); n_gtbox += len(b)
    elif len(b):
        try:
            radar = FrameDataLoader(kitti_locations=gl.loc,
                                    frame_number="%05d" % f).radar_data[:, :3]
        except Exception:
            radar = np.zeros((0, 3), np.float32)
        for caja_gt, oid in zip(b, np.asarray(i, int)):
            m = dentro(radar, caja_gt) if len(radar) else np.zeros(0, bool)
            if m.sum() < 1:            # sin retornos: la segmentacion no lo ve
                n_perdidos += 1
                continue
            c = fit_oriented_box(radar[m])
            if int(m.sum()) < 12:
                l, w, h = _prior_por_span(float(c[3]))
                c[3], c[4], c[5] = l, w, h
            cajas.append(c); ids.append(int(oid)); n_seg += 1
    dets[f] = {"boxes": np.asarray(cajas, np.float32).reshape(-1, 7),
               "track_ids": np.asarray(ids, int),
               "num_points": np.ones(len(cajas), int)}
pickle.dump(dets, open(salida, "wb"))
etq = "alternado 2,1 (media 0.15 s, 6.7 Hz)" if alterno else "cada %d cuadros (%.2f s, %.1f Hz)" % (cada, 0.1*cada, 10.0/cada)
print("[gt-box-hz] caja %s | cajas GT=%d, reconstruidas por segmentacion=%d, "
      "sin retorno=%d -> %s" % (etq, n_gtbox, n_seg, n_perdidos, salida))
