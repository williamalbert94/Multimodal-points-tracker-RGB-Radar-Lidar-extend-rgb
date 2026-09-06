"""Gallery-based multi-cue tracker (adaptado de la referencia que ya funcionó).

Cada track mantiene una GALERÍA con: embedding Re-ID (+ EMA e historial),
geometría de la caja, densidad de puntos, movimiento (velocidad/dirección) y
contexto espacial (distancias a otros objetos). La asociación entre frames
combina esas señales en una matriz de costo y resuelve con Hungarian.

Diferencias con la referencia:
  - Movimiento INTERNO: cada track predice su centro con velocidad constante
    (`predict()`), en vez de recibir un `motion_dict` externo.
  - Geometría con IoU-BEV ROTADO (considera yaw), no axis-aligned.
  - `_next_id` por instancia (no de clase): cada clip arranca limpio.
  - `update()` devuelve (out_boxes [K,7], out_ids [K]) para encajar con el eval.

Con `embeddings=None` la señal de apariencia se desactiva y los pesos se
renormalizan sobre las señales disponibles (permite el ablation sin ReID).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from tracker.detection.metrics_3d import compute_rotated_iou_2d


class TrackGallery:
    """Estado/galería de un track individual."""

    def __init__(self, track_id, box, embedding=None, num_points=0, ema_alpha=0.9):
        self.id = track_id
        # Peso del descriptor ACUMULADO frente al de la última observación.
        # 0.9 = galería con EMA (comportamiento original); 0.0 = el descriptor es
        # siempre el de la última observación, sin memoria de apariencia.
        self.ema_alpha = float(ema_alpha)
        # apariencia
        self.embedding = embedding
        self.embedding_history = [embedding] if embedding is not None else []
        # geometría
        self.box = box.copy()
        self.center = box[:3].copy()
        self.size = box[3:6].copy()
        self.yaw = box[6]
        # densidad
        self.num_points = num_points
        self.num_points_history = [num_points]
        self.avg_num_points = float(num_points)
        # movimiento
        self.velocity = np.zeros(3)
        self.direction_2d = 0.0
        self.speed = 0.0
        # contexto espacial: {other_id: dist_bev}
        self.distances_to_others = {}
        # manejo
        self.hits = 1
        self.age = 0
        self.frames_since_update = 0

    def predict(self, dt=0.1):
        """Centro predicho por velocidad constante."""
        return self.center + self.velocity * dt

    def update(self, box, embedding=None, num_points=0, distances=None, dt=0.1):
        disp = box[:3] - self.center
        self.velocity = disp / dt
        self.speed = float(np.linalg.norm(self.velocity))
        self.direction_2d = float(np.arctan2(disp[1], disp[0]))

        self.box = box.copy()
        self.center = box[:3].copy()
        self.size = box[3:6].copy()
        self.yaw = box[6]

        if embedding is not None:
            if self.embedding is None:
                self.embedding = embedding.copy()
            else:
                a = self.ema_alpha
                self.embedding = a * self.embedding + (1 - a) * embedding
            self.embedding_history.append(embedding)
            self.embedding_history = self.embedding_history[-10:]

        self.num_points = num_points
        self.num_points_history.append(num_points)
        self.num_points_history = self.num_points_history[-10:]
        self.avg_num_points = float(np.mean(self.num_points_history))

        if distances is not None:
            self.distances_to_others = distances

        self.hits += 1
        self.frames_since_update = 0
        self.age += 1

    def mark_missed(self):
        self.frames_since_update += 1
        self.age += 1

    def should_delete(self, max_age):
        return self.frames_since_update >= max_age


def _bev_distances(boxes, ids):
    """{id_i: {id_j: dist_bev}} entre centros en BEV."""
    centers = boxes[:, :2]
    out = {}
    for i, ti in enumerate(ids):
        d = {}
        for j, tj in enumerate(ids):
            if i == j:
                continue
            d[int(tj)] = float(np.linalg.norm(centers[i] - centers[j]))
        out[int(ti)] = d
    return out


class GalleryTracker:
    """Tracker multi-cue basado en galería."""

    def __init__(self, max_age=10, min_hits=1, matching_threshold=0.3,
                 weight_appearance=0.30, weight_geometry=0.20, weight_density=0.10,
                 weight_motion=0.20, weight_spatial=0.20, use_appearance=True,
                 coast_frames=0, ema_alpha=0.9):
        self.max_age = max_age
        # Ver TrackGallery: 0.9 = galería con EMA; 0.0 = última observación.
        self.ema_alpha = float(ema_alpha)
        self.min_hits = min_hits
        self.matching_threshold = matching_threshold
        # Cuántos frames se sigue EMITIENDO una trayectoria sin detección, con la
        # posición extrapolada por velocidad constante. Con 0 la trayectoria queda
        # viva internamente pero invisible en la salida, así que un hueco corto de
        # detección se contabiliza como falso negativo aunque la asociación no se
        # haya perdido.
        self.coast_frames = int(coast_frames)
        self.use_appearance = use_appearance

        w = {"app": weight_appearance, "geom": weight_geometry,
             "dens": weight_density, "motion": weight_motion,
             "spatial": weight_spatial}
        if not use_appearance:
            w["app"] = 0.0
        s = sum(w.values())
        self.w = {k: v / s for k, v in w.items()}   # renormalizados a 1

        self.tracks = []
        self._next_id = 1
        self.frame_count = 0

    # ── API ──────────────────────────────────────────────────────────────────
    def update(self, boxes, embeddings=None, num_points=None):
        """boxes [M,7], embeddings [M,D] o None, num_points [M] o None.
        Devuelve (out_boxes [K,7], out_ids [K])."""
        boxes = np.asarray(boxes, np.float32).reshape(-1, 7)
        if not self.use_appearance:
            embeddings = None
        M = len(boxes)
        self.frame_count += 1

        # primer frame: inicializa
        if len(self.tracks) == 0:
            ids = [self._nuevo(boxes[i],
                               embeddings[i] if embeddings is not None else None,
                               num_points[i] if num_points is not None else 0)
                   for i in range(M)]
            self._set_distances(boxes, [t.id for t in self.tracks])
            return self._salida()

        N = len(self.tracks)
        cost = self._cost_matrix(boxes, embeddings, num_points)

        matched = []
        if N > 0 and M > 0:
            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                if (1.0 - cost[r, c]) >= self.matching_threshold:
                    matched.append((r, c))

        m_trk = {r for r, _ in matched}
        m_det = {c for _, c in matched}

        # distancias BEV actuales (con ids temporales = índice de detección)
        det_dists_tmp = _bev_distances(boxes, list(range(M)))

        for r, c in matched:
            emb = embeddings[c] if embeddings is not None else None
            npt = num_points[c] if num_points is not None else 0
            # mapear ids temporales de vecinos a ids reales de tracks matched
            real_d = {}
            for tmp_id, dist in det_dists_tmp[c].items():
                if tmp_id in m_det:
                    real_idx = [rr for rr, cc in matched if cc == tmp_id][0]
                    real_d[self.tracks[real_idx].id] = dist
            self.tracks[r].update(boxes[c], emb, npt, real_d)

        for r in range(N):
            if r not in m_trk:
                self.tracks[r].mark_missed()

        for c in range(M):
            if c not in m_det:
                self._nuevo(boxes[c],
                            embeddings[c] if embeddings is not None else None,
                            num_points[c] if num_points is not None else 0)

        self.tracks = [t for t in self.tracks if not t.should_delete(self.max_age)]
        return self._salida()

    # ── internos ─────────────────────────────────────────────────────────────
    def _nuevo(self, box, emb, npts):
        t = TrackGallery(self._next_id, box, emb, npts, self.ema_alpha)
        self._next_id += 1
        self.tracks.append(t)
        return t.id

    def _set_distances(self, boxes, ids):
        d = _bev_distances(boxes, ids)
        for t in self.tracks:
            if t.id in d:
                t.distances_to_others = d[t.id]

    def _salida(self):
        out_boxes, out_ids = [], []
        for t in self.tracks:
            if t.hits < self.min_hits:
                continue
            if t.frames_since_update == 0:
                out_boxes.append(t.box)
                out_ids.append(t.id)
            elif t.frames_since_update <= self.coast_frames:
                caja = t.box.copy()
                caja[:3] = t.center + t.velocity * (0.1 * t.frames_since_update)
                out_boxes.append(caja)
                out_ids.append(t.id)
        if not out_boxes:
            return np.zeros((0, 7), np.float32), np.zeros(0, int)
        return np.stack(out_boxes).astype(np.float32), np.array(out_ids, int)

    def _cost_matrix(self, boxes, embeddings, num_points):
        N, M = len(self.tracks), len(boxes)
        cost = np.ones((N, M), np.float64)
        det_dists_tmp = _bev_distances(boxes, list(range(M)))

        for i, trk in enumerate(self.tracks):
            pred_c = trk.predict()
            for j in range(M):
                s = {}

                # 1. apariencia (cosine)
                if self.use_appearance and trk.embedding is not None and embeddings is not None:
                    e1 = trk.embedding / (np.linalg.norm(trk.embedding) + 1e-8)
                    e2 = embeddings[j] / (np.linalg.norm(embeddings[j]) + 1e-8)
                    s["app"] = (float(np.dot(e1, e2)) + 1) / 2
                else:
                    s["app"] = 0.5

                # 2. geometría: IoU-BEV rotado + similitud de tamaño
                iou = compute_rotated_iou_2d(trk.box, boxes[j])
                size_diff = np.linalg.norm(trk.size - boxes[j][3:6])
                s["geom"] = 0.7 * iou + 0.3 * float(np.exp(-size_diff / 2.0))

                # 3. densidad
                if num_points is not None and trk.avg_num_points > 0 and num_points[j] > 0:
                    s["dens"] = (min(num_points[j], trk.avg_num_points) /
                                 max(num_points[j], trk.avg_num_points))
                else:
                    s["dens"] = 0.5

                # 4. movimiento: error entre centro predicho y observado
                err = np.linalg.norm(pred_c - boxes[j][:3])
                mscore = float(np.exp(-0.5 * (err / 1.5) ** 2))
                if trk.speed > 0.5:
                    obs_dir = np.arctan2(boxes[j][1] - trk.center[1],
                                         boxes[j][0] - trk.center[0])
                    ad = abs(trk.direction_2d - obs_dir)
                    ad = min(ad, 2 * np.pi - ad)
                    mscore = 0.7 * mscore + 0.3 * float(np.exp(-ad))
                s["motion"] = mscore

                # 5. espacial: consistencia de distancias a otros objetos
                if len(trk.distances_to_others) > 0:
                    cons = [np.exp(-abs(pd - cd) / 2.0)
                            for pd in trk.distances_to_others.values()
                            for cd in det_dists_tmp[j].values()]
                    s["spatial"] = float(np.mean(cons)) if cons else 0.5
                else:
                    s["spatial"] = 0.5

                total = (self.w["app"] * s["app"] + self.w["geom"] * s["geom"] +
                         self.w["dens"] * s["dens"] + self.w["motion"] * s["motion"] +
                         self.w["spatial"] * s["spatial"])
                cost[i, j] = 1.0 - total
        return cost
