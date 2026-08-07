"""Tracker baseline por asociación IoU (sin apariencia / sin ReID).

Es el PISO contra el que se mide el aporte del ReID. Solo usa geometría:
en cada frame asocia las detecciones a los tracks activos por IoU-BEV rotado
(asignación húngara), continúa el id en los emparejados, abre id nuevo en los no
emparejados, y mata los tracks que llevan `max_age` frames sin verse.

No hay modelo de movimiento (posición constante): suficiente a 10 Hz de VoD y
deja limpio el efecto de agregar apariencia después.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from tracker.detection.metrics_3d import compute_rotated_iou_2d


class SimpleIoUTracker:
    """Asociación greedy por IoU-BEV. `update` devuelve un id por detección."""

    def __init__(self, iou_threshold=0.1, max_age=3, min_hits=1):
        self.iou_threshold = iou_threshold
        self.max_age = max_age            # frames sin verse antes de morir
        self.min_hits = min_hits          # (reservado) hits mínimos para reportar
        self.tracks = []                  # [{id, box, age, hits}]
        self._next_id = 1

    def _iou_matrix(self, det_boxes, trk_boxes):
        M, N = len(det_boxes), len(trk_boxes)
        iou = np.zeros((M, N), np.float32)
        for i in range(M):
            for j in range(N):
                iou[i, j] = compute_rotated_iou_2d(det_boxes[i], trk_boxes[j])
        return iou

    def update(self, det_boxes, embeddings=None, num_points=None):
        """Procesa un frame. det_boxes: [M,7]. `embeddings`/`num_points` se
        ignoran (este baseline es solo geométrico). Devuelve (boxes, ids)."""
        det_boxes = np.asarray(det_boxes, np.float32).reshape(-1, 7)
        M = len(det_boxes)
        ids = np.zeros(M, int)

        if len(self.tracks) == 0:
            for i in range(M):
                ids[i] = self._nuevo_track(det_boxes[i])
            self._envejecer(matched_trk=set())
            return det_boxes, ids

        trk_boxes = np.stack([t["box"] for t in self.tracks])
        iou = self._iou_matrix(det_boxes, trk_boxes)

        matched_det, matched_trk = set(), set()
        if M > 0 and len(trk_boxes) > 0:
            di, tj = linear_sum_assignment(-iou)
            for i, j in zip(di, tj):
                if iou[i, j] >= self.iou_threshold:
                    t = self.tracks[j]
                    t["box"] = det_boxes[i]
                    t["age"] = 0
                    t["hits"] += 1
                    ids[i] = t["id"]
                    matched_det.add(i)
                    matched_trk.add(j)

        for i in range(M):
            if i not in matched_det:
                ids[i] = self._nuevo_track(det_boxes[i])

        self._envejecer(matched_trk)
        return det_boxes, ids

    def _nuevo_track(self, box):
        tid = self._next_id
        self._next_id += 1
        self.tracks.append({"id": tid, "box": box.copy(), "age": 0, "hits": 1})
        return tid

    def _envejecer(self, matched_trk):
        vivos = []
        for j, t in enumerate(self.tracks):
            if j not in matched_trk:
                t["age"] += 1
            if t["age"] <= self.max_age:
                vivos.append(t)
        self.tracks = vivos
