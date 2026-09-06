"""
MOT Metrics Implementation
Compute standard tracking metrics: MOTA, IDF1, AMOTA, sAMOTA, etc.

Reference:
- MOTA/MOTP: Bernardin & Stiefelhagen (2008) - CLEAR MOT
- IDF1: Ristani et al. (2016) - proper global ID assignment
- AMOTA/sAMOTA: Weng & Kitani (2020) - AB3DMOT
  sAMOTA = (1/n) * sum_t [ max(0, MOTA_t) * recall_t ]
  where recall_t = TP_t / max(GT_t, 1)
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


class MOTMetricsAccumulator:
    """
    Accumulate tracking results for MOT metrics computation.

    Usage:
        accumulator = MOTMetricsAccumulator()

        for frame in sequence:
            pred_boxes, pred_ids = tracker.update(detections)
            accumulator.update(frame_id, gt_boxes, gt_ids, pred_boxes, pred_ids)

        metrics = accumulator.compute_metrics()
    """

    def __init__(self):
        self.frame_data = []
        self.gt_tracks = {}   # {track_id: [frame_ids]}
        self.pred_tracks = {}  # {track_id: [frame_ids]}
        # Cache de la matriz de IoU pred×gt por frame. La matriz NO depende del
        # umbral, así que se computa una vez y se reusa en los ~15 umbrales del
        # loop AMOTA (el IoU rotado en Python puro es el costo dominante).
        self._iou_cache = {}

    def update(self, frame_id, gt_boxes, gt_ids, pred_boxes, pred_ids, iou_threshold=0.5):
        """
        Update accumulator with results from one frame.

        Args:
            frame_id: int - frame number
            gt_boxes: [N, 7] tensor/array - GT boxes (x, y, z, l, w, h, yaw)
            gt_ids: [N] tensor/array - GT track IDs
            pred_boxes: [M, 7] tensor/array - predicted boxes
            pred_ids: [M] tensor/array - predicted track IDs
            iou_threshold: float - IoU threshold for matching
        """
        if isinstance(gt_boxes, torch.Tensor):
            gt_boxes = gt_boxes.cpu().numpy()
        if isinstance(gt_ids, torch.Tensor):
            gt_ids = gt_ids.cpu().numpy()
        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes = pred_boxes.cpu().numpy()
        if isinstance(pred_ids, torch.Tensor):
            pred_ids = pred_ids.cpu().numpy()

        self.frame_data.append({
            'frame_id': frame_id,
            'gt_boxes': gt_boxes,
            'gt_ids': gt_ids,
            'pred_boxes': pred_boxes,
            'pred_ids': pred_ids,
        })

        if gt_ids is not None:
            for track_id in gt_ids:
                self.gt_tracks.setdefault(track_id, []).append(frame_id)

        if pred_ids is not None:
            for track_id in pred_ids:
                self.pred_tracks.setdefault(track_id, []).append(frame_id)

    def compute_metrics(self, iou_thresholds=None):
        """
        Compute all MOT metrics.

        AMOTA/sAMOTA: averaged over IoU thresholds per AB3DMOT standard.
        sAMOTA includes recall weighting per Weng & Kitani (2020).

        Returns:
            metrics: dict with all MOT metrics
        """
        if iou_thresholds is None:
            # AB3DMOT / nuScenes standard: 0.25 to 0.95 step 0.05
            # For 3D radar tracking with imprecise boxes, start at 0.25
            iou_thresholds = np.arange(0.25, 1.0, 0.05)

        # Compute per-frame matches ONCE at common thresholds (cache)
        frame_matches_cache = {}
        for iou_thresh in [0.25, 0.5]:
            frame_matches_cache[iou_thresh] = self._compute_all_frame_matches(iou_thresh)

        # Primary metrics at IoU=0.5
        metrics_0_5 = self._compute_single_threshold_metrics(
            iou_threshold=0.5,
            frame_matches=frame_matches_cache.get(0.5)
        )

        # AMOTA and sAMOTA: average over IoU thresholds.
        motas = []
        motps = []
        tps = []
        total_gts = []
        errores = []          # FN+FP+IDSW por umbral (para el sMOTA escalado)
        for iou_thresh in iou_thresholds:
            fm = frame_matches_cache.get(iou_thresh)
            if fm is None:
                fm = self._compute_all_frame_matches(iou_thresh)
            m = self._compute_single_threshold_metrics(iou_threshold=iou_thresh, frame_matches=fm)
            motas.append(max(0.0, m['MOTA']))  # clamp negative MOTA to 0
            motps.append(m['MOTP'])
            tps.append(m['TP'])
            total_gts.append(m['total_GT'])
            errores.append(m['FN'] + m['FP'] + m['ID_switches'])

        amota = float(np.mean(motas))
        amotp = float(np.mean(motps))

        # sAMOTA: scaled MOTA de AB3DMOT (Weng & Kitani 2020). La escala por recall
        # RESTA el término (1-recall)*num_gt, que cancela el castigo por los objetos
        # legítimamente no detectados -> mide la limpieza (FP+IDSW) de lo trackeado:
        #   sMOTA_t = max(0, 1 - (FN+FP+IDSW - (1-recall_t)*ngt) / (recall_t*ngt))
        # (NOTA: la versión oficial barre la CONFIANZA de las detecciones; la
        #  implementación completa con barrido de score está en `amota_ab3dmot.py`.)
        samota_components = []
        for err, tp_val, gt_val in zip(errores, tps, total_gts):
            if gt_val > 0:
                recall = min(tp_val, gt_val) / gt_val
                if recall > 0:
                    smota = max(0.0, 1.0 - (err - (1.0 - recall) * gt_val) / (recall * gt_val))
                else:
                    smota = 0.0
                samota_components.append(smota)
            else:
                samota_components.append(0.0)
        samota = float(np.mean(samota_components)) * 100.0 if samota_components else 0.0

        # IDF1: proper global ID-based computation
        idf1 = self._compute_idf1(iou_threshold=0.5)

        # MT/PT/ML using cached matches
        mt, pt, ml = self._compute_track_quality(
            iou_threshold=0.5,
            frame_matches=frame_matches_cache.get(0.5)
        )

        # Fragmentations: count tracking interruptions per GT track
        fragmentations = self._compute_fragmentations(
            iou_threshold=0.5,
            frame_matches=frame_matches_cache.get(0.5)
        )

        all_metrics = {
            # Primary metrics (IoU=0.5)
            'MOTA': metrics_0_5['MOTA'],
            'IDF1': idf1,
            'MOTP': metrics_0_5['MOTP'],

            # Average metrics over IoU thresholds
            'sAMOTA': samota,
            'AMOTA': amota,
            'AMOTP': amotp,

            # Detection-only
            'MODA': metrics_0_5['MODA'],

            # Track quality
            'MT': mt,
            'PT': pt,
            'ML': ml,

            # Error breakdown
            'ID_switches': metrics_0_5['ID_switches'],
            'Fragmentations': fragmentations,
            'FP': metrics_0_5['FP'],
            'FN': metrics_0_5['FN'],
            'TP': metrics_0_5['TP'],
        }

        return all_metrics

    # =====================================================================
    # Core per-frame matching (computed once, reused everywhere)
    # =====================================================================

    def _compute_all_frame_matches(self, iou_threshold):
        """
        Compute matches for ALL frames at a given IoU threshold.
        Returns a list of match results parallel to self.frame_data.
        """
        results = []
        for fi, frame_data in enumerate(self.frame_data):
            gt_boxes = frame_data.get('gt_boxes', np.zeros((0, 7)))
            gt_ids = frame_data.get('gt_ids', np.array([]))
            pred_boxes = frame_data.get('pred_boxes', np.zeros((0, 7)))
            pred_ids = frame_data.get('pred_ids', np.array([]))

            if gt_ids is None or pred_ids is None:
                results.append(None)
                continue

            # Matriz de IoU cacheada por frame (independiente del umbral).
            if fi not in self._iou_cache:
                if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                    self._iou_cache[fi] = None
                else:
                    self._iou_cache[fi] = self._compute_rotated_iou_matrix(
                        pred_boxes, gt_boxes)

            matches, unmatched_gt, unmatched_pred = self._match_boxes(
                pred_boxes, pred_ids, gt_boxes, gt_ids, iou_threshold,
                iou_matrix=self._iou_cache[fi]
            )
            results.append({
                'matches': matches,
                'unmatched_gt': unmatched_gt,
                'unmatched_pred': unmatched_pred,
            })
        return results

    # =====================================================================
    # MOTA / MOTP / MODA (Bernardin & Stiefelhagen 2008)
    # =====================================================================

    def _compute_single_threshold_metrics(self, iou_threshold=0.5, frame_matches=None):
        """
        Compute MOTA, MOTP, MODA, ID switches for a single IoU threshold.

        MOTA = 1 - (FN + FP + IDSW) / total_GT
        MOTP = sum(IoU of matched pairs) / TP
        """
        if frame_matches is None:
            frame_matches = self._compute_all_frame_matches(iou_threshold)

        total_gt = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_id_switches = 0
        sum_iou = 0.0

        # Track last matched pred ID per GT track (for ID switch detection)
        gt_track_last_match = {}

        for i, frame_data in enumerate(self.frame_data):
            gt_ids = frame_data.get('gt_ids', None)
            pred_ids = frame_data.get('pred_ids', None)

            if gt_ids is None or pred_ids is None or frame_matches[i] is None:
                continue

            total_gt += len(gt_ids)

            fm = frame_matches[i]
            matches = fm['matches']

            for pred_idx, gt_idx, iou in matches:
                total_tp += 1
                sum_iou += iou

                gt_id = gt_ids[gt_idx]
                pred_id = pred_ids[pred_idx]

                if gt_id in gt_track_last_match:
                    if gt_track_last_match[gt_id] != pred_id:
                        total_id_switches += 1

                gt_track_last_match[gt_id] = pred_id

            total_fp += len(fm['unmatched_pred'])
            total_fn += len(fm['unmatched_gt'])

        mota = (1.0 - (total_fn + total_fp + total_id_switches) / max(total_gt, 1)) * 100.0
        motp = (sum_iou / max(total_tp, 1)) * 100.0
        moda = (1.0 - (total_fn + total_fp) / max(total_gt, 1)) * 100.0

        return {
            'MOTA': mota,
            'MOTP': motp,
            'MODA': moda,
            'ID_switches': total_id_switches,
            'FP': total_fp,
            'FN': total_fn,
            'TP': total_tp,
            'total_GT': total_gt,
        }

    # =====================================================================
    # IDF1 (Ristani et al. 2016) - Global optimal ID assignment
    # =====================================================================

    def _compute_idf1(self, iou_threshold=0.5):
        """
        Compute IDF1 using global optimal assignment between pred and GT tracks.

        IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)

        where IDTP/IDFP/IDFN are computed after finding the optimal
        one-to-one mapping between predicted and GT track identities.
        """
        gt_tracks, pred_tracks = self._build_track_trajectories()

        if len(gt_tracks) == 0 or len(pred_tracks) == 0:
            return 0.0

        gt_ids_list = list(gt_tracks.keys())
        pred_ids_list = list(pred_tracks.keys())

        # Build cost matrix: number of matching frames (with IoU > threshold)
        # between each (pred_track, gt_track) pair
        n_pred = len(pred_ids_list)
        n_gt = len(gt_ids_list)
        match_count_matrix = np.zeros((n_pred, n_gt))

        for i, pred_id in enumerate(pred_ids_list):
            pred_frame_set = set(pred_tracks[pred_id]['frames'])
            pred_frame_to_box = dict(zip(
                pred_tracks[pred_id]['frames'],
                pred_tracks[pred_id]['boxes']
            ))

            for j, gt_id in enumerate(gt_ids_list):
                gt_frame_set = set(gt_tracks[gt_id]['frames'])
                gt_frame_to_box = dict(zip(
                    gt_tracks[gt_id]['frames'],
                    gt_tracks[gt_id]['boxes']
                ))

                overlap_frames = pred_frame_set & gt_frame_set
                count = 0
                for f in overlap_frames:
                    iou = self._compute_rotated_iou_2d(
                        pred_frame_to_box[f], gt_frame_to_box[f]
                    )
                    if iou >= iou_threshold:
                        count += 1
                match_count_matrix[i, j] = count

        # Hungarian assignment (maximize matches)
        pred_indices, gt_indices = linear_sum_assignment(-match_count_matrix)

        # Compute IDTP, IDFP, IDFN
        total_idtp = 0

        for pi, gi in zip(pred_indices, gt_indices):
            idtp = int(match_count_matrix[pi, gi])
            if idtp > 0:
                total_idtp += idtp

        # Total frames across all pred and gt tracks
        total_pred_frames = sum(len(t['frames']) for t in pred_tracks.values())
        total_gt_frames = sum(len(t['frames']) for t in gt_tracks.values())

        idfp = total_pred_frames - total_idtp
        idfn = total_gt_frames - total_idtp

        idf1 = (2.0 * total_idtp) / max(2.0 * total_idtp + idfp + idfn, 1.0) * 100.0

        return idf1

    # =====================================================================
    # Track Quality: MT / PT / ML
    # =====================================================================

    def _compute_track_quality(self, iou_threshold=0.5, frame_matches=None):
        """
        Compute Mostly Tracked (MT), Partially Tracked (PT), Mostly Lost (ML).

        MT: GT track matched in > 80% of its frames
        ML: GT track matched in < 20% of its frames
        PT: everything else
        """
        if frame_matches is None:
            frame_matches = self._compute_all_frame_matches(iou_threshold)

        # Build per-frame: which GT indices were matched?
        # frame_idx -> set of matched gt_ids
        gt_matched_per_frame = {}
        for i, frame_data in enumerate(self.frame_data):
            gt_ids = frame_data.get('gt_ids', None)
            if gt_ids is None or frame_matches[i] is None:
                continue
            matched_gt_ids = set()
            for _, gt_idx, _ in frame_matches[i]['matches']:
                matched_gt_ids.add(gt_ids[gt_idx])
            gt_matched_per_frame[frame_data['frame_id']] = matched_gt_ids

        num_gt_tracks = len(self.gt_tracks)
        if num_gt_tracks == 0:
            return 0.0, 0.0, 0.0

        mt_count = 0
        pt_count = 0
        ml_count = 0

        for gt_id, frame_list in self.gt_tracks.items():
            total_frames = len(frame_list)
            matched_frames = sum(
                1 for f in frame_list
                if gt_id in gt_matched_per_frame.get(f, set())
            )
            ratio = matched_frames / max(total_frames, 1)

            if ratio > 0.8:
                mt_count += 1
            elif ratio < 0.2:
                ml_count += 1
            else:
                pt_count += 1

        mt = mt_count / num_gt_tracks * 100.0
        pt = pt_count / num_gt_tracks * 100.0
        ml = ml_count / num_gt_tracks * 100.0

        return mt, pt, ml

    # =====================================================================
    # Fragmentations (tracking interruptions)
    # =====================================================================

    def _compute_fragmentations(self, iou_threshold=0.5, frame_matches=None):
        """
        Count fragmentations: number of times a GT track switches from
        'tracked' to 'untracked' (i.e., tracking interruptions).
        """
        if frame_matches is None:
            frame_matches = self._compute_all_frame_matches(iou_threshold)

        # Build per-frame matched GT IDs
        frame_id_to_matched_gt = {}
        for i, frame_data in enumerate(self.frame_data):
            gt_ids = frame_data.get('gt_ids', None)
            if gt_ids is None or frame_matches[i] is None:
                continue
            matched_gt_ids = set()
            for _, gt_idx, _ in frame_matches[i]['matches']:
                matched_gt_ids.add(gt_ids[gt_idx])
            frame_id_to_matched_gt[frame_data['frame_id']] = matched_gt_ids

        total_fragmentations = 0

        for gt_id, frame_list in self.gt_tracks.items():
            sorted_frames = sorted(frame_list)
            was_tracked = False
            for f in sorted_frames:
                is_tracked = gt_id in frame_id_to_matched_gt.get(f, set())
                if was_tracked and not is_tracked:
                    total_fragmentations += 1
                was_tracked = is_tracked

        return total_fragmentations

    # =====================================================================
    # Box matching (Hungarian algorithm)
    # =====================================================================

    def _match_boxes(self, pred_boxes, pred_ids, gt_boxes, gt_ids, iou_threshold,
                     iou_matrix=None):
        """
        Match predicted boxes to GT using Hungarian algorithm based on
        rotated 2D BEV IoU (considers yaw orientation).

        `iou_matrix`: matriz pred×gt precomputada (opcional); si es None se
        computa acá. Permite reusar la matriz entre umbrales.

        Returns:
            matches: list of (pred_idx, gt_idx, iou)
            unmatched_gt: list of gt indices
            unmatched_pred: list of pred indices
        """
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return [], list(range(len(gt_boxes))), list(range(len(pred_boxes)))

        if iou_matrix is None:
            iou_matrix = self._compute_rotated_iou_matrix(pred_boxes, gt_boxes)

        pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)

        matches = []
        matched_pred = set()
        matched_gt = set()

        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            iou = iou_matrix[pred_idx, gt_idx]
            if iou >= iou_threshold:
                matches.append((pred_idx, gt_idx, iou))
                matched_pred.add(pred_idx)
                matched_gt.add(gt_idx)

        unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]
        unmatched_pred = [i for i in range(len(pred_boxes)) if i not in matched_pred]

        return matches, unmatched_gt, unmatched_pred

    # =====================================================================
    # Rotated 2D BEV IoU (accounts for yaw orientation)
    # =====================================================================

    def _box_corners_2d(self, box):
        """
        Get 4 BEV corners of a rotated box.

        Args:
            box: [7] array (x, y, z, l, w, h, yaw)

        Returns:
            corners: [4, 2] array
        """
        x, y = box[0], box[1]
        l, w = box[3], box[4]
        yaw = box[6] if len(box) > 6 else 0.0

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        # Half extents
        hl, hw = l / 2.0, w / 2.0

        # Corner offsets in local frame
        dx = np.array([-hl, hl, hl, -hl])
        dy = np.array([-hw, -hw, hw, hw])

        # Rotate to world frame
        corners_x = x + dx * cos_yaw - dy * sin_yaw
        corners_y = y + dx * sin_yaw + dy * cos_yaw

        return np.stack([corners_x, corners_y], axis=1)  # [4, 2]

    def _polygon_area(self, vertices):
        """Shoelace formula for polygon area."""
        n = len(vertices)
        if n < 3:
            return 0.0
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        return abs(area) / 2.0

    def _line_segment_intersection(self, p1, p2, p3, p4):
        """
        Find intersection point of two line segments (p1-p2) and (p3-p4).
        Returns (x, y) or None if no intersection.
        """
        d1x = p2[0] - p1[0]
        d1y = p2[1] - p1[1]
        d2x = p4[0] - p3[0]
        d2y = p4[1] - p3[1]

        cross = d1x * d2y - d1y * d2x
        if abs(cross) < 1e-10:
            return None

        t = ((p3[0] - p1[0]) * d2y - (p3[1] - p1[1]) * d2x) / cross
        u = ((p3[0] - p1[0]) * d1y - (p3[1] - p1[1]) * d1x) / cross

        if 0 <= t <= 1 and 0 <= u <= 1:
            return (p1[0] + t * d1x, p1[1] + t * d1y)
        return None

    def _point_in_polygon(self, point, polygon):
        """Check if point is inside polygon using ray casting."""
        x, y = point
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def _polygon_intersection_area(self, poly1, poly2):
        """
        Compute intersection area of two convex polygons using
        Sutherland-Hodgman clipping.
        """
        # Collect all candidate points:
        # 1. Corners of poly1 inside poly2
        # 2. Corners of poly2 inside poly1
        # 3. Edge intersection points
        intersection_points = []

        n1 = len(poly1)
        n2 = len(poly2)

        for p in poly1:
            if self._point_in_polygon(p, poly2):
                intersection_points.append(tuple(p))

        for p in poly2:
            if self._point_in_polygon(p, poly1):
                intersection_points.append(tuple(p))

        for i in range(n1):
            for j in range(n2):
                pt = self._line_segment_intersection(
                    poly1[i], poly1[(i + 1) % n1],
                    poly2[j], poly2[(j + 1) % n2]
                )
                if pt is not None:
                    intersection_points.append(pt)

        if len(intersection_points) < 3:
            return 0.0

        # Remove near-duplicates
        unique_points = []
        for p in intersection_points:
            is_dup = False
            for q in unique_points:
                if abs(p[0] - q[0]) < 1e-8 and abs(p[1] - q[1]) < 1e-8:
                    is_dup = True
                    break
            if not is_dup:
                unique_points.append(p)

        if len(unique_points) < 3:
            return 0.0

        # Sort by angle around centroid for convex hull ordering
        cx = sum(p[0] for p in unique_points) / len(unique_points)
        cy = sum(p[1] for p in unique_points) / len(unique_points)
        unique_points.sort(key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))

        return self._polygon_area(unique_points)

    def _compute_rotated_iou_2d(self, box1, box2):
        """Compute rotated 2D BEV IoU between two boxes (considers yaw)."""
        corners1 = self._box_corners_2d(box1)
        corners2 = self._box_corners_2d(box2)

        inter_area = self._polygon_intersection_area(corners1, corners2)

        area1 = box1[3] * box1[4]  # l * w
        area2 = box2[3] * box2[4]
        union_area = area1 + area2 - inter_area

        return inter_area / max(union_area, 1e-6)

    def _compute_rotated_iou_matrix(self, boxes1, boxes2):
        """
        Compute rotated 2D BEV IoU matrix between two sets of boxes.
        Accounts for yaw orientation (unlike axis-aligned IoU).

        Pre-filtro: si la distancia entre centros supera la suma de los radios
        circunscritos, las cajas no pueden solaparse -> IoU 0 sin clipping. La
        mayoría de los pares pred×gt de un frame caen acá (evita el polígono).
        """
        b1 = np.asarray(boxes1, np.float64)
        b2 = np.asarray(boxes2, np.float64)
        M, N = len(b1), len(b2)
        iou_matrix = np.zeros((M, N))

        # radio circunscrito en BEV = 0.5 * sqrt(l^2 + w^2)
        r1 = 0.5 * np.hypot(b1[:, 3], b1[:, 4])
        r2 = 0.5 * np.hypot(b2[:, 3], b2[:, 4])
        c1 = b1[:, :2]
        c2 = b2[:, :2]
        # distancia entre centros [M, N]
        d = np.sqrt(((c1[:, None, :] - c2[None, :, :]) ** 2).sum(-1))
        rsum = r1[:, None] + r2[None, :]

        for i in range(M):
            for j in range(N):
                if d[i, j] >= rsum[i, j]:
                    continue  # no se solapan -> IoU 0
                iou_matrix[i, j] = self._compute_rotated_iou_2d(boxes1[i], boxes2[j])

        return iou_matrix

    # =====================================================================
    # Track trajectory helpers
    # =====================================================================

    def _build_track_trajectories(self):
        """Build trajectory dicts: {track_id: {'frames': [...], 'boxes': [...]}}"""
        gt_tracks = {}
        pred_tracks = {}

        for frame_data in self.frame_data:
            frame_id = frame_data['frame_id']

            gt_ids = frame_data.get('gt_ids', None)
            if gt_ids is not None:
                for i, gt_id in enumerate(gt_ids):
                    if gt_id not in gt_tracks:
                        gt_tracks[gt_id] = {'frames': [], 'boxes': []}
                    gt_tracks[gt_id]['frames'].append(frame_id)
                    gt_tracks[gt_id]['boxes'].append(frame_data['gt_boxes'][i])

            pred_ids = frame_data.get('pred_ids', None)
            if pred_ids is not None:
                for i, pred_id in enumerate(pred_ids):
                    if pred_id not in pred_tracks:
                        pred_tracks[pred_id] = {'frames': [], 'boxes': []}
                    pred_tracks[pred_id]['frames'].append(frame_id)
                    pred_tracks[pred_id]['boxes'].append(frame_data['pred_boxes'][i])

        return gt_tracks, pred_tracks


def compute_mot_metrics_simple(pred_tracks_all, gt_tracks_all):
    """
    Simple wrapper to compute MOT metrics from full sequence predictions.

    Args:
        pred_tracks_all: dict {frame_id: {'boxes': [M, 7], 'ids': [M]}}
        gt_tracks_all: dict {frame_id: {'boxes': [N, 7], 'ids': [N]}}

    Returns:
        metrics: dict with all MOT metrics
    """
    accumulator = MOTMetricsAccumulator()

    all_frame_ids = sorted(set(pred_tracks_all.keys()) | set(gt_tracks_all.keys()))

    for frame_id in all_frame_ids:
        pred_data = pred_tracks_all.get(frame_id, {'boxes': np.zeros((0, 7)), 'ids': np.zeros(0)})
        gt_data = gt_tracks_all.get(frame_id, {'boxes': np.zeros((0, 7)), 'ids': np.zeros(0)})

        accumulator.update(
            frame_id=frame_id,
            gt_boxes=gt_data['boxes'],
            gt_ids=gt_data['ids'],
            pred_boxes=pred_data['boxes'],
            pred_ids=pred_data['ids']
        )

    return accumulator.compute_metrics()
