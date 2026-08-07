"""Carga las cajas GT de tracking (objetos móviles) en el frame del radar.

Las anotaciones de tracking de VoD viven en `lidar/training/label_2_tracking`,
un archivo por frame. Cada línea es una caja KITTI (frame cámara) con un
`obj_id` PERSISTENTE entre frames (ese es el track_id que evalúa el tracking):

    type obj_id truncated alpha  x1 y1 x2 y2  h w l  x y z  ry  moving

Solo contiene objetos MÓVILES (la última columna, moving, es siempre 1), que es
justo lo que sigue el resto del pipeline.

La conversión cámara -> radar usa exactamente la misma convención que
`gt_labels.get_bbx_param` (centro por `t_radar_camera`, rotación por
`t_radar_lidar`), para que las cajas GT queden en el MISMO sistema que las cajas
que predice el detector.
"""
import os
import sys
import types

import numpy as np
from scipy.spatial.transform import Rotation as R

# El paquete vod arrastra un subpaquete de visualización que depende de k3d
# (no instalado y que no necesitamos). Se stubea antes de importar vod.
for _mod in ["k3d", "external.vod.visualization",
             "external.vod.visualization.vis_2d",
             "external.vod.visualization.vis_3d",
             "external.vod.visualization.helpers"]:
    sys.modules.setdefault(_mod, types.ModuleType(_mod))

from external.vod.configuration import VodTrackLocations
from external.vod.frame import FrameDataLoader, FrameTransformMatrix


def parse_tracking_label_line(line):
    """Parsea una línea de label_2_tracking.

    Returns dict con type, obj_id, h, w, l, x, y, z, ry  (o None si inválida).
    """
    p = line.strip().split()
    if len(p) < 15:
        return None
    return {
        "type": p[0],
        "obj_id": int(p[1]),
        "h": float(p[8]), "w": float(p[9]), "l": float(p[10]),
        "x": float(p[11]), "y": float(p[12]), "z": float(p[13]),
        "ry": float(p[14]),
    }


def label_to_radar_box(lab, transforms):
    """Caja KITTI (cámara) -> [x, y, z, l, w, h, yaw] en frame radar.

    Misma transformación que `gt_labels.get_bbx_param(sensor='radar')`.
    """
    c = np.array([lab["x"], lab["y"], lab["z"], 1.0], np.float64)
    cr = (transforms.t_radar_camera @ c)[:3]
    rot = R.from_euler("XYZ", [0.0, 0.0, -(lab["ry"] + np.pi / 2.0)]).as_matrix()
    rot_r = transforms.t_radar_lidar[:3, :3] @ rot
    yaw = float(np.arctan2(rot_r[1, 0], rot_r[0, 0]))
    return np.array([cr[0], cr[1], cr[2], lab["l"], lab["w"], lab["h"], yaw],
                    np.float32)


class GtTrackLoader:
    """Carga cajas + track_ids GT por frame, en frame radar."""

    MIN_DIM = 0.1        # descarta cajas degeneradas (DontCare = 0.01³)
    # Clases que se excluyen del GT de tracking:
    #   DontCare      = región "ignorar" de KITTI (caja placeholder 0.01³)
    #   bicycle_rack  = estante de bicis ESTÁTICO (moving_flag=1 es artefacto)
    #   ride_uncertain= etiqueta incierta
    #   Pedestrian    = peatón: 1-2 puntos de radar, casi indetectable; se excluye
    #                   del eval de tracking radar (como la referencia/RaTrack).
    EXCLUDE_TYPES = {"DontCare", "bicycle_rack", "ride_uncertain", "Pedestrian"}
    IMG_W, IMG_H = 1936, 1216       # tamaño de la imagen VoD (para el filtro FOV)

    def __init__(self, dataset_path, keep_types=None, min_radar_points=0,
                 fov_only=False):
        """`keep_types`: conjunto de tipos a conservar (None = todos los móviles).
        `min_radar_points`: si >0, solo conserva cajas GT con ≥ ese nº de puntos
        de RADAR dentro (protocolo de RaTrack: objetos móviles con ≥5 puntos).
        `fov_only`: si True, solo conserva cajas cuyo CENTRO cae dentro del FOV de
        la cámara (adelante y dentro de la imagen). El pipeline es forward-facing
        (RGB+radar), así que los objetos fuera del FOV son inevaluables."""
        self.dataset_path = dataset_path
        self.keep_types = set(keep_types) if keep_types else None
        self.min_radar_points = int(min_radar_points)
        self.fov_only = bool(fov_only)
        self.loc = VodTrackLocations(root_dir=dataset_path, output_dir=dataset_path,
                                     frame_set_path="", pred_dir="")
        self._tf_cache = {}

    def frame_transforms(self, frame_id):
        """FrameTransformMatrix del frame (cacheado)."""
        fs = str(int(frame_id)).zfill(5)
        if fs not in self._tf_cache:
            try:
                fd = FrameDataLoader(kitti_locations=self.loc, frame_number=fs)
                self._tf_cache[fs] = FrameTransformMatrix(fd)
            except Exception:
                self._tf_cache[fs] = None
        return self._tf_cache[fs]

    def fov_mask(self, boxes, frame_id):
        """Máscara de cajas (frame radar) cuyo centro cae dentro del FOV de la
        cámara en ese frame. Para filtrar detecciones igual que el GT."""
        boxes = np.asarray(boxes, np.float32).reshape(-1, 7)
        tf = self.frame_transforms(frame_id)
        if tf is None or len(boxes) == 0:
            return np.ones(len(boxes), bool)
        return centros_en_fov(boxes, tf, self.IMG_W, self.IMG_H)

    def load_frame(self, frame_id):
        """Devuelve (boxes [N,7], ids [N] int, types [N] str) para el frame dado."""
        fs = str(int(frame_id)).zfill(5)
        label_file = os.path.join(self.loc.tracking_label_dir, f"{fs}.txt")
        empty = (np.zeros((0, 7), np.float32), np.zeros(0, int), [])
        if not os.path.exists(label_file):
            return empty
        try:
            fd = FrameDataLoader(kitti_locations=self.loc, frame_number=fs)
            transforms = FrameTransformMatrix(fd)
            self._tf_cache[fs] = transforms
            with open(label_file) as f:
                lines = f.readlines()
        except Exception:
            return empty

        boxes, ids, types = [], [], []
        for line in lines:
            lab = parse_tracking_label_line(line)
            if lab is None:
                continue
            # Descarta clases no-móviles/inválidas (DontCare, bicycle_rack,
            # ride_uncertain) y cajas DEGENERADAS: KITTI mete placeholders de
            # 0.01³ con id dummy que NO son objetos reales; contarlos infla FN y
            # ensucia las métricas (id sin caja visible).
            if lab["type"] in self.EXCLUDE_TYPES:
                continue
            if min(lab["h"], lab["w"], lab["l"]) < self.MIN_DIM:
                continue
            if self.keep_types is not None and lab["type"] not in self.keep_types:
                continue
            try:
                box = label_to_radar_box(lab, transforms)
            except Exception:
                continue
            boxes.append(box)
            ids.append(lab["obj_id"])
            types.append(lab["type"])

        if not boxes:
            return empty
        boxes = np.stack(boxes).astype(np.float32)
        ids = np.array(ids, int)

        # Filtro de FOV: solo cajas cuyo centro se proyecta dentro de la imagen
        # (adelante de la cámara). Los objetos fuera del FOV no los ve el pipeline
        # forward-facing (RGB+radar) -> no deben contar en la métrica ni en el plot.
        if self.fov_only:
            keep = centros_en_fov(boxes, transforms, self.IMG_W, self.IMG_H)
            boxes, ids = boxes[keep], ids[keep]
            types = [t for t, k in zip(types, keep) if k]
            if len(boxes) == 0:
                return empty

        # Filtro de observabilidad (protocolo RaTrack): solo cajas con ≥ N puntos
        # de RADAR dentro. Reduce num_gt a los objetos que el radar realmente ve.
        if self.min_radar_points > 0:
            try:
                radar = fd.radar_data[:, :3]           # frame radar
            except Exception:
                radar = None
            if radar is not None and len(radar):
                keep = np.array([
                    _puntos_en_caja(radar, b).sum() >= self.min_radar_points
                    for b in boxes])
                boxes, ids = boxes[keep], ids[keep]
                types = [t for t, k in zip(types, keep) if k]
            else:
                return empty
        if len(boxes) == 0:
            return empty
        return boxes, ids, types


def _puntos_en_caja(pts_xyz, box, margen=0.0):
    """Máscara BEV de puntos dentro de la caja rotada. box=[x,y,z,l,w,h,yaw]."""
    rel = pts_xyz[:, :2] - box[:2]
    c, s = np.cos(-box[6]), np.sin(-box[6])
    rx = rel[:, 0] * c - rel[:, 1] * s
    ry = rel[:, 0] * s + rel[:, 1] * c
    return (np.abs(rx) <= box[3] / 2 + margen) & (np.abs(ry) <= box[4] / 2 + margen)


def centros_en_fov(boxes, transforms, W, H):
    """Máscara: el centro de cada caja (frame radar) se proyecta dentro de la
    imagen y adelante de la cámara (depth>0). boxes: [N,7]."""
    n = len(boxes)
    if n == 0:
        return np.zeros(0, bool)
    homo = np.hstack([boxes[:, :3], np.ones((n, 1), np.float32)])
    cam = (transforms.t_camera_radar @ homo.T).T          # [N,4] frame cámara
    depth = cam[:, 2]
    P = transforms.camera_projection_matrix
    uvw = P @ cam.T
    uvw = uvw / np.where(np.abs(uvw[2]) < 1e-6, 1e-6, uvw[2])
    u, v = uvw[0], uvw[1]
    return (depth > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)


def read_clip_frames(clip_path):
    """Lee un archivo de clip (un número de frame por línea) -> lista de ints."""
    with open(clip_path) as f:
        return [int(x) for x in f.read().split()]
