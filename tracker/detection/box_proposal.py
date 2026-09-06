"""Propuesta de cajas 3D a partir de la segmentación móvil.

Etapa 1 del detector: los puntos que el modelo marca como MÓVIL se agrupan con
DBSCAN (en XY, porque el radar es esparso y vive sobre el plano) y a cada grupo
se le ajusta una caja orientada por PCA. Salida por caja: [x, y, z, l, w, h, yaw].

No hay aprendizaje acá: es el método "dbscan" simple, que es el punto de partida
razonable antes de un regresor de cajas aprendido.
"""
import numpy as np

try:
    from sklearn.cluster import DBSCAN
except ImportError:                                       # pragma: no cover
    DBSCAN = None

try:
    from scipy.spatial import cKDTree
except ImportError:                                       # pragma: no cover
    cKDTree = None


# Priors de tamaño (l, w, h) por clase, dimensiones crudas típicas de VoD/KITTI.
# El radar es demasiado esparso para medir la EXTENSIÓN del objeto (con 2-5 puntos
# la caja ajustada sale diminuta aunque el centro sea casi perfecto), así que para
# clusters con pocos puntos se usa un prior en vez del tamaño medido.
PRIOR_SIZES = {
    "car":        (3.9, 1.6, 1.5),
    "cyclist":    (1.8, 0.6, 1.7),
    "pedestrian": (0.8, 0.6, 1.7),
}


def _prior_por_span(span):
    """Elige un prior de clase según el largo medido del cluster (span en m)."""
    if span >= 2.5:
        return PRIOR_SIZES["car"]
    if span >= 1.0:
        return PRIOR_SIZES["cyclist"]
    return PRIOR_SIZES["pedestrian"]


def fit_oriented_box(cluster_xyz):
    """Ajusta una caja orientada (PCA en XY) a un grupo de puntos.

    Args:
        cluster_xyz: [P, 3] puntos del cluster (frame radar).

    Returns:
        [7] = (x, y, z, l, w, h, yaw). Yaw es el ángulo del eje principal en XY.
    """
    center = cluster_xyz.mean(axis=0)                     # [3]

    if len(cluster_xyz) == 1:
        return np.array([center[0], center[1], center[2], 0.5, 0.5, 0.3, 0.0],
                        dtype=np.float32)

    pts_xy = cluster_xyz[:, :2] - center[:2]

    if len(cluster_xyz) == 2 or np.allclose(pts_xy, 0):
        yaw = 0.0
        pts_rot = pts_xy
    else:
        cov = np.cov(pts_xy.T)
        if not np.all(np.isfinite(cov)):
            yaw, pts_rot = 0.0, pts_xy
        else:
            try:
                eigvals, eigvecs = np.linalg.eig(cov)
                main = eigvecs[:, int(np.argmax(eigvals))]
                yaw = float(np.arctan2(main[1], main[0]))
                c, s = np.cos(-yaw), np.sin(-yaw)
                rot = np.array([[c, -s], [s, c]])
                pts_rot = pts_xy @ rot.T
            except np.linalg.LinAlgError:
                yaw, pts_rot = 0.0, pts_xy

    size_xy = np.maximum(pts_rot.max(axis=0) - pts_rot.min(axis=0), 0.3)
    height = max(float(cluster_xyz[:, 2].max() - cluster_xyz[:, 2].min()), 0.3)

    return np.array([center[0], center[1], center[2],
                     size_xy[0], size_xy[1], height, yaw], dtype=np.float32)


def propose_boxes(points_xyz, moving_mask, prob=None,
                  eps=2.0, min_samples=1, min_points=2,
                  usar_priors=True, npts_confiar=12):
    """Propone cajas 3D a partir de los puntos marcados como móviles.

    Args:
        points_xyz: [N, 3] toda la nube de radar del frame.
        moving_mask: [N] bool, True = punto móvil (predicho).
        prob: [N] probabilidad de móvil por punto (opcional). Si se pasa, el
            score de cada caja es la probabilidad media de sus puntos; si no, se
            usa la cantidad de puntos normalizada. El score ordena las cajas en mAP.
        eps: radio de DBSCAN (m). min_samples: núcleo de DBSCAN.
        min_points: descarta clusters con menos de estos puntos.
        usar_priors: si True, para clusters con < `npts_confiar` puntos reemplaza
            el tamaño ajustado por un prior de clase (el radar no mide extensión
            con pocos puntos). Se conservan siempre el centro y el yaw del PCA.
        npts_confiar: a partir de estos puntos se confía en el tamaño medido.

    Returns:
        boxes: [M, 7] (x, y, z, l, w, h, yaw).
        scores: [M] confianza por caja.
    """
    points_xyz = np.asarray(points_xyz)[:, :3]
    moving_mask = np.asarray(moving_mask).astype(bool)
    idx_mov = np.where(moving_mask)[0]

    if len(idx_mov) < max(1, min_points) or DBSCAN is None:
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    pts_mov = points_xyz[idx_mov]
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts_mov[:, :2])

    boxes, scores = [], []
    for c in np.unique(labels):
        if c == -1:                                       # ruido de DBSCAN
            continue
        sel = labels == c
        npts = int(sel.sum())
        if npts < min_points:
            continue
        cluster_pts = pts_mov[sel]
        box = fit_oriented_box(cluster_pts)
        # Con pocos puntos el tamaño medido es poco fiable -> prior de clase
        # (se mantienen centro x,y,z y yaw, que sí son buenos aun con 3 puntos).
        if usar_priors and npts < npts_confiar:
            l, w, h = _prior_por_span(float(box[3]))
            box[3], box[4], box[5] = l, w, h
        boxes.append(box)
        if prob is not None:
            scores.append(float(np.asarray(prob)[idx_mov][sel].mean()))
        else:
            scores.append(float(npts))

    if len(boxes) == 0:
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return np.stack(boxes).astype(np.float32), np.asarray(scores, dtype=np.float32)


def _quitar_suelo(pts, base_z, alto_suelo=0.3, alto_max=4.0):
    """Filtra el suelo/fondo de un recorte LiDAR por altura relativa a la base."""
    z = pts[:, 2]
    return pts[(z > base_z + alto_suelo) & (z < base_z + alto_max)]


def propose_boxes_lidar(points_xyz, moving_mask, lidar_xyz, prob=None,
                        eps=2.0, min_samples=1, min_points=2,
                        radio_crop=2.5, min_lidar=5):
    """Fusión tardía: el RADAR detecta el objeto móvil, el LiDAR define su caja.

    El radar es esparso: localiza el CENTRO de los objetos muy bien pero no mide
    su extensión. El LiDAR es denso y sí la mide. Para cada cluster móvil de
    radar se recorta el LiDAR alrededor de su centro, se quita el suelo y se
    ajusta la caja (extensión + yaw) con esos puntos densos. Si no hay LiDAR
    suficiente, se cae al método radar-solo con priors.

    Args:
        points_xyz: [N, 3] nube de radar del frame.
        moving_mask: [N] bool, puntos móviles predichos.
        lidar_xyz: [L, 3] nube LiDAR del MISMO frame, en frame radar (alineada).
        prob: [N] prob por punto (para el score de la caja).
        eps/min_samples/min_points: DBSCAN sobre el radar móvil.
        radio_crop: radio XY (m) para juntar LiDAR alrededor del cluster.
        min_lidar: puntos LiDAR mínimos para confiar en el ajuste LiDAR.

    Returns:
        boxes: [M, 7], scores: [M].
    """
    points_xyz = np.asarray(points_xyz)[:, :3]
    lidar_xyz = np.asarray(lidar_xyz)[:, :3]
    moving_mask = np.asarray(moving_mask).astype(bool)
    idx_mov = np.where(moving_mask)[0]

    if len(idx_mov) < max(1, min_points) or DBSCAN is None:
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    pts_mov = points_xyz[idx_mov]
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts_mov[:, :2])

    arbol = cKDTree(lidar_xyz[:, :2]) if (cKDTree is not None and len(lidar_xyz)) else None

    boxes, scores = [], []
    for c in np.unique(labels):
        if c == -1:
            continue
        sel = labels == c
        npts = int(sel.sum())
        if npts < min_points:
            continue
        cluster_pts = pts_mov[sel]
        centro_xy = cluster_pts[:, :2].mean(axis=0)

        caja = None
        if arbol is not None:
            vecinos = arbol.query_ball_point(centro_xy, radio_crop)
            if len(vecinos) >= min_lidar:
                crop = lidar_xyz[vecinos]
                crop = _quitar_suelo(crop, base_z=float(crop[:, 2].min()))
                if len(crop) >= min_lidar:
                    # caja del LiDAR (extensión + yaw reales), centro anclado en
                    # el LiDAR del objeto (denso, sin sesgo del radar de un lado).
                    caja = fit_oriented_box(crop)

        if caja is None:
            # Sin LiDAR suficiente -> radar-solo con prior de tamaño.
            caja = fit_oriented_box(cluster_pts)
            l, w, h = _prior_por_span(float(caja[3]))
            caja[3], caja[4], caja[5] = l, w, h

        boxes.append(caja)
        scores.append(float(np.asarray(prob)[idx_mov][sel].mean()) if prob is not None else float(npts))

    if len(boxes) == 0:
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.stack(boxes).astype(np.float32), np.asarray(scores, dtype=np.float32)
