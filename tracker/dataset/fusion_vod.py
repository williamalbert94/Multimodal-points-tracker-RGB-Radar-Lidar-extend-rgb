"""Early LiDAR–radar fusion for View-of-Delft.

Both sensors are already in the radar frame by the time this runs (the loader
applies the extrinsic `T_radar<-lidar`, and ego-motion compensation for the
temporal pair), so fusion here is a pure per-point association problem.

Why several modes
-----------------
The natural formulation — take LiDAR as the geometric base and copy the nearest
radar point's attributes $[v, \\rho]$ within a radius — is degenerate at this
sensor ratio. A VoD frame carries ~178,000 LiDAR points against ~300 radar
returns, so measured over six frames:

| radius | LiDAR points receiving radar attributes |
|--------|----------------------------------------|
| 0.5 m  | 2.9 – 8.9 %                            |
| 1.0 m  | 8.3 – 15.9 %                           |
| 2.0 m  | 16.9 – 26.0 %                          |

With `num_points = 256` sampled from 178k, roughly 95 % of the points the
network sees would carry `[0, 0]` — the radar channel becomes almost pure
padding and the model degenerates to LiDAR-only geometry. `lidar_base` is
provided because it is the formulation in the write-up, but `matched` and
`radar_base` below keep both modalities present on **every** point.

Modes
-----
``none``        Radar points and radar features, unchanged. Current behaviour.
``lidar_base``  LiDAR geometry, radar attributes by nearest neighbour within
                `radius`, zeros where there is no match. Faithful to the
                write-up; see the coverage caveat above.
``matched``     Only LiDAR points that *have* a radar match. Dense LiDAR
                geometry with 100 % radar coverage, at the cost of discarding
                LiDAR far from any return.
``radar_base``  Radar points (all keep their true `[v, ρ]`), each augmented with
                local LiDAR geometry: neighbour count, mean height and height
                spread inside `radius`. 100 % coverage of both modalities, and
                the point count stays close to what the network is tuned for.

Feature widths
--------------
Todos los modos arrastran los **3** canales de radar `(RCS, v_r, v_r_comp)`.

Ojo con `v_r_comp` (la velocidad radial COMPENSADA por el ego-movimiento): es el
canal que mejor separa móvil de estático. Medido sobre 48 frames de VoD:

    |v_r|      cruda        móvil 1.10   estático 3.74   <- confunde: el estático
                                                            "parece" más rápido
    |v_r_comp| compensada   móvil 1.40   estático 0.008  <- separación casi perfecta

Una versión anterior de este módulo recortaba con `radar_feat[:, :2]` y botaba
justo `v_r_comp`, así que activar la fusión echaba a perder la mejor señal. Por
eso ahora se conservan los tres canales.

Anchos por modo: ``none`` / ``lidar_base`` / ``matched`` -> **3**;
``radar_base`` -> **6** (los 3 de radar + 3 estadísticas del LiDAR). El extractor
se debe construir con `in_channels` igual a ese número.
"""
import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:                                     # pragma: no cover
    cKDTree = None

MODES = ("none", "lidar_base", "matched", "radar_base")

# Canales de features que entrega cada modo.
# Todos conservan los 3 de radar [RCS, v_r, v_r_comp]; `radar_base` les suma 3
# estadísticas del LiDAR (cantidad de vecinos, altura media y rango de altura).
MODE_FEATURE_DIM = {"none": 3, "lidar_base": 3, "matched": 3, "radar_base": 6}

# Cuántas columnas del radar se arrastran: RCS, v_r y v_r_comp.
RADAR_FEATS = 3

DEFAULT_RADIUS = 0.5


def feature_dim(mode):
    if mode not in MODES:
        raise ValueError(f"unknown fusion mode {mode!r}; expected one of {MODES}")
    return MODE_FEATURE_DIM[mode]


def fuse(radar_xyz, radar_feat, lidar_xyz, mode="none",
         radius=DEFAULT_RADIUS, max_points=None, rng=None):
    """Fuse one frame.

    Args:
        radar_xyz:  (R, 3) radar points, radar frame.
        radar_feat: (R, F) atributos del radar; se usan las 3 primeras columnas
            `(RCS, v_r, v_r_comp)`.
        lidar_xyz:  (L, 3) LiDAR points, already transformed into the radar frame.
        mode:       one of `MODES`.
        radius:     association radius in metres.
        max_points: cap on returned points (random subsample, for the LiDAR-based
            modes which would otherwise return ~178k points).
        rng:        optional `np.random.Generator` for reproducible subsampling.

    Returns:
        (points (N,3) float32, features (N,F') float32) with F' = feature_dim(mode).
    """
    if mode not in MODES:
        raise ValueError(f"unknown fusion mode {mode!r}; expected one of {MODES}")

    radar_xyz = np.asarray(radar_xyz, dtype=np.float32).reshape(-1, 3)
    radar_feat = np.asarray(radar_feat, dtype=np.float32).reshape(len(radar_xyz), -1)
    # Nos quedamos con los 3 canales del radar (RCS, v_r, v_r_comp). Si vinieran
    # menos columnas, se rellena con ceros para no romper las formas.
    rf = radar_feat[:, :RADAR_FEATS] if radar_feat.shape[1] >= RADAR_FEATS else \
        np.pad(radar_feat, ((0, 0), (0, RADAR_FEATS - radar_feat.shape[1])))

    if mode == "none":
        return radar_xyz, rf

    if cKDTree is None:
        raise ImportError("scipy is required for fusion modes other than 'none'")

    lidar_xyz = np.asarray(lidar_xyz, dtype=np.float32).reshape(-1, 3)
    if len(lidar_xyz) == 0 or len(radar_xyz) == 0:
        # No hay con qué asociar: devolvemos el radar para no frenar el entrenamiento.
        return (radar_xyz, rf) if feature_dim(mode) == RADAR_FEATS else \
            (radar_xyz, np.hstack([rf, np.zeros((len(rf), 3), np.float32)]))

    rng = rng or np.random.default_rng()

    if mode in ("lidar_base", "matched"):
        # Nearest radar return for every LiDAR point.
        dist, idx = cKDTree(radar_xyz).query(lidar_xyz, distance_upper_bound=radius)
        hit = np.isfinite(dist)

        if mode == "matched":
            lidar_xyz = lidar_xyz[hit]
            idx = idx[hit]
            feats = rf[idx]
        else:                                    # lidar_base: ceros donde no hubo match
            feats = np.zeros((len(lidar_xyz), RADAR_FEATS), np.float32)
            feats[hit] = rf[idx[hit]]

        pts = lidar_xyz
        if max_points and len(pts) > max_points:
            sel = rng.choice(len(pts), max_points, replace=False)
            pts, feats = pts[sel], feats[sel]
        return pts.astype(np.float32), feats.astype(np.float32)

    # radar_base: keep radar points, describe the LiDAR around each one.
    tree = cKDTree(lidar_xyz)
    neigh = tree.query_ball_point(radar_xyz, r=radius)
    extra = np.zeros((len(radar_xyz), 3), np.float32)
    for i, ids in enumerate(neigh):
        if not ids:
            continue                              # count 0, height stats stay 0
        z = lidar_xyz[ids, 2]
        extra[i] = (len(ids), z.mean(), z.max() - z.min())
    # Compress the unbounded count so it does not dominate the other channels.
    extra[:, 0] = np.log1p(extra[:, 0])
    return radar_xyz, np.hstack([rf, extra]).astype(np.float32)
