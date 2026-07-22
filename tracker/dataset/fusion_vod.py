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
``none`` / ``lidar_base`` / ``matched`` emit **2** feature channels `(RCS, v_r)`,
so the network is unchanged. ``radar_base`` emits **5** — the model's extractor
must then be built with `in_channels=5`; the loader raises if that is
inconsistent rather than letting shapes fail deep inside the network.
"""
import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:                                     # pragma: no cover
    cKDTree = None

MODES = ("none", "lidar_base", "matched", "radar_base")

# Feature channels each mode makes available to the loader.
# `none` is the pass-through of VoD's raw radar columns [RCS, v_r, v_r_comp];
# the trainer slices the first two. The fusion modes emit exactly what they use.
MODE_FEATURE_DIM = {"none": 3, "lidar_base": 2, "matched": 2, "radar_base": 5}

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
        radar_feat: (R, F) radar attributes; the first two columns are used as
            `(RCS, v_r)`.
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
    rf = radar_feat[:, :2] if radar_feat.shape[1] >= 2 else \
        np.pad(radar_feat, ((0, 0), (0, 2 - radar_feat.shape[1])))

    if mode == "none":
        return radar_xyz, rf

    if cKDTree is None:
        raise ImportError("scipy is required for fusion modes other than 'none'")

    lidar_xyz = np.asarray(lidar_xyz, dtype=np.float32).reshape(-1, 3)
    if len(lidar_xyz) == 0 or len(radar_xyz) == 0:
        # Nothing to associate — fall back to radar so training does not stall.
        return (radar_xyz, rf) if feature_dim(mode) == 2 else \
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
        else:                                    # lidar_base: zeros where no match
            feats = np.zeros((len(lidar_xyz), 2), np.float32)
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
