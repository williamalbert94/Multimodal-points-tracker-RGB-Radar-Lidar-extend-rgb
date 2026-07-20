# Introduction

## Radar/LiDAR Data Alignment

Radar and LiDAR are mounted at different positions/orientations on the vehicle and
are sampled while the ego-vehicle is moving, so raw point clouds from the two
sensors and from consecutive frames do not share a common reference frame. Two
corrections are applied before fusion (see `dataloader/track_vod_3d.py`):

**1. Extrinsic (spatial) alignment.** Points are converted to homogeneous
coordinates and mapped between sensors with the calibrated rigid transform
(`homogeneous_transformation`, `external/vod/frame/transformations.py`):

$$
\mathbf{p}_{radar} = T_{lidar \rightarrow radar} \cdot \mathbf{p}_{lidar}^{hom}
$$

**2. Temporal (ego-motion) alignment.** The relative pose between two frames is
obtained by chaining each frame's odometry $\rightarrow$ camera $\rightarrow$ radar
extrinsics and inverting the composition:

$$
T_{ego} = \big(T_{odom \rightarrow radar}(t)\big)^{-1} \cdot T_{odom \rightarrow radar}(t+1)
$$

The earlier point cloud is then re-expressed in the later frame:

$$
\mathbf{p}_{comp} = \mathbf{p}_{t} \cdot \big(T_{ego}\big)^{-T}
$$

### Importance of Alignment

Both corrections operate on the same ground (BEV) plane used throughout this
project, and both are necessary for the multimodal fusion and tracking to
associate points/detections correctly:

- If $T_{lidar \rightarrow radar}$ (Eq. 1) is **not** applied, radar and LiDAR
  points that hit the same physical object land in different $(x, y)$
  locations of the plane purely because of the fixed sensor-mounting offset —
  the fusion backbone would then be matching geometry from two different
  places, not two views of the same place.
- If $T_{ego}$ (Eq. 2) is **not** applied, a *static* object appears to move
  between consecutive frames simply because the ego-vehicle moved — the
  tracker would confuse this apparent drift with real object motion, biasing
  motion cues and ID association.

The figure below shows this on synthetic LiDAR/radar point clouds of a single
vehicle, in the same BEV plane as the equations above: left column is the raw,
unaligned points; right column is the result of applying Eq. 1 (top row) and
Eq. 2 (bottom row). Without alignment the two point sets describe what looks
like two different objects; after alignment they overlap on the same footprint.

![Radar/LiDAR alignment — before vs. after](../figures/radar_lidar_alignment.png)

## BEV 3D-to-2D Projection

Point clouds are rasterized to a top-down (bird's-eye-view) image for
inference visualization (see `birds_eye_point_cloud`,
`external/gnd/module/lidar_projection.py`, used from `utils/visualization_eval.py`).
For a resolution $res$ (meters/pixel) and a point $(x, y, z)$:

$$
\text{col} = \left\lfloor \frac{-y}{res} \right\rfloor - \left\lfloor \frac{side_{min}}{res} \right\rfloor
\qquad
\text{row} = \left\lfloor \frac{x}{res} \right\rfloor - \left\lfloor \frac{fwd_{min}}{res} \right\rfloor
$$

$$
\text{pixel} = 255 \cdot \frac{z_{max} - \text{clip}(z,\, z_{min},\, z_{max})}{z_{max} - z_{min}}
$$

**Why:** this turns an unordered 3D point cloud into a dense, ground-plane-indexed
2D image (height-encoded), which is inexpensive to render and lets predicted
boxes/tracks be overlaid and visually compared against ground truth per frame.

## Caso de estudio

## Datasets relacionado

## Tracking vs Re identificacion

TODO: Write a project introduction here.
