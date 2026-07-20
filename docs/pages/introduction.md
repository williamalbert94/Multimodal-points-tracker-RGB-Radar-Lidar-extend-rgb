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

**2. Temporal (ego-motion) alignment.** Radar returns are sparse (a VoD frame
typically yields on the order of only a few dozen points per object), so the
pipeline compensates and accumulates radar sweeps from consecutive frames into
one common frame instead of relying on a single sweep. Each frame $t$ has its
own pose chain $T_{odom \rightarrow camera}(t)$ and $T_{camera \rightarrow radar}(t)$
(these are the extrinsics VoD's calibration actually exposes per frame, via
`FrameTransformMatrix`), so they are first composed per frame:

$$
T_{odom \rightarrow radar}(t) = T_{odom \rightarrow camera}(t) \cdot T_{camera \rightarrow radar}(t)
$$

The relative ego-motion between frame $t$ and $t{+}1$ is then the pose change
expressed in the radar frame — how far and in which direction the sensor rig
itself translated/rotated while the world stood still:

$$
T_{ego} = \big(T_{odom \rightarrow radar}(t)\big)^{-1} \cdot T_{odom \rightarrow radar}(t+1)
$$

$T_{ego}$ is a rigid-body (SE(3)) transform, i.e. it carries both the
translation and the rotation of the ego-vehicle between the two timestamps —
this matters because on turns a pure translation offset would under-correct
points far from the sensor. The earlier point cloud is finally re-expressed
in the later frame by inverting this motion out of it:

$$
\mathbf{p}_{comp} = \mathbf{p}_{t} \cdot \big(T_{ego}\big)^{-T}
$$

so that $\mathbf{p}_{comp}$ and the frame-$(t{+}1)$ points describe the scene
from the *same* ego pose and can be safely stacked into one denser radar point
cloud, or diffed frame-to-frame to read off an object's true motion.

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
  between consecutive frames simply because the ego-vehicle moved — every
  point in the scene, not just the tracked object, inherits the same apparent
  drift. Two things break as a consequence:
  - **Radar accumulation.** Stacking raw (uncompensated) sweeps to densify the
    sparse radar point cloud would smear a single object's returns across
    several offset positions instead of reinforcing one footprint.
  - **Tracking/motion cues.** The tracker's frame-to-frame displacement signal
    would mix real object motion with ego-vehicle motion, biasing velocity
    estimates and increasing ID switches — a parked car would look like it is
    moving toward or away from the ego-vehicle.

The figure below renders this in a style closer to real automotive sensor
output rather than an idealized outline: the LiDAR trace only covers the two
vehicle faces actually visible to (facing) the ego sensor, denser near the
closest corner as real beam sampling would produce, while the radar is a
handful of sparse returns clustered on strong reflectors (corners / wheel
arches) — the dashed rectangle is the ground-truth 3D box shown only for
reference. Left column is the raw, unaligned points; right column is the
result of applying Eq. 1 (top row) and Eq. 2 (bottom row). Without alignment
the two point sets describe what looks like two different objects; after
alignment they collapse onto the same footprint.

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

## Case Study

Two distinct failure modes motivate why motion-only, single-modality tracking
is not sufficient here:

**1. Occlusion inherent to RGB.** A camera projects the 3D scene onto a 2D
image plane, so two objects that are meters apart in 3D but aligned along the
same viewing ray fully occlude each other in the image, even though neither
is physically blocking the other in space. This is a structural limitation of
the RGB modality, not a corner case — it is exactly the effect AB3DMOT points
to when explaining why 3D tracking has fewer identity mismatches than 2D
tracking: *"tracking in 3D can better resolve depth ambiguities and lead to
fewer mismatches than tracking in 2D"* (Weng et al., 2020). In this project
RGB is only used for the projection/visualization overlay
(`plot_rgb_projection_simple`, `utils/visualization_rgb.py`), while the
primary 3D perception comes from radar/LiDAR precisely to avoid inheriting
this failure mode.

**2. Range/FOV boundary effect (not occlusion).** A separate, easily confused
failure mode is a pipeline configured with a maximum operating range (e.g.
20 m): an instance oscillating near that boundary — briefly at 21 m, back at
19 m — repeatedly exits and re-enters the valid detection zone. Nothing is
physically blocking the sensor's view here; the drop is an artifact of a hard
distance cutoff, not true occlusion. It produces the same symptom (an ID gets
dropped and a new one is created on re-entry) but the fix is different: a
softer threshold with hysteresis, rather than an appearance/ReID-based
re-association. Attributing observed ID switches to the correct one of these
two causes (true occlusion vs. range-boundary flicker) is necessary before
concluding a re-identification module is the right solution.

## Related Datasets

This project uses the **View-of-Delft (VoD)** dataset: 8600+ frames of
synchronized and calibrated 64-layer LiDAR, stereo camera, and 3+1D radar,
recorded in urban traffic in Delft. VoD's own official benchmarks cover only
3D object detection and trajectory prediction — there is no official VoD
tracking benchmark. The tracking-ID annotations used here come from
**RaTrack** (Pan et al., 2024), an external contribution built on top of VoD
(see the References section and the repository's main
[README](../../README.md)).

**Test-set restriction.** RaTrack's own dataset section states this
explicitly: *"As an official benchmark specific for 3D object detection, the
annotations of its test split are not publicly available, thereby we
evaluate our trained models with its validation split, which is unseen
during our training process."* This matches what we verified directly against
the tracking labels shipped with this project: the `label_2_tracking` files
are entirely missing for exactly the clips listed as `test` in
`dataloader/track_vod_3d.py` (`delft_7, delft_8, delft_16, delft_18,
delft_20, delft_21, delft_25`), while every clip in `train`/`val` has full,
dense label coverage. Like RaTrack, this project therefore trains on `train`
and treats `val` as the effective held-out evaluation set — `test` exists as
a frame-index split in code but cannot be scored against ground truth because
no public labels exist for it.

## Tracking vs. Re-Identification

Multi-object tracking (MOT) and re-identification (ReID) cover different
failure modes. **Tracking** maintains an object's identity across
*consecutive* frames using motion/geometry (e.g. predicted position + Hungarian
matching). **Re-identification** recovers an object's identity by *appearance*
after a gap — an occlusion, a missed detection, a temporary exit from the
field of view — where motion continuity alone breaks down. Phase 2 of this
project (`config/reid_phase2.yml`) trains a gallery-based ReID head on top of
the fused radar/LiDAR backbone precisely so identity can survive such gaps,
not just frame-to-frame motion.

**Existing work on VoD.** The official View-of-Delft benchmarks cover 3D
object detection and trajectory prediction only — there is no official VoD
tracking or re-identification benchmark. RaTrack is, to our knowledge, the
only published multi-object *tracker* evaluated on VoD, and it is purely
motion-based: it clusters points via motion segmentation and scene-flow
estimation and builds an affinity matrix for association, with no appearance
or ReID component. We did not find published work adding a gallery-based
re-identification stage on VoD.

**Existing work on nuScenes.** nuScenes has an official 3D MOT benchmark, but
its leading entries (AB3DMOT-style trackers, CenterPoint tracking) are also
predominantly motion/geometry-based (Kalman filter + IoU or
Mahalanobis-distance matching on 3D boxes). Appearance-based ReID research
that does use nuScenes tends to stay in the camera modality (e.g. a
nuScenes-derived pedestrian ReID dataset built from 2D camera crops) or adds
ReID at the box/track level inside later fusion trackers (e.g. Stereo3DMOT,
joint detection+ReID for multi-camera 3D). We did not find nuScenes work doing
point-level (LiDAR/radar) gallery-based ReID comparable to this project's
Phase 2.

**Why this gap exists.**
1. Appearance-based ReID relies on discriminative texture (color, decals,
   license plates) that camera images provide but point clouds largely do
   not — radar returns in particular carry almost no visual texture, and even
   dense LiDAR gives only geometric shape. This is why most vehicle-ReID
   literature (VeRi-776, VehicleID, etc.) targets camera crops rather than
   point clouds.
2. Autonomous-driving tracking benchmarks (KITTI, nuScenes, VoD) are built
   from short, single-ego-vehicle sequences where objects rarely leave and
   re-enter the field of view for long — motion continuity is usually enough
   to keep an identity, so there has been little incentive to add a dedicated
   appearance-ReID stage, unlike multi-camera surveillance ReID where an
   identity must be re-acquired across disjoint camera views.

Given this, the gallery-based radar/LiDAR ReID trained in Phase 2 targets a
combination — point-cloud-level, VoD-specific, tracking *and* ReID — that does
not appear to have prior published results to directly compare against.

References:
- Pan, L. et al., ["RaTrack: Moving Object Detection and Tracking with 4D Radar Point Cloud"](https://arxiv.org/abs/2309.09737), 2024.
- Palffy, A. et al., ["Multi-Class Road User Detection with 3+1D Radar in the View-of-Delft Dataset"](https://www.researchgate.net/publication/358328092_Multi-class_Road_User_Detection_with_31D_Radar_in_the_View-of-Delft_Dataset), IEEE RA-L, 2022.
- Caesar, H. et al., ["nuScenes: A Multimodal Dataset for Autonomous Driving"](https://arxiv.org/abs/1903.11027), CVPR, 2020.
- [View-of-Delft dataset & benchmarks — TU Delft Intelligent Vehicles Group](https://intelligent-vehicles.org/datasets/view-of-delft/)

