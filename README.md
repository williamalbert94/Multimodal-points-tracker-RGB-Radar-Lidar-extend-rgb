
<div align="center">
  <img src="./docs/logo/logo.png" alt="Logo" width="600">
</div>

<div align="center">

[![python](https://img.shields.io/badge/python-3.9-3776AB.svg)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/pytorch-2.2.0-EE4C2C.svg)](https://pytorch.org/)
[![docker](https://img.shields.io/badge/docker-supported-2496ED.svg)](./docker/Dockerfile)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

📖 [Introduction](./docs/pages/introduction.md) | ⚙️ [Installation](./docs/pages/installation.md) | 🚀 [Get Started](#-getting-started) | 📘 [Documentation](./docs/pages/documentation.md)

</div>

## Table of Contents

- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [Environment Setup](#environment-setup)
  - [Pre-trained weights](#pre-trained-weights)
- [Method](#method)
- [🚀 Getting Started](#-getting-started)
  - [Phase 1: Backbone Pre-training](#phase-1-backbone-pre-training)
  - [Phase 1: Inference and Evaluation](#phase-1-inference-and-evaluation)
  - [Phase 2: Detections, Re-ID and Tracking](#phase-2-detections-re-id-and-tracking)
- [Evaluation](#evaluation)
- [Results](#results)
- [References](#references)

## Prerequisites

This project requires access to the View of Delft dataset and corresponding tracking annotations.

Weigths: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18933010.svg)](https://doi.org/10.5281/zenodo.19012896)
## Dataset

### Required Data

1. **View of Delft Dataset**
   - Link: [tudelft-iv/view-of-delft-dataset](https://github.com/tudelft-iv/view-of-delft-dataset/tree/main)
   - Requires authorization from the authors

2. **Tracking Annotations**
   - Link: [RaTrack Annotations](https://github.com/LJacksonPan/RaTrack/tree/main)

3. **2D Projected Annotations**
   - Available on Zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18933010.svg)](https://doi.org/10.5281/zenodo.18933010)

### Expected Dataset Structure

```
view_of_delft_PUBLIC/
├── lidar/
│   ├── ImageSets/
│   ├── testing/
│   │   ├── calib/
│   │   ├── image_2/
│   │   ├── label_2_tracking/
│   │   ├── pose/
│   │   └── velodyne/
│   └── training/
│       ├── calib/
│       ├── image_2/
│       ├── label_2/
│       ├── label_2_tracking/
│       ├── pose/
│       └── velodyne/
├── radar/
│   ├── ImageSets/
│   ├── testing/
│   │   ├── calib/
│   │   ├── image_2/
│   │   ├── pose/
│   │   └── velodyne/
│   └── training/
│       ├── calib/
│       ├── image_2/
│       ├── label_2/
│       ├── pose/
│       └── velodyne/
├── radar_3frames/
│   └── ...
└── radar_5frames/
    └── ...
```

## Environment Setup

Everything runs inside the container; nothing needs to be installed on the host
beyond Docker and the NVIDIA container toolkit.

**1. Point the compose file at your dataset.** `docker/docker-compose.yml` has
the VoD path hard-coded — edit this line before building:

```yaml
volumes:
  - /home/williamramirez/view_of_delft_PUBLIC:/project/view_of_delft_PUBLIC   # <- your path
  - ./root:/home/${USER}
```

**2. Build and open a shell.**

```bash
cd docker
docker compose build
docker compose run --rm tracker_multimodal_mira
```

The compose entrypoint is `bash`, and `~/.bashrc` activates the `mira` conda
environment and compiles the PointNet++ CUDA ops (`external/lib/setup.py`) on
first shell start. That compilation is GPU-architecture specific, so it happens
in the container rather than at image build time.

Inside the container the repository is at `/project` and the interpreter is
`/opt/conda/envs/mira/bin/python`. Every command below assumes you are at
`/project`.

To run a single command without an interactive shell:

```bash
docker compose run --rm tracker_multimodal_mira -lc \
  '/opt/conda/envs/mira/bin/python -u -m tracker.runner.train --config tracker/config/seg_exp_Q_lidarflow.yaml'
```

### Pre-trained weights

Download from
[Google Drive](https://drive.google.com/drive/folders/1ES5mvwTNd957d1wRxdbb_hcUX75J3KGs?usp=sharing)
and place the files so the paths below resolve:

```
tracker/checkpoints/seg_exp_Q_lidarflow/best_miou_model.pth   # Phase 1 backbone (mIoU 0.7877)
tracker/results/reid_head.pth                                 # Phase 2 Re-ID head
```

With these two files you can skip training entirely and go straight to
[Inference](#phase-1-inference-and-evaluation) and [Evaluation](#evaluation).

### GPU Considerations

When using PointNet with GPU acceleration, the library needs to be loaded and compiled for the specific GPU. This environment has been tested on:
- Personal machine with **RTX 3060**
- Cluster with dual **A100-PCIE-40GB**

**Note:** In multi-GPU scenarios, the environment does not properly utilize hybrid usage.

## Method

Our approach uses a local-global feature fusion backbone for multimodal 3D object tracking:

![Backbone Architecture](./docs/figures/backbone.png)

The architecture combines:
- **Local features**: Point-level representations from PointNet++
- **Global features**: Scene-level context from voxelized representations
- **Multimodal fusion**: Integration of LiDAR point clouds and camera images

## 🚀 Getting Started

All commands run inside the container from `/project`. `PY` is used below as a
shorthand for the interpreter:

```bash
PY=/opt/conda/envs/mira/bin/python
```

### Phase 1: Backbone Pre-training

Trains the moving/static point segmentation backbone (LiDAR-radar early fusion,
temporal branch, MASG global aggregation). Everything is driven by the YAML:

```bash
$PY -u -m tracker.runner.train --config tracker/config/seg_exp_Q_lidarflow.yaml
```

`seg_exp_Q_lidarflow.yaml` is the configuration behind the reported results:
2048 points, `in_channels: 6`, `fusion: radar_base` with a 2.0 m radius,
`use_temporal` plus `lidar_temporal`, 95 epochs at `lr 3e-4`. It writes the best
checkpoint by mIoU to `tracker/checkpoints/<exp_name>/best_miou_model.pth` and
logs to MLflow (`http://localhost:5000`, started by `docker compose up mlflow`).

Other configs in `tracker/config/` are the ablation arms: `seg_exp_B_fusion`
(fusion only), `seg_exp_C_movil`, `seg_exp_A_vrcomp`. `seg_train_smoke.yaml` is
a short run for checking the pipeline end to end.

### Phase 1: Inference and Evaluation

Scores the backbone on the validation split and writes figures, per-frame
predictions and a metrics report:

```bash
$PY -u -m tracker.runner.inference_seg \
    --config     tracker/config/seg_exp_Q_lidarflow.yaml \
    --checkpoint tracker/checkpoints/seg_exp_Q_lidarflow/best_miou_model.pth
```

Produces under `tracker/results/seg_exp_Q_lidarflow/`:

| path | contents |
|---|---|
| `metrics.txt` | mIoU / IoU_moving / IoU_static / F1 / recall, plus a threshold sweep |
| `vis/` | 3-panel figures: GT, RGB, prediction |
| `vis_bev/` | 2-panel bird's-eye view |
| `data/` | per-frame predictions in RaTrack's text format |

Useful flags: `--cada N` writes a figure every N frames (`--cada 100000`
effectively disables them and makes the run much faster), `--clases Car`
restricts the evaluation to one object type, `--sin-postproc` disables cluster
post-processing, `--sufijo` renames the output folder.

`metrics.txt` reports mIoU under **two conventions**: the micro-averaged one
(pooled TP/FP/FN over the split) and the frame-weighted one that replicates
RaTrack's `eval_motion_seg`. Only the second is comparable to their published
57.0 — see the caveat in [Results](#results).

![Segmentation Results](./docs/figures/repo2.png)

### Phase 2: Detections, Re-ID and Tracking

Phase 2 runs on top of a trained Phase-1 backbone, in three steps.

**Step 1 — precompute detections.** Boxes are GT boxes kept only when the
segmentation head fires inside them, which isolates the tracking study from the
detection bottleneck (see [Results](#results) for what this implies). You need
the validation split for evaluation and the train split for fitting the Re-ID
head:

```bash
CKPT=tracker/checkpoints/seg_exp_Q_lidarflow/best_miou_model.pth
CFG=tracker/config/seg_exp_Q_lidarflow.yaml

# validation, moving objects only (RaTrack's evaluation scope)
$PY -u -m tracker.tracking.precompute_detections_gtseg \
    --config $CFG --checkpoint $CKPT --split val --moving-only \
    --umbral 0.5 --min-pts 1 \
    --out tracker/results/detections_gtseg_val_mov.pkl

# train split, for the Re-ID head
$PY -u -m tracker.tracking.precompute_detections_gtseg \
    --config $CFG --checkpoint $CKPT --split train \
    --out tracker/results/detections_gtseg_train.pkl
```

Add `--min-radar-pts N` to additionally require N raw radar returns per box.

**Step 2 — train the Re-ID head.** Triplet loss over GT track identities, on
pooled backbone features plus box geometry:

```bash
$PY -u -m tracker.tracking.train_reid \
    --train tracker/results/detections_gtseg_train.pkl \
    --out   tracker/results/reid_head.pth \
    --epochs 40 --lr 1e-3 --margin 0.3 --embedding-dim 256
```

**Step 3 — run the tracker.** The gallery tracker combines appearance, geometry,
density, motion and spatial cues under Hungarian assignment:

```bash
# motion only
$PY -u -m tracker.tracking.track_inference \
    --detections tracker/results/detections_gtseg_val_mov.pkl \
    --gt-moving-only \
    --out tracker/results/track_mov

# with the appearance cue
$PY -u -m tracker.tracking.track_inference \
    --detections tracker/results/detections_gtseg_val_mov.pkl \
    --gt-moving-only --reid-head tracker/results/reid_head.pth \
    --out tracker/results/track_mov_reid
```

Writes `metrics.txt`, `vis/` (GT | RGB | prediction with track IDs) and `data/`
(one row per track, radar frame). `--max-age`, `--match-threshold` and
`--cada` control track lifetime, the association threshold and figure density.

![Tracking Results](./docs/figures/image.png)

*Left: Ground truth track IDs | Right: Predicted track IDs (vehicle class example)*

## Evaluation

`track_inference.py` already prints MOTA, IDF1, MOTP, MT/PT/ML, ID switches and
TP/FP/FN at a single operating point. The integral metrics need the AB3DMOT
protocol sweep, which re-runs the tracker across confidence thresholds:

```bash
$PY -u -m tracker.tracking.amota_ab3dmot \
    --detections tracker/results/detections_gtseg_val_mov.pkl \
    --gt-moving-only --explain
```

`--explain` prints the full curve (threshold, recall, TP/FP/FN, IDSW, MOTA,
sMOTA) so the averaging is auditable rather than a single number.

**Evaluation scope flags.** These decide which GT objects count, and they change
the numbers substantially, so quote them alongside any result:

| flag | effect | `n_gt` on val |
|---|---|---:|
| *(none)* | every annotated object, including parked ones | 11457 |
| `--gt-moving-only` | moving objects only, RaTrack's scope | 3474 |
| `--gt-moving-only --gt-min-radar-points 2` | also requires 2 raw radar returns | 2581 |
| `--fov-only` | drops objects outside the camera frustum | — |

The moving-object flag reads column 2 of `label_2`, which VoD ships row-aligned
with `label_2_tracking`; the last column of the tracking file is constant 1 and
is **not** a moving flag. Without it, parked vehicles are scored as missed
detections and object recall drops from 79% to 23%.

To re-evaluate RaTrack's own released predictions under the same protocol, see
[`examples/ratrack/`](./examples/ratrack/), which also recovers its unpublished
ID-switch count from the metric identities.

### Inference Demo

![Inference Visualization](./docs/figures/inferencia.gif)

*Real-time tracking and segmentation inference on View of Delft test sequences*

## Results

Performance comparison on the View of Delft dataset:

| Method | sAMOTA ↑ | AMOTA ↑ | MOTA ↑ | IDSW ↓ | MT ↑ | ML ↓ | mIoU ↑ |
|--------|----------|---------|--------|--------|------|------|--------|
| CenterPoint | 43.21 | 14.40 | 38.44 | -- | 19.12 | 38.24 | evals only tracker |
| CenterPoint-PP | 44.54 | 16.33 | 43.96 | -- | 19.12 | 54.41 | evals only tracker |
| AB3DMOT | 51.23 | 15.00 | 46.72 | -- | 20.59 | 39.71 | evals only tracker |
| AB3DMOT-PP *(PointPillars det.)* | 60.71 | 21.51 | 49.38 | 313 ‡ | 26.47 | 33.82 | n/a |
| RaTrack | 74.16 | 31.50 | 67.27 | 404 | 42.65 | 14.71 | 57.00 |
| **LocalGlobalFusion (Ours)** § | 74.54 | 30.23 | **78.44** | **9** | **58.5** | **12.3** | **74.66** |
| VoxelPointFusion | 70.34 | 30.70 | 64.00 | 320 | — | — | 53.30 |

§ **Read the protocol before quoting this row.** Measured on the VoD validation
split (`delft_1/10/14/22`, 1288 frames) with the moving-object filter enabled,
which gives `n_gt = 3474` against RaTrack's 3116 — close but not identical, since
RaTrack drops only `DontCare` while we additionally exclude `Pedestrian`,
`bicycle_rack` and `ride_uncertain`. Two caveats that work in our favour and must
be carried with the numbers:

* **Detections are GT boxes filtered by segmentation**, not a detector's output,
  so `FP = 0` by construction. That inflates MOTA (78.44, MODA 78.70) and sAMOTA
  (which measures precision and identity, and is blind to recall — see
  [`examples/ratrack/`](./examples/ratrack/)). The detection side is therefore an
  upper bound, not a like-for-like result. What *is* comparable is the identity
  column: **9 switches against RaTrack's 329–367 at near-identical MODA
  (78.70 vs 77.83)**, i.e. 0.33 vs 13.57 switches per 100 tracked objects.
  sAMOTA and AMOTA both come from the AB3DMOT-protocol sweep in
  `amota_ab3dmot.py` (IoU 0.25), so the two are on the same footing; the
  per-frame accumulator in `track_inference.py` prints a much higher sAMOTA
  (99.67) because it scores a single operating point rather than averaging over
  the recall range, and should not be quoted against RaTrack.
* **sAMOTA and AMOTA are capped by reachable recall, and the cap is real.** Both
  average over 40 evenly spaced recall targets; our detection stage tops out at
  78.7% recall, so 9 of the 40 targets are unreachable and contribute 0. That
  puts a ceiling of 31/40 = 77.5% on sAMOTA before any tracking quality is
  considered, and we reach 74.54 of it — i.e. ~96% average sMOTA across every
  operating point we can actually occupy. AMOTA lands at 30.23 against RaTrack's
  31.50, essentially level. Reproduce with
  `tracker/tracking/amota_ab3dmot.py --gt-moving-only --explain`.

The mIoU cell is the **frame-weighted** mean that replicates RaTrack's
`eval_motion_seg`, which is the only convention comparable to their published
57.0. The micro-averaged mIoU over the same predictions is 77.67 and must not be
quoted against RaTrack; per-class it decomposes into IoU_moving 57.88 and
IoU_static 97.47. Reproduce with
`tracker/runner/inference_seg.py` (both conventions are printed) and
`tracker/tracking/track_inference.py --gt-moving-only`.

> **Do not read the IDSW column as a ranking.** The four baseline rows above
> show 20–149 switches against RaTrack's 404, which does **not** mean they
> preserve identity better — a switch can only be recorded for an object the
> tracker actually re-acquires, and those baselines recover roughly half the
> moving instances RaTrack does (MODA 41.96–49.86 vs 77.83), while mostly-losing
> 33–54% of trajectories. Their low counts measure detection failure. See
> [On comparing identity switches](#on-comparing-identity-switches) for the
> recall-normalised view, which is the comparable one.

### AB3DMOT-RT: why the ablation row re-uses RaTrack's detector

`AB3DMOT-RT` is `AB3DMOT-PP` with the detector swapped: the same AB3DMOT
tracking algorithm (3D Kalman + Hungarian), fed **RaTrack's DBSCAN clusters**
instead of PointPillars boxes. Following the naming already used in the
literature, the suffix denotes the detector — `-PP` = PointPillars,
`-RT` = RaTrack.


RaTrack and AB3DMOT differ in **two** places simultaneously: the detector
(per-frame DBSCAN clustering of radar points vs PointPillars 3D boxes) and the
associator (learned affinity + Sinkhorn vs 3D Kalman + Hungarian). Comparing the
published rows therefore cannot attribute RaTrack's ID switches to either
component — the comparison is confounded.

The ablation row removes that confound by **holding the detector fixed**. Both it
and the RaTrack row consume RaTrack's own exported per-frame clusters, are matched
to ground truth with the same point-based IoU, and are scored with the same IDSW
definition; **only the association algorithm changes**. Re-using RaTrack's
detector is also the only option available in practice: AB3DMOT is
tracking-by-detection and never detects anything itself, and the PointPillars
detections behind the published `AB3DMOT-PP` row were never released.

The result isolates the associator's contribution: **404 → 313 switches, a 22%
reduction from swapping association alone**, with identical track coverage (81
established GT tracks in both arms) and a `dist_3d` gate with `max_age = 2`.
AB3DMOT's stock `Pedestrian` parameters do better still (267, −34%), and tuning
`max_age` upward reaches 150 under the stricter validity rule. The switches that
survive *any* associator are the floor imposed by detection instability — DBSCAN
clusters that split, merge and change membership between frames, so the object
hypothesis itself is not stable.

All figures use a validity rule of ≥1 radar point. Under RaTrack's shipped
default (`min_obj_points: 2`) the same ablation reads 367 → 288. Both rows must
be quoted from the same validity setting; see
[Documentation §9](./docs/pages/documentation.md#9-measured-results) for the full
grid.

> **Consequence for benchmarking Phase 2.** Measuring a ReID stage against
> RaTrack's as-shipped 404 overstates its benefit, since roughly a third of those
> switches are recoverable with classical tracking alone. The ReID contribution
> should be measured on top of a tuned classical baseline, not against untuned
> association.

### On comparing identity switches

**Raw IDSW counts are not comparable across trackers at different recall.** A
switch can only be recorded for an object the tracker actually re-acquires, so a
method that detects less accumulates fewer switches without preserving identity
any better. The fix is to normalise by the objects each tracker actually keeps
(recall ≥ MODA). Applying the metric identity below to every baseline in
RaTrack's own Table I (n_gt = 3116 valid moving instances):

| Method | MOTA | MODA | IDS | GT tracked | **IDS per 100 tracked** | MT ↑ | ML ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| CenterPoint | 38.44 | 41.96 | 110 | 1307 | 8.39 | 19.12 | 38.24 |
| CenterPoint-PP | 43.96 | 44.91 | 30 | 1399 | 2.12 | 19.12 | 54.41 |
| AB3DMOT | 46.72 | 47.38 | 21 | 1476 | 1.39 | 20.59 | 39.71 |
| AB3DMOT-PP | 49.38 | 49.86 | 15 | 1554 | **0.96** | 26.47 | 33.82 |
| **RaTrack** | 67.27 | 77.83 | 329 | 2425 | **13.57** | 42.65 | 14.71 |
| **Ours** (moving-only, n_gt = 3474) § | 78.44 | 78.70 | 9 | 2734 | **0.33** | 58.5 | 12.3 |

Normalised, the picture is coherent rather than anomalous: every box-based
tracker sits at 1–2 switches per 100 tracked objects, CenterPoint's
centre-matching sits at 8.4, and RaTrack is the clear outlier at 13.6. That
ordering follows the architecture. AB3DMOT propagates *parameterised 3D boxes*
through a Kalman filter, so the track state — and with it the identity — is
stable, but it depends on PointPillars regressing usable boxes, which sparse
radar defeats (ML 33.82). RaTrack instead re-runs DBSCAN clustering *from
scratch on every frame*, precisely to avoid box regression on sparse radar; that
buys markedly better detection (MODA 77.83, ML 14.71) at the cost of cluster
identities that split, merge and churn between frames.

**RaTrack therefore solved detection on sparse radar and left identity
unsolved** — which is precisely the gap the Phase-2 re-identification stage
targets.

#### Why higher MOTA can coexist with far more ID switches

MOTA sums `FN + FP + IDS`, and on sparse radar the detection terms are an order
of magnitude larger than the identity term, so they dominate the arithmetic.
Decomposing the same rows:

| Method | FN+FP | IDS | Total error | **Identity share of error** |
|---|---:|---:|---:|---:|
| CenterPoint | 1809 | 110 | 1918 | 5.7% |
| CenterPoint-PP | 1717 | 30 | 1746 | 1.7% |
| AB3DMOT | 1640 | 21 | 1660 | 1.2% |
| AB3DMOT-PP | 1562 | 15 | 1577 | **0.9%** |
| **RaTrack** | 691 | 329 | 1020 | **32.3%** |

RaTrack accepts 314 extra ID switches in exchange for eliminating 872 detection
errors — a net reduction of 557, which is why its MOTA is far higher despite the
switches. MOTA rewards that trade because it does not distinguish error types.

This also answers why the simplest baselines show the *fewest* switches: for them
identity is only ~1% of total error. They do not preserve identity well in any
interesting sense — they detect so little, with such a stable box+Kalman state,
that identity never becomes the binding constraint. RaTrack inverts this: by
solving detection, it turns identity from ~1% of the error budget into a third
of it. **Adding re-identification to an AB3DMOT-style baseline would address
0.9% of its error; adding it to a RaTrack-style architecture addresses 32%** —
which is the quantitative case for this project's Phase 2.

Note finally that IDSW, MOTA and MODA all conflate detection with association by
construction; IDF1 and HOTA's AssA component are designed to isolate association
quality and are worth reporting alongside.

For the same reason, **IDSW must always be read together with MT/ML**. Our own
row now carries them: MT 58.5 / ML 12.3 against RaTrack's 42.65 / 14.71, at
MODA 78.70 vs 77.83. Coverage is therefore comparable or better, which is what
licenses reading the 9 switches as identity preservation rather than as an
artefact of tracking fewer objects. The caveat of §Results still applies — our
detections are GT boxes filtered by segmentation, so the *detection* side of
that comparison is an upper bound.

† RaTrack publishes no IDSW. Recovered by re-evaluating its released per-frame
predictions on the VoD validation split under RaTrack's own protocol: moving
objects only, point-based IoU 0.25, and its default validity rule
`min_obj_points: 2` (`src/configs_eval.yaml`). Cross-checked against RaTrack's
*own published* table via the metric identity `MOTA = 1 − (FN+FP+IDS)/n_gt` and
`MODA = 1 − (FN+FP)/n_gt`, hence `IDS = (MODA − MOTA)·n_gt`: with n_gt = 3116
valid moving instances that implies 329, against our measured 367 (+12%, two
independent routes). See [`examples/ratrack/`](./examples/ratrack/) for the
reproducible evaluator.

‡ Derived, not measured — AB3DMOT-PP's predictions were never released, so this
figure comes from applying the same identity to its published MOTA/MODA row. It
must be read at its operating point: AB3DMOT-PP recovers roughly half the moving
instances that RaTrack does, so compare the normalised column above rather than
the raw count. Note this row is genuine `AB3DMOT-PP` (PointPillars detections),
which is a *different experiment* from the ablation row below it.

§ Ablation measured here; see [`examples/ab3dmot/`](./examples/ab3dmot/)
(`run_hybrid.py`). **mIoU is inherited, not re-measured**: segmentation is
produced by the detector, which is RaTrack's in this arm, so the value is
identical by construction. The remaining cells are deliberately left blank rather
than filled: our reimplementation of the tracking-metric suite does not reproduce
RaTrack's published MOTA/MODA/MT/ML (its evaluator was never released, and our
GT point-set construction caps matchable recall at ~69% against their ~78%), so
absolute values from it would be misleading. IDSW is reported because it is
independently validated against RaTrack's own table via the identity above,
agreeing within 12% across four validity thresholds — and because the 367→232
delta is measured with one evaluator on identical detections, making it
evaluator-independent.

## References

This repository contains basic elements for evaluating metrics and loading data, adapted from:

- [AB3DMOT](https://github.com/xinshuoweng/AB3DMOT)
- [RaTrack](https://github.com/LJacksonPan/RaTrack)
- [View of Delft Dataset](https://github.com/tudelft-iv/view-of-delft-dataset)

### BibTeX Citations

```bibtex
@inproceedings{weng2020_ab3dmot,
  title     = {AB3DMOT: A Baseline for 3D Multi-Object Tracking and New Evaluation Metrics},
  author    = {Weng, Xinshuo and Wang, Jianren and Held, David and Kitani, Kris},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2020}
}

@article{pan2024ratrack,
  title   = {RaTrack: Moving Object Detection and Tracking with 4D Radar Point Cloud},
  author  = {Pan, Liang and Liu, Zhihao and Thompson, Simon and others},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2024}
}

@inproceedings{palffy2022vod,
  title     = {Multi-Class Road User Detection with 3+1D Radar in the View-of-Delft Dataset},
  author    = {Palffy, Andras and Dong, Jiaao and Kooij, Julian F. P. and Gavrila, Dariu M.},
  booktitle = {IEEE Robotics and Automation Letters},
  volume    = {7},
  number    = {2},
  pages     = {4961--4968},
  year      = {2022}
}
```

---

<div align="center">
</div>
