
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
- [Method](#method)
- [🚀 Getting Started](#-getting-started)
  - [Phase 1: Backbone Pre-training](#phase-1-backbone-pre-training)
  - [Phase 2: Tracking Model](#phase-2-tracking-model)
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

Build and run the Docker environment:

```bash
docker compose build
docker compose run
```

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

### Phase 1: Backbone Pre-training

Pre-train the backbone using the segmentation configuration:

```bash
python train.py --config config/segmentation_phase1.yml
```

This will:
- Create logs in TXT format
- Save the best model checkpoint based on mIoU or F1-score at point level
- Optionally, use pre-trained weights (link provided in repository)

#### Evaluation and Visualization

To evaluate the trained model on the test set:
1. Point to the pre-trained weights in `config/segmentation_phase1.yml`
2. Enable `plot_segmentation` to generate inference visualizations

**Expected output:**

![Segmentation Results](./docs/figures/repo2.png)

### Phase 2: Tracking Model

Train the tracking model with gallery-based re-identification:

```bash
python train.py --config config/reid_phase2.yml
```

Enable `plot_reid` to visualize results similar to the following:

![Tracking Results](./docs/figures/image.png)

*Left: Ground truth track IDs | Right: Predicted track IDs (vehicle class example)*

**Note:** This phase trains with perfect boxes and segmentation. During inference, the pre-trained model from Phase 1 is used.

## Evaluation

Run inference with the pre-trained segmentation model:

```bash
python eval.py --config config/eval_config.yaml
```

Pre-trained weights for both phases are available. Simply point to them in the configuration and run the evaluation script.

### Inference Demo

![Inference Visualization](./docs/figures/inferencia.gif)

*Real-time tracking and segmentation inference on View of Delft test sequences*

## Results

Performance comparison on the View of Delft dataset:

| Method | sAMOTA ↑ | AMOTA ↑ | MOTA ↑ | IDSW ↓ | MT ↑ | ML ↓ | mIoU ↑ |
|--------|----------|---------|--------|--------|------|------|--------|
| CenterPoint | 43.21 | 14.40 | 38.44 | 149 ‡ | 19.12 | 38.24 | evals only tracker |
| CenterPoint-PP | 44.54 | 16.33 | 43.96 | 40 ‡ | 19.12 | 54.41 | evals only tracker |
| AB3DMOT | 51.23 | 15.00 | 46.72 | 28 ‡ | 20.59 | 39.71 | evals only tracker |
| AB3DMOT-PP *(PointPillars det.)* | 60.71 | 21.51 | 49.38 | 20 ‡ | 26.47 | 33.82 | n/a |
| **AB3DMOT-RT** *(RaTrack det., ours)* | — | — | — | **313** § | — | — | 57.00 § |
| RaTrack | 74.16 | 31.50 | 67.27 | 404 † | 42.65 | 14.71 | 57.00 |
| **LocalGlobalFusion (Ours)** | **76.50** | **34.03** | **69.45** | **119** | — | — | **65.00** |
| VoxelPointFusion | 70.34 | 30.70 | 64.00 | 320 | — | — | 53.30 |

> ⚠️ **Do not read the IDSW column as a ranking.** The four baseline rows above
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

For the same reason, **IDSW must always be read together with MT/ML**. The MT/ML
columns for our method still need to be filled in before this table can argue
that 84 reflects genuine identity preservation rather than reduced coverage.

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
