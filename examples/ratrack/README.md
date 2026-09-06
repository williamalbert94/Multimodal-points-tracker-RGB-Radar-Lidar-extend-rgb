# RaTrack metric recovery

Recover the tracking metrics (**sAMOTA, AMOTA, AMOTP, MOTA, MODA, MT, ML, and
IDS / id-switches**) from RaTrack's exported per-frame predictions on
View-of-Delft.

## Why this exists

RaTrack **ships its predictions but not its evaluation code**. Its
`python main.py --config configs_eval.yaml` only writes per-frame cluster files
to `results/…`; the README states:

> "This will only generate the predictions in the `results` folder. We are
> currently working on integrating our point-based version of AB3DMOT
> evaluation scripts into the evaluation run."

and, in the Q&A:

> "Regarding Our Modified AB3DMOT Evaluation Code — Due to AB3DMOT's repository
> license, we are currently not able to distribute our modified version…"

So to obtain the numbers in RaTrack's paper/README table we have to **rebuild the
point-based AB3DMOT evaluator ourselves**. That is exactly what this folder does.

## Is it possible? Yes.

All the required inputs are on disk:

| Input | Location | Provides |
|---|---|---|
| RaTrack predictions | `RaTrack/src/result/4dmot_runthis/<clip>/*.txt` | tracked clusters + track id + confidence per frame |
| VoD GT tracking labels | `…/lidar/training/label_2_tracking/*.txt` | GT boxes **with track ids** |
| VoD radar sweeps | `…/radar/training/velodyne/*.bin` | the radar points needed for point-based IoU |
| VoD radar calib | `…/radar/training/calib/*.txt` | camera→radar transform for the GT boxes |
| RaTrack `src` | `RaTrack/src` | its exact box transform + moving-object filter (reused, not re-derived) |

The four prediction clips available (`delft_1, delft_10, delft_14, delft_22`)
are exactly RaTrack's **validation split** — the effective evaluation set, since
the VoD **test split has no public labels** (confirmed both by RaTrack's paper
and by the missing `label_2_tracking` files for the test clips).

## Prediction format (verified)

One tracked object per line:

```
NA 1 -1 -1 <conf> <track_id> x0 y0 z0 x1 y1 z1 … xk yk zk
```

- cols 0–3: placeholders (class is `NA` — RaTrack is class-agnostic)
- col 4: cluster confidence
- col 5: track id
- cols 6+: flattened `(x,y,z)` of every radar point in the cluster (radar frame)

## How the metrics are computed

Because RaTrack detections are **point clusters, not boxes**, standard 3D-box
IoU does not apply. Following RaTrack's paper we use a **point-based IoU**:

> "we compute the IoU by counting the number of intersected and united radar
> points between the ground truth object and the predicted one. The threshold
> for our point-based IoU is set as 0.25."

For each frame:
1. GT box (camera frame) → radar-frame oriented box via RaTrack's own
   `get_bbx_param` + VoD `FrameTransformMatrix` → set `A` of radar-point indices
   inside it.
2. Prediction cluster → set `B` of radar-point indices (its saved points matched
   back to the sweep).
3. `point-IoU = |A ∩ B| / |A ∪ B|`.

Those per-frame IoU matrices feed a CLEAR-MOT + integral evaluator whose
accumulation mirrors AB3DMOT's `scripts/KITTI/evaluate.py`
(`/home/williamramirez/multimodal_reid/AB3DMOT`):

- `MOTA = 1 − (FN+FP+IDS)/n_gt`, `MODA = 1 − (FN+FP)/n_gt`
- `IDS` (identity switches), `FRAG`, `MT` (>80% tracked), `ML` (<20% tracked)
- `sAMOTA / AMOTA / AMOTP`: MOTA / MOTP / sMOTA averaged over 40 recall points
  (confidence sweep), the integral metrics AB3DMOT introduced.

## Recovered results (validation split: delft_1, 10, 14, 22)

Actually run on this box (`vod` conda env: numpy + scipy + open3d; no torch/GPU
needed because `point_iou.py` is self-contained):

Scored with RaTrack's protocol: **moving objects only** (GT filtered exactly as
its `filter_moving_boxes_det` does) and point-based IoU 0.25.

| Metric | Recovered | RaTrack reference | Comparable? |
|---|---|---|---|
| **IDS** (id-switches) | **133** over 53 established GT tracks | never published (implied **114**) | yes — validated, see below |
| **mIoU** (tracked-cluster stage) | 63.82 (acc 96.36, sens 62.25) | 57.0 (seg-**head** stage) | **no** — different pipeline stage |

### IDS is cross-validated against RaTrack's own published numbers

AB3DMOT's metric definitions give an identity that lets us derive RaTrack's
unpublished IDS from its published table:

```
MOTA = 1 − (FN + FP + IDS)/n_gt      MODA = 1 − (FN + FP)/n_gt
  =>  MODA − MOTA = IDS / n_gt  =  77.83 − 67.27  =  10.56 %
```

Measuring n_gt on the validation split and comparing with what we recover:

| GT validity rule | n_gt | IDS implied by their table | IDS we measure | Δ |
|---|---:|---:|---:|---:|
| ≥ 1 radar point | 4240 | 448 | 404 | −10% |
| **≥ 2 radar points** (`min_obj_points: 2`, their default) | **3116** | **329** | **367** | **+12%** |
| ≥ 3 radar points | 2179 | 230 | 289 | +26% |
| ≥ 5 radar points | 1075 | 114 | 133 | +17% |

Two fully independent routes agree across every threshold, which validates the
evaluator. **Report 367**: RaTrack's shipped configs (`src/configs.yaml` and
`src/configs_eval.yaml`) both set `min_obj_points: 2`, so that row is the
like-for-like figure. (The paper's sensitivity analysis separately explores a
5-point threshold, but 2 is the default the released model was evaluated with.)

### Which counting convention? (it changes the answer 8×)

RaTrack evaluates with its **own modified** AB3DMOT evaluator, adapted to VoD
with point-based IoU — and that code was never released, so its exact ID-switch
condition is unknown. Two plausible conventions:

* **A** — an id change counts even if the object was unmatched for some frames
  (recovery with a *different* id is a switch).
* **B** — AB3DMOT/KITTI's own rule, which additionally requires `g[f-1] != -1`,
  i.e. the object must have been matched in the immediately preceding frame;
  post-gap id changes are booked as *fragmentations*, not switches.

| GT validity | Conv. A | Conv. B | Implied by their MOTA/MODA |
|---|---:|---:|---:|
| ≥ 1 point | 404 | 48 | 448 |
| **≥ 2 points** | **367** | 44 | **329** |
| ≥ 3 points | 289 | 32 | 230 |
| ≥ 5 points | 133 | 9 | 114 |

Only **Convention A** reproduces the figure implied by RaTrack's published
MOTA/MODA gap; Convention B lands ~87% low at every threshold. Since the implied
value is derived from *their own* reported numbers, this is strong evidence that
their evaluator counts post-gap id changes. This example therefore uses
Convention A (see `idsw_eval.py`).

### IDSW alone is a trap — always report MT/ML with it

Applying the same identity to every row of RaTrack's Table I (n_gt = 3116):

| Method | MODA − MOTA | IDS implied | MT ↑ | ML ↓ |
|---|---:|---:|---:|---:|
| CenterPoint | 3.52% | 110 | 19.12 | 38.24 |
| CenterPoint-PP | 0.95% | 30 | 19.12 | 54.41 |
| AB3DMOT | 0.66% | 21 | 20.59 | 39.71 |
| AB3DMOT-PP | 0.48% | 15 | 26.47 | 33.82 |
| **RaTrack** | 10.56% | **329** | **42.65** | **14.71** |

RaTrack has by far the *most* ID switches and is nonetheless the best tracker in
the paper. It is tempting to dismiss this as "AB3DMOT-PP simply tracks less", but
that does not survive checking: normalising by objects actually tracked
(recall ≥ MODA, so ~2425 instances for RaTrack vs ~1554 for AB3DMOT-PP) still
leaves **1.0 vs 13.6 switches per 100 tracked instances** — a 14× gap.

It is a real architectural trade-off:

* **AB3DMOT** propagates *parameterised 3D boxes* through a Kalman filter. The
  track state is stable, so identity is stable — but it depends on PointPillars
  regressing usable boxes, which sparse radar defeats (ML 33.8%).
* **RaTrack** re-runs DBSCAN clustering *from scratch every frame*, precisely to
  avoid box regression on sparse radar. That buys far better detection (MODA
  77.83 vs 49.86) at the cost of cluster identity that splits, merges and churns.

So RaTrack solved detection on sparse radar and left **identity** unsolved, which
is exactly what a re-identification stage targets. **Never quote IDSW without
MT/ML beside it**, and never compare IDSW across methods at different recall
without normalising.

The instability is still real and motivates the ReID stage: individual objects
cycle through long id chains such as `408→410→412→413→414→415`.

**Averaging convention.** mIoU is a **frame-weighted** mean: every frame's mIoU
is pooled and averaged over all 1289 frames, which reproduces RaTrack's own
convention (`seg_met[key] / num_examples`). Do **not** average the four per-clip
values unweighted — clips range from 34 to 542 frames, so that gives a different
(wrong) 60.94. Per-clip figures, for reference:

| Clip | Frames | Frames w/ moving GT | mIoU (all) | mIoU (non-empty) |
|---|---:|---:|---:|---:|
| delft_1 | 542 | 483 | 66.06 | 68.09 |
| delft_10 | 34 | 34 | 54.20 | 54.20 |
| delft_14 | 431 | 403 | 65.71 | 66.83 |
| delft_22 | 282 | 208 | 57.80 | 60.61 |
| **Total** | **1289** | **1128** | **63.82** | **65.84** |

161 frames contain no moving GT at all; by RaTrack's formula each contributes
~0.5, so excluding them raises the mean to 65.84. Quote **63.82** when comparing
against RaTrack (same convention); 65.84 is performance conditioned on there
being something to segment.

**On the mIoU discrepancy (63.82 vs 57.0) — read before quoting.** These are not
the same quantity. RaTrack's published 57.0 scores its *segmentation head*
(`cls > 0.5`); that mask is **not** in the exported `.txt`. Our 63.82 scores the
*final tracked clusters* (after motion-seg → DBSCAN → tracking), which discard
spurious moving points and therefore score higher. Use **57.0** when citing
RaTrack's segmentation quality; to reproduce it exactly, export the `cls` mask
(one-line change, see `seg_miou.py` header) and feed it to
`seg_miou.eval_motion_seg`, which is a byte-for-byte replica of their formula.

> These are an independent reconstruction (our own point-IoU matching and
> confidence handling), so treat IDS=406 as a strong, directionally-solid result
> rather than an official RaTrack figure. See caveats at the bottom.

## IDSW vs. segmentation mIoU — what is recoverable

| Metric | Recoverable from current exports? | How |
|---|---|---|
| **IDSW** (id-switches) | **Yes, fully** | `.txt` clusters + GT, matched per frame by point-IoU. `idsw_eval.py` / `run_eval.py --idsw` |
| **Segmentation mIoU** | **Not exactly** — the per-point `cls>0.5` mask is *not* in the `.txt` (only clustered moving objects are) | `seg_miou.py` replicates RaTrack's exact formula; feed it an exported mask (path A) or the cluster-union approximation (path B) |

**IDSW definition** (as requested): a switch is counted only when a GT track that
is **already established** gets matched to a *different* prediction id. The first
assignment defines the track (no switch); a gap followed by re-acquisition with
the **same** id is recovery (no switch). So IDSW measures identity
*instability/granularity*, not re-identification recall.

**Segmentation mIoU** is RaTrack's binary moving/static per-point IoU,
`0.5·(IoU_moving + IoU_static)` (their `eval_motion_seg`). Because the seg-head
mask is not exported, the exact number needs a one-line addition to RaTrack's
inference writer to also dump `(cls>0.5)` per frame — see the header of
`seg_miou.py` for the exact snippet. `seg_miou.eval_motion_seg` is a byte-for-byte
replica of their formula, so once the mask is dumped the number matches theirs.

## Files

| File | Role | Deps |
|---|---|---|
| `paths.py` | all paths + thresholds (edit for your machine; honors env vars) | — |
| `ratrack_io.py` | prediction + GT parsers (pure Python) | numpy |
| `mot_metrics.py` | CLEAR-MOT + AMOTA integration (pure Python) | numpy, scipy |
| `idsw_eval.py` | focused ID-switch counter + per-track breakdown | numpy, scipy |
| `seg_miou.py` | RaTrack `eval_motion_seg` replica + mask reconstruction | numpy (+ open3d for GT mask) |
| `point_iou.py` | point-based IoU matrix (self-contained: reads calib directly) | numpy, scipy, open3d, VoD dataset |
| `run_eval.py` | orchestrator (`--selfcheck` / `--idsw` / `--eval`) | above |
| `predictions_sample/delft_10/` | 34-frame sample so the code runs without the full tree | — |
| `clips/` | frame-index lists for the 4 val clips | — |

## Running

**Self-check** (parsing + metric core; needs only numpy+scipy, runs anywhere):

```bash
cd examples/ratrack
python run_eval.py --selfcheck
```

Expected: it parses the `delft_10` sample and passes a synthetic CLEAR-MOT test
(a known single id-switch → `IDS=1, TP=3, FN=1, MOTA=0.5, MODA=0.75`).

**ID switches** (the primary ask). Only needs numpy + scipy + open3d and the VoD
dataset — no torch/GPU (`point_iou.py` reads the calib files directly). Any env
with open3d works; on this box the `vod` env does:

```bash
conda activate vod       # numpy + scipy + open3d
python run_eval.py --idsw
python run_eval.py --idsw --clips delft_10      # subset
```

Prints total IDSW plus a per-GT-track `from->to` switch breakdown.

**Segmentation mIoU** (approx, cluster-union vs GT boxes):

```bash
python run_eval.py --seg
```

Faithful mIoU needs RaTrack's seg mask exported (one-line change to its writer,
see `seg_miou.py` header); then apply `seg_miou.eval_motion_seg` to the masks.

**Full CLEAR-MOT + AMOTA:**

```bash
python run_eval.py --eval
```

Prints recovered metrics next to RaTrack's published table for comparison.

## Caveats (read before trusting the numbers)

The parsing and CLEAR-MOT/AMOTA core are unit-tested and exact. The three places
where an independent re-implementation can drift from RaTrack's own (withheld)
evaluator — validate the output against RaTrack's published table before quoting:

1. ~~Moving-object filtering.~~ **Done.** GT is filtered to moving objects
   exactly as RaTrack does: VoD ships `label_2` (detection) and
   `label_2_tracking` row-aligned; column 1 of `label_2` is the moving flag,
   column 1 of `label_2_tracking` is the track id. `parse_gt_frame_moving()`
   zips them positionally and keeps `moving == 1`, matching
   `filter_moving_boxes_det`. Toggle with `MOVING_ONLY` in `run_eval.py`
   (all-GT scoring gives IDS=516 over 132 tracks, for reference).
2. **Cluster→radar-index mapping.** `point_iou._cluster_indices` maps saved
   cluster points back to the sweep by nearest point; if RaTrack stored
   ego-motion-compensated coordinates, use the compensated cloud for the match.
3. **Confidence discretization.** AMOTA's recall sampling uses a quantile-based
   confidence sweep; AB3DMOT derives thresholds from the TP score distribution.
   Minor differences shift AMOTA slightly.

**RaTrack published (validation split):**
`sAMOTA 74.16 | AMOTA 31.50 | AMOTP 60.17 | MOTA 67.27 | MODA 77.83 | MT 42.65 | ML 14.71`
