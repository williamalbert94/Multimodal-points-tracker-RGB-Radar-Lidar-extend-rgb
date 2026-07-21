# AB3DMOT on View-of-Delft — ID-switch recovery

Runs the **real AB3DMOT tracker** (3D Kalman + Hungarian, from
`/home/williamramirez/multimodal_reid/AB3DMOT`) on VoD using RaTrack's split, and
reports IDSW with the **same** `idsw_eval` used by [`../ratrack/`](../ratrack/),
so the numbers are directly comparable.

## Read this first: what this is NOT

**This is not `AB3DMOT-PP`.** AB3DMOT is tracking-by-*detection* — it never
detects anything, it only associates boxes you give it. RaTrack's `AB3DMOT-PP`
baseline (sAMOTA 60.71 / AMOTA 21.51 / MOTA 49.38 in their Table I) feeds it
**PointPillars detections trained on VoD radar**, and those detections were never
released. So that exact figure cannot be reproduced without training PointPillars.

What this example gives you instead is the **IDSW floor**: AB3DMOT fed with
*ground-truth boxes as perfect detections*. It isolates how much identity is lost
by motion-only association alone, with detection error removed entirely.

When you do have PointPillars detections, pass them with `--dets <dir>` (KITTI
label format, one `.txt` per frame) to get the true AB3DMOT-PP number.

## Results (moving-only GT, RaTrack's 4 validation clips, 1296 frames)

| Setting | Established GT tracks | Tracks with ≥1 switch | **IDSW** |
|---|---:|---:|---:|
| Perfect detections, `max_age=2` | 116 | 36 (31%) | **95** |
| Perfect detections, `max_age=10` | 116 | 39 (34%) | 107 |

For comparison, measured identically in [`../ratrack/`](../ratrack/):

| Tracker | Detections | IDSW | Tracks w/ switch |
|---|---|---:|---|
| AB3DMOT | **perfect (GT)** | 95 | 31% |
| RaTrack | real (its own) | 404 | 86% |

> ⚠️ **Protocol mismatch — do not put these two straight into a paper table.**
> Both numbers above score *every* moving GT object. RaTrack's published
> evaluation instead discards objects with fewer than 5 radar points, and under
> that rule its IDSW is **133**, not 404 (see [`../ratrack/`](../ratrack/), where
> the figure is cross-validated against RaTrack's own MOTA/MODA gap). The
> AB3DMOT floor here has **not** yet been re-measured under that validity rule,
> so 95 vs 133 is not apples-to-apples either.

### Two findings that matter for the thesis

1. **Detection failure is not the whole story.** Even with *flawless* detection,
   motion-only association still loses identity 95 times across 116 tracks. So
   IDSW is not purely a detector problem — the association/re-identification
   mechanism is a substantial contributor in its own right.
2. **Longer coasting does not fix it.** Raising `max_age` from 2 to 10 does not
   reduce switches; it slightly *increases* them (95 → 107), because a track
   that coasts longer drifts and then grabs the wrong detection. This is direct
   evidence that the fix is appearance-based re-identification, not simply
   letting the Kalman filter predict for more frames.

## Ablation: is RaTrack's IDSW caused by its detector or its associator?

`run_hybrid.py` answers this by holding the **detector fixed** and swapping only
the association algorithm. Both arms consume RaTrack's own exported clusters, are
matched to GT with the same point-based IoU, and are scored with the same IDSW
definition — so any difference is attributable to association alone.

```bash
docker compose run --rm ab3dmot_eval python run_hybrid.py
```

Association parameters are AB3DMOT's **own suggested values** from `get_param()`
(`configs/KITTI.yml` selects `det_name: pointrcnn`), not tuned here. Track counts
are reported because configs that establish fewer tracks trivially accumulate
fewer switches:

| Association (all on RaTrack's detections) | Tracks | IDSW | IDSW/track |
|---|---:|---:|---:|
| RaTrack — learned affinity + Sinkhorn, as shipped | 75 | **367** | 4.89 |
| **AB3DMOT official `Pedestrian` (`giou_3d −0.4, min_hits 1, max_age 4`)** | **75** | **232** | **3.09** |
| AB3DMOT official `Car` (`giou_3d −0.2, min_hits 3, max_age 2`) | 54 | 112 | 2.07 |
| AB3DMOT official `Cyclist` (`dist_3d −2`, sign-corrected) | 68 | 68 | 1.00 |
| AB3DMOT, tuned here (`dist_3d −2, min_hits 1, max_age 15`) | 75 | 150 | 2.00 |

**The like-for-like row is `Pedestrian`**: VoD's moving objects are almost all
VRUs, and it is the only official config that establishes the same 75 tracks as
RaTrack. On identical detections and identical coverage, AB3DMOT's stock
association yields **232 vs RaTrack's 367 — 37% fewer switches**. The `Car` and
`Cyclist` rows show lower raw counts only because `min_hits: 3` suppresses short
tracks, dropping coverage to 54 and 68 — the same coverage trap discussed above,
which is why the normalised column is shown.

So the error splits roughly:

* **~135 switches (37%) come from the associator** and are recovered by stock
  AB3DMOT parameters; tuning `max_age` up to 15 recovers ~217 (59%).
* **~150–232 switches survive the associator swap** — a floor imposed by
  detection instability, i.e. DBSCAN clusters that split, merge and change
  membership between frames, so the object hypothesis itself is not stable.

**`max_age` matters most.** Raising it (2 → 4 → 15) monotonically reduces
switches, the opposite of the perfect-detection experiment above where it changed
nothing: with flawless continuous detections a track never needs to coast, but
RaTrack's real cluster detections are intermittent and long coasting bridges the
gaps.

### Two bugs worth knowing about

**In this example (fixed).** Detections for which AB3DMOT emitted no track were
passed to the evaluator with id `-1`, which it treated as a legitimate id — a GT
object repeatedly matched to `-1` looked perfectly stable and its real switches
vanished. This understated IDSW badly for `min_hits > 1` configs (it reported 4
switches for a config that had in fact collapsed). `to_eval()` now drops them.

**In AB3DMOT itself.** The `Cyclist` config sets `dist_3d` with `thres = 2`
(positive), but `compute_affinity` returns `-dist3d(...)` (always ≤ 0) and
`data_association` rejects a pair when `affinity < threshold`. As shipped this
rejects essentially every association — run verbatim it established only 5 of 75
tracks. The table above uses `thres = -2`; treat any `dist_*` threshold in
`get_param()` as needing its sign checked.

> **Consequence for benchmarking a ReID stage.** Comparing a ReID module against
> RaTrack's as-shipped 367 overstates its benefit, because ~59% of those switches
> are recoverable with classical tracking alone. A ReID contribution should be
> measured on top of a *tuned* classical baseline (150), not against untuned
> association.

Caveat: this compares RaTrack *as published* against AB3DMOT *tuned here* —
RaTrack's ids are fixed in its exported files, so its side could not be tuned.

## Configuration notes (important, non-obvious)

**Association metric: `dist_3d`, not AB3DMOT's default `giou_3d`.** Using
`giou_3d` on VoD produces a catastrophic failure mode: most moving objects are
VRUs whose boxes are only ~0.5–0.7 m wide, and with a moving ego vehicle they
shift ~1.4 m per frame in camera coordinates. Consecutive boxes therefore have
**zero overlap**, affinity falls below threshold, and the tracker re-births a new
ID every single frame (observed: GT track 773 cycling `15→19→22→24→27→30→…`).
Measured on delft_1 (first 200 frames): 56 spurious births with `giou_3d` vs 13
with `dist_3d`, and `dist_3d` is stable for thresholds in [-2, -10].

AB3DMOT normally counters ego motion with its `ego_com` compensation, but that
path needs KITTI-format `oxts`, and VoD ships poses as `pose/*.json` instead, so
it is disabled here.

**3D IoU for *evaluation* uses shapely, not AB3DMOT's `dist_metrics.iou`.** Its
convex-hull polygon clipping divides by zero when two box edges are exactly
parallel — exactly what happens when GT boxes are fed as detections and the
tracker output coincides with the GT box. It returns NaN (and we measured
`giou_3d(box, box) = 1.2949`, outside GIoU's valid [-1, 1] range) which crashes
qhull. The shapely implementation in `run_ab3dmot.py` is numerically robust.

**`_shims/xinshuo_visualization`** is a stub for one unused helper: AB3DMOT's
`model.py` imports `vis.py` unconditionally, which imports `random_colors`, but
visualization is off. The other `xinshuo_*` modules come from this repo's
vendored `external/`.

## Running

Build the dedicated image (kept separate from the main `will/tracker_mra`):

```bash
cd examples/ab3dmot
docker compose build          # -> will/tracker_multimodal_mira
```

The dataset path defaults to the dev-box location; override it with `VOD_HOST`:

```bash
# perfect detections (IDSW floor) — reproduces 116 tracks / 95 IDSW
docker compose run --rm ab3dmot_eval

# elsewhere:
VOD_HOST=/your/view_of_delft_PUBLIC docker compose run --rm ab3dmot_eval

# true AB3DMOT-PP, once you have PointPillars detections
docker compose run --rm ab3dmot_eval \
  python run_ab3dmot.py --dets /path/to/pointpillars_dets

# single clip / longer coasting
docker compose run --rm ab3dmot_eval \
  python run_ab3dmot.py --clips delft_10 --max-age 5
```

CPU-only by design — the tracker is a Kalman filter, no CUDA needed.

If the dataset volume is wrong or missing, `check_dataset()` aborts with an
explicit error instead of silently reporting "0 ID switches" (which is what a
missing mount would otherwise produce, since zero GT means zero switches).

## Caveats

- The comparison against RaTrack's 406 is **not** like-for-like on detections:
  AB3DMOT here gets perfect boxes, RaTrack gets its own real ones. It bounds the
  association-only contribution; it does not rank the two trackers.
- Association parameters (`dist_3d`, `thres=-4`, `min_hits=1`) were chosen to
  avoid the pathological re-birth behaviour, not tuned for best IDSW. Different
  settings will shift the absolute number.
