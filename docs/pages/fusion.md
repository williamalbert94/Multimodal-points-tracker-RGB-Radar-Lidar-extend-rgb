# Early LiDAR–radar fusion

How the two modalities are combined, why the obvious formulation does not
survive contact with the sensor densities, and how to switch between the
options.

- [Where fusion belongs](#where-fusion-belongs)
- [The association problem](#the-association-problem)
- [Modes](#modes)
- [Configuration](#configuration)
- [Choosing a mode](#choosing-a-mode)

## Where fusion belongs

Fusion is implemented in the **dataloader**, not in the model. Three reasons:

* It is a choice of *input representation*, not of architecture. Keeping it in
  the dataset means the same network trains on any of the variants.
* It runs inside DataLoader worker processes, so the cost overlaps with GPU
  compute instead of competing with it. The association is a k-d tree query on
  a few hundred thousand points — cheap on CPU, and moving it to GPU would be
  slower (see [`loader_vod.py`](../../tracker/dataset/loader_vod.py) for the
  measurements behind that).
* By the time it runs, the extrinsic $T_{radar \leftarrow lidar}$ and the
  ego-motion compensation have already been applied, so both modalities live in
  one consistent frame and association is a pure nearest-neighbour problem.

The compensated cloud is rebuilt from the fused points with the same $T_{ego}$,
so the temporal pair stays aligned whichever mode is selected.

## The association problem

The natural formulation — LiDAR provides geometry, each LiDAR point copies the
attributes $[\rho, v_r]$ of the nearest radar return within $\epsilon$ — is
degenerate at VoD's sensor ratio. A frame carries **~178,000 LiDAR points
against ~300 radar returns**. Measured over six frames spread across the splits:

| $\epsilon$ | LiDAR points receiving radar attributes |
|---|---|
| 0.5 m | 2.9 – 8.9 % |
| 1.0 m | 8.3 – 15.9 % |
| 2.0 m | 16.9 – 26.0 % |

Sampling `num_points` from that population makes it worse still: measured on a
real frame, `lidar_base` with a 4096-point cap left **0.4 %** of the sampled
points carrying non-zero radar attributes. The radar channel becomes padding and
the network degenerates to LiDAR-only geometry — the opposite of the intent.

## What fusion is actually for

The frame-level ratio above makes association awkward, but it also states the
opportunity precisely. Measured per **moving object** over 394 GT instances
(`delft_1` + `delft_14`, boxes in the radar frame):

| | median | mean | p90 | objects with < 5 points |
|---|---:|---:|---:|---:|
| Radar | 2 | 3.0 | 7 | **78.7 %** |
| LiDAR | 30 | 119.4 | 196 | **13.5 %** |

**Roughly four out of five moving objects carry fewer than five radar returns.**
A shape descriptor — which is what a re-identification embedding is — cannot be
computed from two points; there is no geometry to describe. LiDAR brings the same
objects to a median of 30 points and cuts the "too sparse to describe" cases by
about six times.

That is the concrete target for fusion in this project: not detection, which
RaTrack already solves on radar alone, but the **appearance/shape channel the
ReID stage depends on**. It is the same conclusion the metric analysis reaches
from the other direction — identity, not detection, is the binding constraint
(see [Documentation](documentation.md#why-higher-mota-can-coexist-with-far-more-id-switches)).

> **Comparability caveat.** RaTrack and the other baselines are radar-only. A
> LiDAR+radar method is not directly comparable to them, and a table that mixes
> the two must say so explicitly, otherwise a reviewer will read the gain as
> algorithmic when part of it is extra sensing. Report the radar-only
> configuration alongside as the controlled comparison.

## Modes

Set with `args.fusion`. Point counts and coverage below are measured on
`delft_1`, frame 00002 ($\epsilon = 0.5$ m):

| Mode | Points | Features | Radar coverage | Notes |
|---|---:|---:|---:|---|
| `none` | 319 | 3 | 100 % | radar only — **current behaviour, the default** |
| `lidar_base` | 4096 | 2 | **0.4 %** | faithful to the write-up; see caveat above |
| `matched` | 714 | 2 | **100 %** | LiDAR points that have a radar match |
| `radar_base` | 319 | 5 | **100 %** | radar points + local LiDAR geometry |

**`matched`** keeps the LiDAR points whose nearest radar return is within
$\epsilon$ and drops the rest. Every surviving point carries real geometry *and*
real radar attributes, and the count lands in the same order of magnitude the
network is tuned for. The cost is that LiDAR structure far from any radar return
is discarded.

**`radar_base`** inverts the association: keep the radar points — all of which
have true $[\rho, v_r]$ — and describe the LiDAR around each one. The three extra
channels are the neighbour count inside $\epsilon$ (log-compressed so it cannot
dominate), the mean height and the height spread:

$$
\mathbf{x}_i = \big[\underbrace{x, y, z}_{\text{radar}},\;
\underbrace{\rho,\; v_r}_{\text{radar}},\;
\underbrace{\log(1 + |\mathcal{N}_i|),\; \overline{z}_{\mathcal{N}_i},\;
\Delta z_{\mathcal{N}_i}}_{\text{LiDAR}}\big]^\top \in \mathbb{R}^{3+5}
$$

with $\mathcal{N}_i = \{ j : \|\mathbf{l}_j - \mathbf{r}_i\| \le \epsilon \}$.
This gives the radar points the vertical structure they lack while keeping
Doppler and RCS on every point.

## Configuration

```yaml
fusion: matched          # none | lidar_base | matched | radar_base
fusion_radius: 0.5       # association radius, metres
fusion_max_points: 4096  # cap for the LiDAR-based modes
```

`fusion: none` is the default, so existing configs keep the current radar-only
behaviour unchanged.

**Feature width.** `none`, `lidar_base` and `matched` emit ≤ 3 channels and the
trainer slices the first two, so the network is untouched. `radar_base` emits
**5**, and the extractor must be constructed accordingly:

```python
# model/model.py — currently hardcoded to 2
self.pn_head = LocalGlobalFusionSimple(args.num_points, 2)
#                                                       ^ 5 for radar_base
# and FlowDecoder(in_dim=3 + 2 + 256 + 256) becomes 3 + 5 + 256 + 256
```

The loader prints the required `in_channels` at startup when a mode needs more
than the default.

## Measured impact

`examples/feature_analysis/` scores each mode on re-identifiability without
training anything: per-object, position-invariant descriptors, evaluated with
the standard rank-1 protocol (is an observation's nearest cross-frame neighbour
the same track id?). Validation split, 165 observations shared by all modes,
32 identities:

| Mode | Dim | Rank-1 | Δ vs baseline | Silhouette |
|---|---:|---:|---:|---:|
| `none` (radar only) | 11 | 37.6 % | — | −0.143 |
| `lidar_base` | 11 | 40.0 % | +2.4 | −0.076 |
| `matched` | 11 | 39.4 % | +1.8 | −0.108 |
| **`radar_base`** | 17 | **41.8 %** | **+4.2** | −0.085 |

Every mode improves re-identifiability and `radar_base` leads, which matches the
per-object density argument above: it is the only mode that keeps Doppler and
RCS on every point while adding the vertical structure radar cannot measure.

> **The comparison must use a shared corpus.** Modes differ in how many objects
> clear the 2-point threshold, and rank-1 gets harder as identities are added.
> Scored on its own corpus `lidar_base` appears **5 points worse** than the
> baseline; scored on the common observations it is **2.4 points better**. The
> apparent loss was entirely the larger, harder evaluation set — the same
> coverage trap that distorts IDSW comparisons.

## Choosing a mode

`radar_base` is the recommended starting point. It preserves the current point
count and the radar semantics the pipeline already works with, adds the vertical
structure radar cannot measure, and keeps both modalities on every point — so
any gain is attributable to the LiDAR channels rather than to a change in
sampling density. It does require the two-line model change above.

`matched` is the alternative worth running if the goal is denser geometry: no
model change, roughly twice the points, still 100 % coverage. Compare the two
against the `none` baseline on the same split before committing.

`lidar_base` is kept for completeness and for reproducing the formulation as
originally written; it should not be expected to outperform the baseline at this
radius, and the 0.4 % coverage figure explains why.
