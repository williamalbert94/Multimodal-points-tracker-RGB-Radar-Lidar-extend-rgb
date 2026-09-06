# Feature analysis — what each fusion mode buys

Measures whether a fusion mode makes objects **easier to re-identify**, without
training anything. The Phase-2 ReID stage compares object descriptors across
frames, so the question that decides a mode's worth is: *do two observations of
the same object look more alike to each other than to other objects?*

## Method

For every GT **moving** object, the points inside its 3D box are reduced to a
fixed-length, **position-invariant** descriptor:

| Component | Dims | Why |
|---|---:|---|
| `log(1 + n_points)` | 1 | how much evidence the sensor returned |
| extent per axis | 3 | object size |
| normalised covariance eigenvalues | 3 | linearity / planarity / sphericity |
| mean + std per feature channel | 2·F | RCS, Doppler, LiDAR channels |

Position is excluded on purpose — a descriptor encoding *where* the object is
would "re-identify" by location and measure nothing about appearance.

Two scores, both on z-scored descriptors:

* **Rank-1** — for each observation, is its nearest neighbour *in a different
  frame* the same track id? The standard ReID protocol.
* **Silhouette** — how tightly observations of one identity group relative to
  other identities.

### Corpus matching (important)

Modes differ in how many objects clear the 2-point validity threshold — the
LiDAR-based ones see more — and rank-1 gets *harder* as identities are added.
Scoring each mode on its own corpus therefore compares different tasks. By
default every mode is scored on the **observations shared by all modes**.

This is not a detail. On a 60-frame validation sample:

| Mode | Own corpus (unfair) | Common corpus (fair) |
|---|---:|---:|
| `none` | 38.0 % (166 obs, 32 ids) | 37.6 % |
| `lidar_base` | 33.0 % (285 obs, 48 ids) | **40.0 %** |
| `matched` | 33.3 % (237 obs, 40 ids) | 39.4 % |
| `radar_base` | 41.6 % (166 obs, 32 ids) | **41.8 %** |

Uncorrected, `lidar_base` looks 5 points *worse* than the baseline; corrected, it
is 2.4 points better. The apparent loss was entirely the larger, harder corpus.
`--no-common` reproduces the unfair version if you want to see it.

## Results

Validation split, `stride 10`, 60 frames, 165 common observations, 32 identities:

| Mode | Dim | Rank-1 | Δ vs baseline | Silhouette |
|---|---:|---:|---:|---:|
| `none` (radar only) | 11 | 37.6 % | — | −0.143 |
| `lidar_base` | 11 | 40.0 % | **+2.4** | −0.076 |
| `matched` | 11 | 39.4 % | **+1.8** | −0.108 |
| **`radar_base`** | 17 | **41.8 %** | **+4.2** | −0.085 |

Every fusion mode improves re-identifiability, and `radar_base` leads — it keeps
Doppler and RCS on every point while adding the vertical structure radar cannot
measure. Silhouette stays negative throughout, which is expected for hand-crafted
descriptors on 2-point objects: these numbers rank the *input representations*,
they are not a ReID system's performance.

## Running

```bash
python examples/feature_analysis/analyze_features.py
python examples/feature_analysis/analyze_features.py --modes none radar_base
python examples/feature_analysis/analyze_features.py --split both --stride 5 --max-frames 400
python examples/feature_analysis/analyze_features.py --no-common      # unfair, for contrast
```

From the host:

```bash
docker run --rm -e VOD_ROOT=/project/view_of_delft_PUBLIC \
  -v "$PWD":/project \
  -v /path/to/view_of_delft_PUBLIC:/project/view_of_delft_PUBLIC:ro \
  -w /project will/tracker_multimodal_mira \
  conda run -n mira python examples/feature_analysis/analyze_features.py
```

A JSON summary lands in `results/` (git-ignored).

## FiftyOne (Voxel51) — visual inspection

Builds a browsable dataset with a UMAP projection of the descriptors, so
identity clusters can be inspected directly.

**1. Rebuild the image** (adds `fiftyone`, `fiftyone-db`, `umap-learn`):

```bash
cd docker && docker compose build tracker_multimodal_mira
```

**2. Start the container and launch the app:**

```bash
cd docker && docker compose run --rm --service-ports tracker_multimodal_mira \
  conda run -n mira python examples/feature_analysis/launch_fiftyone.py
```

`--service-ports` is what publishes 5151; without it `docker compose run`
ignores the `ports:` block and the app is unreachable from the host.

**3. Open <http://localhost:5151>** and switch datasets with the selector.

Useful flags:

```bash
launch_fiftyone.py --list                     # what already exists
launch_fiftyone.py --modes none radar_base    # which modes to build
launch_fiftyone.py --split both --stride 5    # bigger corpus
launch_fiftyone.py --rebuild                  # regenerate from scratch
```

### Known packaging pitfall

`docker/Dockerfile` pins **`pymongo==4.8.0`**. FiftyOne 0.23.8 declares only
`pymongo>=3.12`, but it still imports `pymongo.database._check_name`, which
pymongo removed in 4.9. Without the pin, pip resolves to 4.17 and `import
fiftyone` fails with:

```
ImportError: cannot import name '_check_name' from 'pymongo.database'
```

If you bump FiftyOne, re-check whether the pin is still needed.

### Networking notes

Two settings in `docker/docker-compose.yml` make this work, both easy to miss:

* `FIFTYONE_DEFAULT_APP_ADDRESS=0.0.0.0` — binding to `localhost` inside a
  container only serves the container itself, so the host sees nothing.
* `FIFTYONE_DATABASE_DIR=/project/.fiftyone/db` — puts the embedded MongoDB on
  the mounted volume so datasets survive container restarts. That directory is
  git-ignored.

`analyze_features.py --fiftyone` does the same export inline. Either way the
script degrades gracefully: without FiftyOne installed it prints a notice and
the numeric analysis still runs.
