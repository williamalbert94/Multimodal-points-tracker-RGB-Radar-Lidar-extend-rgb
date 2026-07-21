"""View-of-Delft dataset loading.

Two loaders are provided, both exposing `TrackingDataVOD`:

* `datagen_vod`   — production loader: point-cloud augmentation, per-clip frame
                    indexing and box-motion features. **Use this for training.**
* `track_vod_3d`  — baseline loader, a direct port of RaTrack's, kept so its
                    published setup can be reproduced exactly.

Split definitions live in `clips/` and are resolved relative to this package, so
loading does not depend on the process working directory. Point a loader at a
different set with `args.clips_dir`.

Loading itself runs on CPU inside DataLoader workers. That is deliberate and
measurably faster on this data; see `loader_vod` for the benchmarks and for
`build_dataloader`, which configures the host->device hand-off (pinned memory,
async copies).
"""
from .datagen_vod import TrackingDataVOD
from .labels_vod import filter_moving_boxes_det
from .loader_vod import build_dataloader, to_device

__all__ = [
    "TrackingDataVOD",
    "filter_moving_boxes_det",
    "build_dataloader",
    "to_device",
]
