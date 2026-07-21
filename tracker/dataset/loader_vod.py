"""DataLoader construction for the View-of-Delft tracking dataset.

Why the dataset itself stays on CPU
-----------------------------------
`TrackingDataVOD.__getitem__` runs inside DataLoader **worker processes**, so
CUDA work there needs a separate CUDA context per worker and breaks `fork`
(forcing the much slower `spawn` start method). It would also be slower on this
data. Measured on a VoD frame:

| operation                       |    CPU | GPU (incl. transfer) |
|---------------------------------|-------:|---------------------:|
| transform radar sweep (284 pts) | 4.1 us |             32.5 us |
| transform LiDAR sweep (178k pts)| 0.74ms |              0.88ms |

The host<->device round trip alone costs ~19 us, more than four times the entire
CPU computation for a radar sweep. And arithmetic is not the bottleneck anyway:
reading the three frames a sample needs (radar + LiDAR, ~178k LiDAR points each)
costs ~720 us against ~25 us of transforms, so **I/O dominates by ~29x**.

The useful optimisation is therefore not "move the dataset to GPU", it is making
the host->device hand-off cheap and keeping the GPU fed:

* `pin_memory=True`      — page-locked staging buffers, required for async copies
* `non_blocking=True`    — on the `.cuda()` calls in the training loop, so the
                           copy overlaps with compute (needs pinned memory)
* `persistent_workers`   — avoids re-spawning workers every epoch
* `prefetch_factor`      — keeps batches queued ahead of the GPU
"""
import torch
from torch.utils.data import DataLoader

from .datagen_vod import TrackingDataVOD


def build_dataloader(args, dataset=None, *, train=True, collate_fn=None):
    """Build a DataLoader tuned for GPU training.

    Args:
        args: config object; reads `dataset_path`, `batch_size`, `num_workers`.
        dataset: pre-built dataset; constructed from `args` when omitted.
        train: shuffle and drop the last partial batch when True.
        collate_fn: custom collate (the VoD sample is a heterogeneous tuple, so
            one is normally required).
    """
    if dataset is None:
        dataset = TrackingDataVOD(args, args.dataset_path)

    num_workers = int(getattr(args, "num_workers", 4))
    # pin_memory only helps when there is a GPU to copy to.
    pin = torch.cuda.is_available()

    kwargs = dict(
        batch_size=int(getattr(args, "batch_size", 1)),
        num_workers=num_workers,
        shuffle=bool(train and getattr(args, "shuffle", False)),
        drop_last=bool(train),
        pin_memory=pin,
        collate_fn=collate_fn,
    )
    # These two are only valid with worker processes.
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(getattr(args, "prefetch_factor", 2))

    return DataLoader(dataset, **kwargs)


def to_device(batch, device="cuda", non_blocking=True):
    """Recursively move tensors to `device` with async copies.

    `non_blocking=True` only actually overlaps transfer with compute when the
    source memory is pinned, i.e. when the DataLoader was built with
    `pin_memory=True` (see `build_dataloader`). Non-tensor entries — label
    dicts, transform matrices, clip names — are passed through untouched.
    """
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, dict):
        return {k: to_device(v, device, non_blocking) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        moved = [to_device(v, device, non_blocking) for v in batch]
        return type(batch)(moved) if isinstance(batch, tuple) else moved
    return batch
