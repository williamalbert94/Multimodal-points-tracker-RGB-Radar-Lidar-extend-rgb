"""Parsers for RaTrack predictions and View-of-Delft GT tracking labels.

Both parsers are pure Python (no torch / open3d / dataset needed) so they can be
imported and tested on their own.

RaTrack prediction row (one tracked object per line), written by RaTrack's
`main_utils.epoch()` in eval mode:

    NA 1 -1 -1 <conf> <track_id> x0 y0 z0 x1 y1 z1 ... xk yk zk

  * columns 0-3 : placeholders (class is "NA" -> RaTrack is class-agnostic)
  * column 4    : detection confidence for the cluster
  * column 5    : track id
  * columns 6+  : flattened (x, y, z) of every radar point in the cluster,
                  already expressed in the radar coordinate frame.

VoD GT tracking row (KITTI label_2 format + a track id in column 1):

    <class> <track_id> <truncation> <alpha> x1 y1 x2 y2 h w l x y z ry <score>

  * columns 4-7  : 2D image bbox (unused here)
  * columns 8-10 : h, w, l (box size, meters)
  * columns 11-13: x, y, z (box center, CAMERA frame)
  * column 14    : ry (yaw, camera frame)
"""
from dataclasses import dataclass, field
from typing import List
import os
import numpy as np


@dataclass
class PredObject:
    track_id: int
    conf: float
    points: np.ndarray  # (N, 3) radar-frame xyz


@dataclass
class GtObject:
    track_id: int
    cls: str
    hwl: np.ndarray          # (3,) h, w, l
    center_cam: np.ndarray   # (3,) x, y, z in camera frame
    ry: float


def parse_prediction_frame(path: str) -> List[PredObject]:
    objs: List[PredObject] = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 6:
                continue
            conf = float(p[4])
            tid = int(float(p[5]))
            coords = np.asarray(list(map(float, p[6:])), dtype=np.float64)
            pts = coords.reshape(-1, 3) if coords.size >= 3 else np.zeros((0, 3))
            objs.append(PredObject(track_id=tid, conf=conf, points=pts))
    return objs


# Classes RaTrack considers as trackable road users (class-agnostic at inference,
# but GT ignore-classes like "DontCare" must be dropped for a fair match).
_IGNORE_CLASSES = {"DontCare", "dontcare"}


def parse_gt_frame(path: str, drop_classes=_IGNORE_CLASSES) -> List[GtObject]:
    objs: List[GtObject] = []
    if not os.path.exists(path):
        return objs
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 15:
                continue
            cls = p[0]
            if cls in drop_classes:
                continue
            tid = int(float(p[1]))
            hwl = np.asarray([float(p[8]), float(p[9]), float(p[10])])
            center_cam = np.asarray([float(p[11]), float(p[12]), float(p[13])])
            ry = float(p[14])
            objs.append(GtObject(track_id=tid, cls=cls, hwl=hwl,
                                 center_cam=center_cam, ry=ry))
    return objs


def parse_gt_frame_moving(tracking_path: str, detection_path: str,
                          drop_classes=_IGNORE_CLASSES) -> List[GtObject]:
    """GT objects filtered to *moving* ones, replicating RaTrack's
    `filter_moving_boxes_det`.

    VoD ships two row-aligned label files per frame:
      * `label_2`          (detection)  -> column index 1 is the MOVING flag (0/1)
      * `label_2_tracking` (tracking)   -> column index 1 is the TRACK ID
    RaTrack zips them positionally and keeps rows whose moving flag == 1. We do
    the same, so the evaluation scope matches RaTrack (it only tracks moving
    objects).
    """
    if not (os.path.exists(tracking_path) and os.path.exists(detection_path)):
        return []
    with open(tracking_path) as f:
        trk_lines = [ln.split() for ln in f if ln.strip()]
    with open(detection_path) as f:
        det_lines = [ln.split() for ln in f if ln.strip()]
    if len(trk_lines) != len(det_lines):
        # misaligned -> fall back to unfiltered tracking labels
        return parse_gt_frame(tracking_path, drop_classes)

    objs: List[GtObject] = []
    for trk, det in zip(trk_lines, det_lines):
        if len(trk) < 15 or len(det) < 2:
            continue
        if trk[0] in drop_classes:
            continue
        if int(float(det[1])) != 1:      # moving flag
            continue
        objs.append(GtObject(
            track_id=int(float(trk[1])),
            cls=trk[0],
            hwl=np.asarray([float(trk[8]), float(trk[9]), float(trk[10])]),
            center_cam=np.asarray([float(trk[11]), float(trk[12]), float(trk[13])]),
            ry=float(trk[14]),
        ))
    return objs


def frame_ids_for_clip(clip_txt: str) -> List[str]:
    """Return zero-padded frame ids (strings) listed in a clip index file."""
    with open(clip_txt) as f:
        return [ln.strip() for ln in f if ln.strip()]
