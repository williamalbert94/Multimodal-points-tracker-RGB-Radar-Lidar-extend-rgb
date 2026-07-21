"""Point-based IoU between RaTrack clusters and GT boxes, in the radar frame.

This is the piece RaTrack's paper describes but does not distribute (its modified
AB3DMOT eval was withheld for licensing reasons):

    "we compute the IoU by counting the number of intersected and united radar
     points between the ground truth object and the predicted one. The threshold
     for our point-based IoU is set as 0.25."

For a frame with radar cloud R (indices 0..N-1):
  * GT box g   -> A_g = radar indices inside its oriented 3D box (radar frame)
  * pred clip p-> B_p = radar indices of the cluster's points
  * point-IoU(g, p) = |A_g ∩ B_p| / |A_g ∪ B_p|

Self-contained: the camera->radar / radar->lidar transforms and the oriented box
construction are re-implemented here directly from the VoD calib files, exactly
following VoD's `FrameTransformMatrix` (calib line 5 `Tr_velo_to_cam` = the
sensor->camera 4x4) and RaTrack's `get_bbx_param`. This avoids importing RaTrack's
model stack (torch / pointnet2_cuda / k3d); only numpy + scipy + open3d are used.
"""
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

import paths


def _read_extrinsic(calib_path):
    """Return the 4x4 sensor->camera transform (calib line 5, Tr_velo_to_cam)."""
    with open(calib_path) as f:
        lines = f.readlines()
    ext = np.array(lines[5].strip().split(" ")[1:], dtype=np.float64).reshape(3, 4)
    return np.vstack([ext, [0, 0, 0, 1]])


def frame_transforms(frame_id):
    """Return (t_radar_camera, t_radar_lidar) 4x4 matrices for a frame,
    matching VoD FrameTransformMatrix conventions."""
    t_cam_radar = _read_extrinsic(os.path.join(paths.RADAR_CALIB_DIR, f"{frame_id}.txt"))
    t_cam_lidar = _read_extrinsic(
        os.path.join(paths.VOD_ROOT, "lidar", "training", "calib", f"{frame_id}.txt"))
    t_radar_camera = np.linalg.inv(t_cam_radar)
    t_radar_lidar = t_radar_camera @ t_cam_lidar      # VoD: t_radar_camera . t_camera_lidar
    return t_radar_camera, t_radar_lidar


def load_radar_cloud(frame_id):
    """VoD radar sweep -> (N,3) xyz (radar frame). Rows are 7 float32:
    [x, y, z, RCS, v_r, v_r_compensated, time]."""
    path = os.path.join(paths.RADAR_VELODYNE_DIR, f"{frame_id}.bin")
    return np.fromfile(path, dtype=np.float32).reshape(-1, 7)[:, :3].astype(np.float64)


def _gt_obb(gt, t_radar_camera, t_radar_lidar):
    """Build an open3d OrientedBoundingBox for a GT object in the radar frame,
    replicating RaTrack's get_bbx_param(sensor='radar')."""
    import open3d as o3d
    h, w, l = gt.hwl
    x, y, z = gt.center_cam
    center = (t_radar_camera @ np.array([x, y, z, 1.0]))[:3]
    extent = np.array([l, w, h])                       # RaTrack order: l, w, h
    rot = R.from_euler("XYZ", [0, 0, -(gt.ry + np.pi / 2)]).as_matrix()
    rot = t_radar_lidar[:3, :3] @ rot
    return o3d.geometry.OrientedBoundingBox(center.reshape(3, 1), rot, extent.reshape(3, 1))


def _cluster_indices(pred_points, radar_cloud):
    """Map a cluster's saved xyz back to radar-cloud indices (nearest point)."""
    idxs = set()
    for p in pred_points:
        idxs.add(int(np.argmin(np.linalg.norm(radar_cloud - p, axis=1))))
    return idxs


def merge_rider_objects(gt_objs, gt_sets, radar):
    """Merge each `rider` GT object into its nearest neighbour, as RaTrack does.

    VoD annotates a cyclist as TWO objects — the `rider` (person) and the
    `bicycle`. RaTrack's `filter_object_points` (models/utils/track4d_utils.py)
    merges them: for every object of type `rider` it finds the nearest other
    object by centroid and unions their point sets.

    This matters for evaluation, not just training. DBSCAN yields a *single*
    cluster for the whole cyclist, so scoring it against two separate GT objects
    forces one to be a TP and the other a FN. Replicating the merge raises
    matched-GT recall from 54% to 68% on the validation split (1019 riders
    merged) and reduces IDSW from 404 to 389.

    Returns (kept_objs, kept_sets).
    """
    def centroid(idxs):
        return radar[list(idxs)].mean(axis=0) if idxs else np.zeros(3)

    sets = list(gt_sets)
    objs = list(gt_objs)
    absorbed = set()
    for i, g in enumerate(objs):
        if g.cls != "rider" or not sets[i]:
            continue
        ci = centroid(sets[i])
        best, best_d = None, np.inf
        for j in range(len(objs)):
            if j == i or j in absorbed or not sets[j]:
                continue
            d = np.linalg.norm(ci - centroid(sets[j]))
            if d < best_d:
                best_d, best = d, j
        if best is not None:
            sets[best] = sets[best] | sets[i]
            absorbed.add(i)

    keep = [i for i in range(len(objs)) if i not in absorbed]
    return [objs[i] for i in keep], [sets[i] for i in keep]


def point_iou_matrix(gt_objs, pred_objs, frame_id):
    """(G, P) point-based IoU matrix for one frame."""
    import open3d as o3d

    radar = load_radar_cloud(frame_id)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(radar)

    t_rc, t_rl = frame_transforms(frame_id)
    gt_sets = []
    for g in gt_objs:
        obb = _gt_obb(g, t_rc, t_rl)
        gt_sets.append(set(obb.get_point_indices_within_bounding_box(cloud.points)))
    pred_sets = [_cluster_indices(p.points, radar) for p in pred_objs]

    iou = np.zeros((len(gt_objs), len(pred_objs)))
    for gi, A in enumerate(gt_sets):
        for pi, B in enumerate(pred_sets):
            union = len(A | B)
            iou[gi, pi] = len(A & B) / union if union else 0.0
    return iou


def radar_indices_in_boxes(gt_objs, frame_id):
    """Boolean mask over the frame's radar cloud: True where a point lies in any
    of the given GT boxes. Used by seg_miou for the GT moving mask."""
    import open3d as o3d
    radar = load_radar_cloud(frame_id)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(radar)
    t_rc, t_rl = frame_transforms(frame_id)
    mask = np.zeros(radar.shape[0], dtype=bool)
    for g in gt_objs:
        obb = _gt_obb(g, t_rc, t_rl)
        for idx in obb.get_point_indices_within_bounding_box(cloud.points):
            mask[idx] = True
    return mask, radar
