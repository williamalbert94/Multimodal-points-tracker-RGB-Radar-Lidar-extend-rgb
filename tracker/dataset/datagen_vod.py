"""View-of-Delft tracking dataloader (production version).

Extends the RaTrack-derived `track_vod_3d.TrackingDataVOD` with point-cloud
augmentation, per-clip frame indexing and box-motion features.
"""
import os
import os.path
import struct
from datetime import time

import numpy as np
from torch.utils.data import Dataset

# Split lists ship with the package, so the loader does not depend on the
# process working directory.
CLIPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips")

from external.kitti.kitti_calib import Calibration
from external.vod.frame.transformations import homogeneous_transformation
from external.kitti.kitti_trk_vod import Tracklet_3D
from external.kitti.kitti_oxts import load_oxts

from external.vod.configuration import VodTrackLocations
from external.vod.frame import FrameDataLoader, FrameTransformMatrix

# from kitti.kitti_oxts import

import matplotlib
# matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
from .labels_vod import filter_moving_boxes_det
from .fusion_vod import fuse as fuse_sensors, feature_dim as fusion_feature_dim
from ..utils.motion_utils import compute_box_motion_features
# Load: raw + label + ego


def augment_point_cloud(pc, rotation_range=45, jitter_std=0.01, scaling_range=(0.9, 1.1), dropout_ratio=0.1):
    """
    Apply data augmentation to point cloud.

    Args:
        pc: [N, 3] point cloud
        rotation_range: max rotation angle in degrees
        jitter_std: gaussian jitter standard deviation
        scaling_range: (min_scale, max_scale) for random scaling
        dropout_ratio: ratio of points to randomly drop

    Returns:
        augmented_pc: [N, 3] augmented point cloud
    """
    aug_pc = pc.copy()

    # Random rotation around Z-axis (since most objects are tracked in XY plane)
    if rotation_range > 0:
        angle = np.random.uniform(-rotation_range, rotation_range) * np.pi / 180.0
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        aug_pc = aug_pc @ rotation_matrix.T

    # Random jittering
    if jitter_std > 0:
        jitter = np.random.normal(0, jitter_std, aug_pc.shape)
        aug_pc = aug_pc + jitter

    # Random scaling
    if scaling_range is not None and scaling_range[0] < scaling_range[1]:
        scale = np.random.uniform(scaling_range[0], scaling_range[1])
        aug_pc = aug_pc * scale

    # Random point dropout
    if dropout_ratio > 0:
        num_drop = int(len(aug_pc) * dropout_ratio)
        if num_drop > 0:
            keep_indices = np.random.choice(len(aug_pc), len(aug_pc) - num_drop, replace=False)
            aug_pc = aug_pc[keep_indices]
            # Pad back to original size if needed (sampling mode)
            if len(aug_pc) < len(pc):
                num_pad = len(pc) - len(aug_pc)
                pad_indices = np.random.choice(len(aug_pc), num_pad, replace=True)
                aug_pc = np.vstack([aug_pc, aug_pc[pad_indices]])

    return aug_pc

class TrackingDataVOD(Dataset):

    def __init__(self, args, data_dir):
        self.eval = args.eval
        self.dataset_path = args.dataset_path
        # Data augmentation parameters
        self.aug = getattr(args, 'aug', False)
        if self.aug:
            aug_config = getattr(args, 'augmentation', {})
            self.rotation_range = aug_config.get('rotation_range', 45) if isinstance(aug_config, dict) else 45
            self.jitter_std = aug_config.get('jitter_std', 0.01) if isinstance(aug_config, dict) else 0.01
            self.scaling_range = aug_config.get('scaling_range', [0.9, 1.1]) if isinstance(aug_config, dict) else [0.9, 1.1]
            self.dropout_ratio = aug_config.get('dropout_ratio', 0.1) if isinstance(aug_config, dict) else 0.1

        # Static object handling
        self.static_object_handling = getattr(args, 'static_object_handling', 'filter')

        # ── Early LiDAR-radar fusion ────────────────────────────────────────
        # Off by default, so existing configs keep the radar-only behaviour
        # byte for byte. See fusion_vod.py for the modes and why more than one
        # exists (the naive LiDAR-base association leaves ~95% of points with
        # null radar attributes at this sensor density).
        self.fusion = getattr(args, 'fusion', 'none')
        self.fusion_radius = float(getattr(args, 'fusion_radius', 0.5))
        # Cap for the LiDAR-based modes; without it they return ~178k points.
        self.fusion_max_points = int(getattr(args, 'fusion_max_points',
                                             max(4 * getattr(args, 'num_points', 256), 4096)))
        self.feature_dim = fusion_feature_dim(self.fusion)
        if self.fusion != 'none':
            print(f"[fusion] mode={self.fusion} radius={self.fusion_radius}m "
                  f"max_points={self.fusion_max_points} -> {self.feature_dim} feature channels")
            if self.feature_dim != 2:
                print(f"[fusion] NOTE: build the extractor with in_channels="
                      f"{self.feature_dim} (default is 2)")

        # set params
        self.dir = data_dir

        test = ['delft_7','delft_8','delft_16','delft_18','delft_20','delft_21','delft_25']
        val = ['delft_1','delft_10','delft_14','delft_22']
        train = ['delft_2','delft_3','delft_4','delft_6','delft_9','delft_11','delft_12','delft_13','delft_19','delft_23','delft_24','delft_26','delft_27']
        # Package-relative so the loader works regardless of the working
        # directory; override with args.clips_dir if the split lists live
        # elsewhere.
        self.clips_dir = getattr(args, 'clips_dir', None) or CLIPS_DIR

        # Optional: merge val into train (for capacity experiments where the
        # split distribution shift is the bottleneck). Toggled by args.merge_val_into_train.
        merge_val = getattr(args, 'merge_val_into_train', False)
        if self.eval:
            self.clips = val
        elif merge_val:
            self.clips = train + val
        else:
            self.clips = train

        # Precalculate all valid frame indices
        # First, collect all available frames to check for gaps
        all_available_frames = set()
        frames_by_clip = {}
        for clip_idx, clip in enumerate(self.clips):
            txt_path = os.path.join(self.clips_dir, clip + '.txt')
            with open(txt_path) as f:
                frames = [int(line.strip()) for line in f.readlines()]
            frames_by_clip[clip_idx] = frames
            all_available_frames.update(frames)

        # Build frame list, excluding frames where frame+1 or frame-1 don't exist
        self.frame_list = []  # List of tuples: (clip_idx, frame_number, is_first_frame)
        total_frames = 0
        excluded_frames = 0

        for clip_idx, clip in enumerate(self.clips):
            frames = frames_by_clip[clip_idx]
            for i, frame_num in enumerate(frames):
                # Check if current_frame+1 and current_frame-1 exist (needed for loading)
                # We load: frame_data_0 (frame+1), frame_data_1 (frame), frame_data_last (frame-1)
                if (frame_num + 1) in all_available_frames and (frame_num - 1) in all_available_frames:
                    is_first = (i == 0)
                    self.frame_list.append((clip_idx, frame_num, is_first))
                    total_frames += 1
                else:
                    excluded_frames += 1

        print(f"Loaded {total_frames} valid frames from {len(self.clips)} clips")
        print(f"Excluded {excluded_frames} frames due to missing adjacent frames")


    def __getitem__(self, index):
        # Use while loop to keep trying until we find a valid frame
        attempts = 0
        max_attempts = 100
        original_index = index

        while attempts < max_attempts:
            try:
                # Get frame info from precalculated list
                clip_idx, current_frame, new_seq = self.frame_list[index]
                clip_name = self.clips[clip_idx]

                kitti_locations = VodTrackLocations(root_dir=self.dataset_path,
                                                output_dir=self.dataset_path,
                                                frame_set_path="",
                                                pred_dir="",
                                                )

                frame_data_0 = FrameDataLoader(kitti_locations=kitti_locations,
                                            frame_number=str(current_frame+1).zfill(5))
                frame_data_1 = FrameDataLoader(kitti_locations=kitti_locations,
                                            frame_number=str(current_frame).zfill(5))
                frame_data_last = FrameDataLoader(kitti_locations=kitti_locations,
                                            frame_number=str(current_frame-1).zfill(5))

                raw_pc0 = frame_data_0.radar_data[:, :3]
                raw_pc1 = frame_data_1.radar_data[:, :3]

                features0 = frame_data_0.radar_data[:, 3:6]
                features1 = frame_data_1.radar_data[:, 3:6]

                transforms0 = FrameTransformMatrix(frame_data_0)
                transforms1 = FrameTransformMatrix(frame_data_1)
                transforms_last = FrameTransformMatrix(frame_data_last)

                raw_pc_last_lidar = frame_data_last.lidar_data[:, :3]
                raw_pc0_lidar = frame_data_0.lidar_data[:, :3]
                raw_pc1_lidar = frame_data_1.lidar_data[:, :3]

                n0_ = raw_pc_last_lidar.shape[0]
                pts_3d_hom0_ = np.hstack((raw_pc_last_lidar, np.ones((n0_, 1))))
                raw_pc_last_lidar = homogeneous_transformation(pts_3d_hom0_, transforms_last.t_lidar_radar)

                n1_ = raw_pc0_lidar.shape[0]
                pts_3d_hom1_ = np.hstack((raw_pc0_lidar, np.ones((n1_, 1))))
                raw_pc0_lidar = homogeneous_transformation(pts_3d_hom1_, transforms0.t_lidar_radar)

                n2_ = raw_pc1_lidar.shape[0]
                pts_3d_hom2_ = np.hstack((raw_pc1_lidar, np.ones((n2_, 1))))
                raw_pc1_lidar = homogeneous_transformation(pts_3d_hom2_, transforms1.t_lidar_radar)

                odom_cam_0 = transforms0.t_odom_camera
                odom_cam_1 = transforms1.t_odom_camera
                cam_radar_0 = transforms0.t_camera_radar
                cam_radar_1 = transforms1.t_camera_radar
                odom_radar_0 = np.dot(odom_cam_0,cam_radar_0)
                odom_radar_2 = np.dot(odom_cam_1,cam_radar_1)
                ego_motion = np.dot(np.linalg.inv(odom_radar_0), odom_radar_2)

                comp_hom = np.hstack((raw_pc0, np.ones((raw_pc0.shape[0], 1))))
                raw_pc0_comp = np.dot(comp_hom, np.linalg.inv(ego_motion.T))

                # ── Early fusion ────────────────────────────────────────────
                # Runs after the extrinsic (LiDAR is already in the radar frame)
                # and after ego-motion compensation, so association happens in a
                # single consistent frame. The compensated cloud is rebuilt from
                # the fused points with the same T_ego, keeping the pair aligned.
                if self.fusion != 'none':
                    rng = np.random.default_rng(current_frame)   # reproducible
                    raw_pc0, features0 = fuse_sensors(
                        raw_pc0, features0, raw_pc0_lidar[:, :3],
                        mode=self.fusion, radius=self.fusion_radius,
                        max_points=self.fusion_max_points, rng=rng)
                    raw_pc1, features1 = fuse_sensors(
                        raw_pc1, features1, raw_pc1_lidar[:, :3],
                        mode=self.fusion, radius=self.fusion_radius,
                        max_points=self.fusion_max_points, rng=rng)
                    comp_hom = np.hstack((raw_pc0, np.ones((raw_pc0.shape[0], 1))))
                    raw_pc0_comp = np.dot(comp_hom, np.linalg.inv(ego_motion.T))

                curr_idx = current_frame + 1

                labels1 = load_labels(frame_data_0.raw_tracking_labels, index + 1)
                labels2 = load_labels(frame_data_1.raw_tracking_labels, index)

                transforms1 = FrameTransformMatrix(frame_data_0)
                transforms2 = FrameTransformMatrix(frame_data_1)

                lbl1 = labels1.data[index + 1]
                lbl2 = labels2.data[index]

                # Apply movement filter based on static_object_handling flag
                if self.static_object_handling == 'filter':
                    # Default behavior: only moving objects
                    lbl1_mov = filter_moving_boxes_det(frame_data_0.raw_detection_labels, lbl1)
                    lbl2_mov = filter_moving_boxes_det(frame_data_1.raw_detection_labels, lbl2)
                    lbl1 = lbl1_mov
                    lbl2 = lbl2_mov
                elif self.static_object_handling == 'include_all':
                    # Keep all GT boxes (moving + static)
                    # lbl1 and lbl2 already have all boxes, no filtering needed
                    pass
                elif self.static_object_handling == 'hybrid':
                    # Filter for GT matching, but HybridBoxProposal will cluster static objects
                    lbl1_mov = filter_moving_boxes_det(frame_data_0.raw_detection_labels, lbl1)
                    lbl2_mov = filter_moving_boxes_det(frame_data_1.raw_detection_labels, lbl2)
                    lbl1 = lbl1_mov
                    lbl2 = lbl2_mov
                else:
                    # Unknown mode, default to filter
                    lbl1_mov = filter_moving_boxes_det(frame_data_0.raw_detection_labels, lbl1)
                    lbl2_mov = filter_moving_boxes_det(frame_data_1.raw_detection_labels, lbl2)
                    lbl1 = lbl1_mov
                    lbl2 = lbl2_mov

                # Apply data augmentation (only during training)
                if self.aug and not self.eval:
                    raw_pc0 = augment_point_cloud(raw_pc0,
                                                   rotation_range=self.rotation_range,
                                                   jitter_std=self.jitter_std,
                                                   scaling_range=tuple(self.scaling_range),
                                                   dropout_ratio=self.dropout_ratio)
                    raw_pc1 = augment_point_cloud(raw_pc1,
                                                   rotation_range=self.rotation_range,
                                                   jitter_std=self.jitter_std,
                                                   scaling_range=tuple(self.scaling_range),
                                                   dropout_ratio=self.dropout_ratio)
                    # OJO: la nube compensada sale con 4 columnas (la 4a es el 1
                    # homogeneo del np.dot de arriba). augment_point_cloud rota con
                    # una matriz 3x3, asi que hay que pasarle solo xyz; si no,
                    # (N,4)@(3,3) revienta con "matmul mismatch". Aguas abajo la nube
                    # compensada se recorta a [:3] igual, por eso quedarnos con xyz
                    # aca no pierde nada.
                    raw_pc0_comp = augment_point_cloud(raw_pc0_comp[:, :3],
                                                        rotation_range=self.rotation_range,
                                                        jitter_std=self.jitter_std,
                                                        scaling_range=tuple(self.scaling_range),
                                                        dropout_ratio=self.dropout_ratio)

                # Successfully loaded all data

                # ===== COMPUTE MOTION FEATURES (NUEVO) =====
                # Calcula dirección, distancia, velocidad entre lbl1 y lbl2
                try:
                    motion_features = compute_box_motion_features(lbl1, lbl2, dt=0.1)
                except Exception as e:
                    # Si falla cálculo de motion, usar dict vacío
                    motion_features = {}

                # Retornar con motion features
                return raw_pc0, raw_pc1, features0, features1, raw_pc0_comp, curr_idx, clip_name, ego_motion, raw_pc_last_lidar, raw_pc0_lidar, raw_pc1_lidar, new_seq, lbl1, lbl2, transforms1, transforms2, motion_features

            except (ImportError, NameError) as e:
                # Genuine programming errors: retrying the next frame would fail
                # identically 100 times and surface as a misleading "failed to
                # load valid frame" message. Fail loudly instead.
                #
                # NOTE: TypeError/AttributeError are deliberately NOT listed
                # here. The VoD devkit returns `None` for missing files, so a
                # missing frame shows up as "'NoneType' is not subscriptable" —
                # a data problem that must be skipped, not a bug.
                raise RuntimeError(
                    f"Bug in the loader (not a data problem) at frame index "
                    f"{index}: {type(e).__name__}: {e}") from e

            except Exception as e:
                # Genuine data problems (missing file, malformed frame): skip on.
                attempts += 1
                index = (index + 1) % len(self.frame_list)
                if attempts % 10 == 0:
                    print(f"Warning: Failed to load frame at index {original_index}, "
                          f"tried {attempts} frames. {type(e).__name__}: {str(e)[:100]}")

        # If we get here, we failed to load any frame after many attempts
        raise RuntimeError(f"Failed to load valid frame after {max_attempts} attempts starting from index {original_index}")

    def __len__(self):
        return len(self.frame_list)


def load_poses(oxts_path, seq):
    file_path = os.path.join(oxts_path, str(seq).zfill(4) + '.txt')
    oxts = load_oxts(file_path)
    return oxts


def load_labels(labels, frame):
    labels_trk = Tracklet_3D(labels, frame)
    return labels_trk


def load_calib(calib_path, seq):
    file_path = os.path.join(calib_path, str(seq).zfill(4) + '.txt')
    calib = Calibration(file_path)
    return calib


def load_raw_pc(velodyne_path, seq):
    seq_path = os.path.join(velodyne_path, str(seq).zfill(4))
    _, _, files = next(os.walk(seq_path))
    file_count = len(files)
    raw_pc = []

    for i in range(file_count):
        file_path = os.path.join(seq_path, str(i).zfill(6) + '.bin')

        point_cloud_data = np.fromfile(file_path, '<f4')  # little-endian float32
        point_cloud_data = np.reshape(point_cloud_data, (-1, 4))  # x, y, z, r

        raw_pc.append(point_cloud_data)

    return raw_pc


def load_raw_pc_frame(velodyne_path, frame):
    # seq_path = os.path.join(velodyne_path, str(seq).zfill(4))
    file_path = os.path.join(velodyne_path, str(frame).zfill(5) + '.bin')

    raw_pc = np.fromfile(file_path, '<f4')  # little-endian float32
    raw_pc = np.reshape(raw_pc, (-1, 4))  # x, y, z, r

    return raw_pc