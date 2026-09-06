"""Central path/config for the RaTrack metric-recovery example.

Edit these to match your machine. Defaults point at the locations found on the
development box; the dataset path mirrors the volume mount in
`docker/docker-compose.yml`.
"""
import os

# --- RaTrack per-frame predictions (the .txt cluster files it writes to ./results) ---
# Full set produced by RaTrack's `python main.py --config configs_eval.yaml`.
RATRACK_RESULT_DIR = os.environ.get(
    "RATRACK_RESULT_DIR",
    "/home/williamramirez/multimodal_reid/RaTrack/src/result/4dmot_runthis",
)

# A tiny in-repo sample (clip delft_10, 34 frames) so the parser/metrics can be
# smoke-tested without the full RaTrack output tree.
SAMPLE_PREDICTION_DIR = os.path.join(os.path.dirname(__file__), "predictions_sample")

# --- View-of-Delft dataset root (same folder mounted into the container) ---
VOD_ROOT = os.environ.get(
    "VOD_ROOT",
    "/local/william/tesis/datasets/multimodal/view_of_delft_PUBLIC",
)

# GT tracking labels (KITTI-style rows + a per-object track id in column 2).
GT_TRACKING_LABEL_DIR = os.path.join(VOD_ROOT, "lidar", "training", "label_2_tracking")
RADAR_CALIB_DIR = os.path.join(VOD_ROOT, "radar", "training", "calib")
RADAR_VELODYNE_DIR = os.path.join(VOD_ROOT, "radar", "training", "velodyne")

# --- RaTrack source tree (reused for coordinate transforms / moving filter) ---
# Reused so the box->radar transform and moving-object filter match RaTrack
# exactly instead of being re-derived (and possibly diverging).
RATRACK_SRC = os.environ.get(
    "RATRACK_SRC", "/home/williamramirez/multimodal_reid/RaTrack/src"
)

# --- Evaluation split ---
# These four clips are RaTrack's validation split (the effective eval set, since
# the VoD test split has no public labels). Frame-index lists live in ./clips.
VAL_CLIPS = ["delft_1", "delft_10", "delft_14", "delft_22"]

# Point-based IoU threshold for a prediction<->GT match (RaTrack uses 0.25).
POINT_IOU_THRESHOLD = 0.25

# Number of recall sample points for the integral metrics (AB3DMOT uses 41 -> 40 steps).
NUM_SAMPLE_PTS = 41
