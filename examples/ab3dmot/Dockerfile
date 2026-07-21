# Evaluation-only image for running the AB3DMOT tracker on View-of-Delft.
# CPU-only on purpose: AB3DMOT's tracker is a 3D Kalman filter + Hungarian
# matching, so no CUDA/torch is needed. Detection is NOT done here (AB3DMOT is
# tracking-by-detection); detections are supplied as input.
FROM python:3.9-slim

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Pinned to versions that work together on py3.9 (AB3DMOT's own requirements.txt
# pins very old builds that no longer install on modern Python).
RUN pip install --no-cache-dir \
        numpy==1.23.5 \
        scipy==1.10.1 \
        numba==0.56.4 \
        llvmlite==0.39.1 \
        filterpy==1.4.5 \
        easydict==1.10 \
        opencv-python-headless==4.9.0.80 \
        pillow==9.5.0 \
        matplotlib==3.7.5 \
        open3d==0.17.0 \
        glob2==0.7 \
        PyYAML==6.0.1 \
        scikit-learn==1.3.2 \
        terminaltables==3.1.10 \
        shapely==2.0.4

WORKDIR /work
