# Installation

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9.18-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-11-8-0-download-archive)
[![Conda](https://img.shields.io/badge/Conda-environment.yml-44A833.svg?logo=anaconda&logoColor=white)](../../requirements/environment.yml)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker&logoColor=white)](../../docker/docker-compose.yml)
[![MLflow](https://img.shields.io/badge/MLflow-2.14.1-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![MkDocs](https://img.shields.io/badge/MkDocs-Material-526CFE.svg?logo=materialformkdocs&logoColor=white)](https://squidfunk.github.io/mkdocs-material/)

</div>

Two supported paths: **Docker** (recommended, reproducible) or **native/conda**
(no Docker). Both end up with the same `mira` conda environment and the same
compiled PointNet++ CUDA ops; pick whichever fits your setup.

## Table of Contents

- [Prerequisites](#prerequisites)
- [1. Clone the repository](#1-clone-the-repository)
- [2. Get the View-of-Delft dataset](#2-get-the-view-of-delft-dataset)
- [3a. Option A: Docker](#3a-option-a-docker)
- [3b. Option B: Native (conda, no Docker)](#3b-option-b-native-conda-no-docker)
- [4. Optional flag: MLflow + MkDocs](#4-optional-flag-mlflow--mkdocs)
- [5. Key package versions](#5-key-package-versions)
- [6. Troubleshooting](#6-troubleshooting)

## Prerequisites

| Requirement | Why | Docker path | Native path |
|---|---|:---:|:---:|
| Linux (Ubuntu 20.04+) | matches the CUDA 11.8 base image / tested environment | ✅ | ✅ |
| NVIDIA GPU + driver ≥ 470 | CUDA 11.8 runtime support | ✅ | ✅ |
| Git | clone the repository | ✅ | ✅ |
| ~15 GB free disk (code + env), dataset is separate (~15-30 GB) | | ✅ | ✅ |
| [Docker Engine](https://docs.docker.com/engine/install/) + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) | build/run the container with GPU access | ✅ | — |
| [Miniconda](https://docs.anaconda.com/miniconda/) or Anaconda | create the `mira` environment | — | ✅ |
| CUDA 11.8 **Toolkit** (`nvcc`) installed system-wide | compiles the PointNet++ CUDA ops (`external/lib`) at first run | — | ✅ |

> The native path needs the full CUDA **Toolkit** (`nvcc` on `PATH`), not just
> a driver. The Docker path gets `nvcc` for free from the
> `nvidia/cuda:11.8.0-cudnn8-devel` base image.

## 1. Clone the repository

```bash
git clone <this-repository-url>
cd Multimodal-points-tracker-RGB-Radar-Lidar-VOD
```

## 2. Get the View-of-Delft dataset

Follow the root [README's Dataset section](../../README.md#dataset) to
request access to VoD and the tracking annotations, and lay them out under
the expected `view_of_delft_PUBLIC/` structure documented there. You will
point both installation paths at wherever you place this folder.

## 3a. Option A: Docker

1. Install Docker Engine and the NVIDIA Container Toolkit, then confirm GPU
   access works:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
   ```
2. Edit [`docker/docker-compose.yml`](../../docker/docker-compose.yml) and
   point the dataset volume at your local VoD path:
   ```yaml
   volumes:
     - /path/to/your/view_of_delft_PUBLIC:/project/view_of_delft_PUBLIC
   ```
3. Build the image (context is the repository root):
   ```bash
   cd docker
   docker compose build
   ```
4. Start the container:
   ```bash
   docker compose run tracker_mra
   ```
   This drops you into a bash shell inside `/project` with the `mira` conda
   environment already active (`.bashrc` runs `conda activate mira`). On the
   **first** shell of each container instance it also compiles the
   PointNet++ CUDA extension under `external/lib` (~1-2 min); this is normal.
5. Verify:
   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   # expected: 2.2.0 True
   ```

## 3b. Option B: Native (conda, no Docker)

1. Install [Miniconda](https://docs.anaconda.com/miniconda/) and the
   **CUDA 11.8 Toolkit** for your distro, then confirm the compiler is on
   `PATH`:
   ```bash
   nvcc --version   # should report release 11.8
   ```
2. From the repository root, create and activate the environment:
   ```bash
   conda env create -f requirements/environment.yml
   conda activate mira
   ```
3. Pin the CUDA-matched PyTorch build (mirrors what the Docker image does
   explicitly, in case the solver picked a different build):
   ```bash
   conda install pytorch=2.2.0 cudatoolkit=11.8 -c pytorch -y
   ```
4. Compile and install the PointNet++ CUDA ops:
   ```bash
   cd external/lib
   python setup.py install
   cd ../..
   ```
5. Verify:
   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   # expected: 2.2.0 True
   ```

## 4. Optional flag: MLflow + MkDocs

Neither is required to train/evaluate models. Two ways to get them, depending
on which installation path you used:

**Docker path — run them as compose services (recommended).**
[`docker/docker-compose.yml`](../../docker/docker-compose.yml) defines two
extra services on top of `tracker_mra`:

| Service | Role | Lifecycle | Port |
|---|---|---|---|
| `mlflow` | experiment tracking server (logs metrics/params/checkpoints) | **background**, starts automatically with `docker compose up` and keeps running (`restart: unless-stopped`) | [localhost:5000](http://localhost:5000) |
| `mkdocs` | serves this `docs/` folder as a browsable site | **on-demand microservice**, gated behind the `docs` compose profile so a plain `docker compose up` does *not* start it | [localhost:8000](http://localhost:8000) |

```bash
cd docker

# tracker_mra + mlflow come up together
docker compose up -d

# mkdocs only when you actually want to browse the docs site
docker compose --profile docs up mkdocs
# (naming the service explicitly also works without the --profile flag:
#  docker compose up mkdocs)
```

`tracker_mra` already gets `MLFLOW_TRACKING_URI=http://mlflow:5000` injected,
so training/eval code inside the container can call `mlflow.log_metric(...)`
without any extra configuration and see it at `http://localhost:5000` on the
host. MLflow run data persists on the host under `./mlruns` (mounted from
`docker/../mlruns`, i.e. the repo root).

**Native path — install as local CLI tools.** With the `mira` environment
active:

```bash
pip install -r requirements/extras.txt
```

| Tool | Gives you | Try it |
|---|---|---|
| [MLflow](https://mlflow.org/) | local tracking server | `mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri ./mlruns` |
| [MkDocs](https://www.mkdocs.org/) + [Material](https://squidfunk.github.io/mkdocs-material/) | renders `docs/` as a site, using the [`mkdocs.yml`](../../mkdocs.yml) at the repo root | `mkdocs serve` (from the repo root) |

## 5. Key package versions

Pinned in [`requirements/environment.yml`](../../requirements/environment.yml)
(conda solve) and [`requirements/extras.txt`](../../requirements/extras.txt)
(optional, pip):

| Package | Version | Role |
|---|---|---|
| Python | 3.9.18 | interpreter |
| PyTorch | 2.2.0 (`py3.9_cuda11.8_cudnn8.7.0`) | deep learning framework |
| CUDA runtime | 11.8 | GPU compute |
| NumPy | 1.26.3 | array backend |
| Open3D | 0.18.0 | point cloud I/O / geometry |
| OpenCV (`opencv-python`) | 4.9.0.80 | image processing |
| SciPy | 1.12.0 | scientific computing |
| scikit-learn | 1.4.1.post1 | classical ML utilities |
| Matplotlib | 3.8.3 | static plotting |
| Plotly / Dash | 5.19.0 / 2.15.0 | interactive visualization |
| pandas | 2.2.1 | tabular data |
| Numba | 0.59.0 | JIT-accelerated routines |
| pyquaternion | 0.9.9 | rotation math |
| **MLflow** *(extra)* | 2.14.1 | experiment tracking |
| **MkDocs** *(extra)* | 1.6.0 | documentation site generator |
| **mkdocs-material** *(extra)* | 9.5.28 | documentation site theme |

## 6. Troubleshooting

- **`nvcc: command not found`** (native path): the CUDA Toolkit is not
  installed or not on `PATH`. The PointNet++ ops in `external/lib` cannot be
  compiled without it; a driver alone is not enough.
- **`docker: Error response from daemon: could not select device driver`**:
  the NVIDIA Container Toolkit is missing or not configured; reinstall it and
  restart the Docker daemon.
- **`torch.cuda.is_available()` returns `False`**: check `nvidia-smi` works
  on the host first; if it does, confirm the container/environment actually
  resolved a `cu118` PyTorch build (step 3 above) rather than a CPU-only one.
- **Dataset paths not found at runtime**: confirm the volume/path from
  [step 2](#2-get-the-view-of-delft-dataset) matches the structure in the
  root [README](../../README.md#expected-dataset-structure) exactly.
