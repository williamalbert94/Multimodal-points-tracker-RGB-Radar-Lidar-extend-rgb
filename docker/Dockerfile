# Usa la imagen base de CUDA 11.8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Establece las variables de entorno
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV CONDA_DIR /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Instala dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    git \
    build-essential \
    libgl1 \
    libcurl4 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# libcurl4: the mongod binary bundled by fiftyone-db links against
# libcurl.so.4, which the CUDA base image does not ship. Without it the
# embedded database fails to start and `import fiftyone` aborts.

# Instala Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR && \
    rm Miniconda3-latest-Linux-x86_64.sh

COPY requirements/environment.yml /project/environment.yml
COPY . /project

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
# Actualiza Conda y crea un entorno
RUN conda update conda -y && \
   conda clean --all -y && \
   conda env create -f /project/environment.yml
   
# Activa el entorno y instala PyTorch 2.2.0
RUN conda run -n mira conda install pytorch=2.2.0 cudatoolkit=11.8 -c pytorch -y
RUN echo "source activate" >> ~/.bashrc
RUN echo "conda activate mira" >> ~/.bashrc

# Dataset curation and feature analysis (examples/feature_analysis).
# FiftyOne (Voxel51) needs mongodb-community bundled via fiftyone-db; umap-learn
# powers the embedding projections. Installed with pip inside the conda env
# because they are not on the conda channels used above.
# pymongo is pinned below 4.9 on purpose: fiftyone 0.23.8 declares only
# `pymongo>=3.12` but still imports `pymongo.database._check_name`, which 4.9
# removed. Without the pin pip resolves to 4.17 and `import fiftyone` dies with
# "cannot import name '_check_name'".
RUN conda run -n mira pip install --no-cache-dir \
        "pymongo==4.8.0" \
        fiftyone==0.23.8 \
        fiftyone-db==1.1.5 \
        umap-learn==0.5.5 \
        scikit-learn==1.4.1.post1 \
        pandas==2.2.1

RUN echo "cd /project/external/lib/" >> ~/.bashrc
RUN echo "python setup.py install" >> ~/.bashrc
RUN echo "cd /project/" >> ~/.bashrc



ENTRYPOINT ["conda", "run", "-n", "mira"]
WORKDIR /project/



