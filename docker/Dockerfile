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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Instala Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR && \
    rm Miniconda3-latest-Linux-x86_64.sh

COPY ./environment.yml /project/
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

RUN echo "cd /project/external/lib/" >> ~/.bashrc
RUN echo "python setup.py install" >> ~/.bashrc
RUN echo "cd /project/" >> ~/.bashrc



ENTRYPOINT ["conda", "run", "-n", "pytorch_env"]
WORKDIR /project/



