FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04 AS base

RUN rm /etc/apt/sources.list.d/cuda.list

RUN apt-get update && \
  apt-get install -y software-properties-common && \
  add-apt-repository ppa:deadsnakes/ppa && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git \
  wget \
  unzip \
  libopenblas-dev \
  python3.10 \
  python3.10-dev \
  python3.10-distutils \
  python3.10-venv \
  nano \
  && \
  apt-get clean autoclean && \
  apt-get autoremove -y && \
  rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py && python3.10 get-pip.py && rm get-pip.py  


# Upgrade pip
RUN python3.10 -m pip install --no-cache-dir --upgrade pip
COPY process_segformer3d/requirements_segformer_limited.txt /tmp/requirements.txt
RUN python3.10 -m pip install --no-cache-dir -r /tmp/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# Install a few dependencies that are not automatically installed
RUN pip3 install \
        graphviz \
        onnx \
        SimpleITK && \
    rm -rf ~/.cache/pip

### USER
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

COPY --chown=user:user process_segformer3d/process_segformer.py /opt/app/
COPY --chown=user:user ./export2onnx.py /opt/app/
COPY --chown=user:user process_segformer3d/segformer3d.py /opt/app/

### ALGORITHM

# Copy model checkpoint to docker (uncomment if you put the model weights directly in this repo)
# COPY --chown=user:user all_data_checkpoint_segformer /opt/ml/model/all_data_checkpoint_segformer

# Copy container testing data to docker (uncomment if you want to see if the model works and put a test image and spacing in this repo)
# COPY --chown=user:user ./architecture/input/ /input/


ENTRYPOINT [ "python3.10", "-m", "process_segformer", "--model_dir", \ 
"all_data_checkpoint_segformer/"]
