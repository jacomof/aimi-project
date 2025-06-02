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
COPY process_nnunet/requirements_nnunet.txt /tmp/requirements.txt
RUN python3.10 -m pip install --no-cache-dir -r /tmp/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# Configure Git, clone the repository without checking out, then checkout the specific commit
RUN git config --global advice.detachedHead false && \
    git clone --no-checkout https://github.com/MIC-DKFZ/nnUNet.git /opt/algorithm/nnunet/ && \
    cd /opt/algorithm/nnunet/ && \
    git checkout 0d042347d9587b8a12b40377066b3eeb5df102bd

# Install a few dependencies that are not automatically installed
RUN pip3 install \
        -e /opt/algorithm/nnunet \
        graphviz \
        onnx \
        SimpleITK && \
    rm -rf ~/.cache/pip

### USER
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN chown -R user /opt/algorithm/

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

COPY --chown=user:user process_nnunet/process_flexible.py /opt/app/
COPY --chown=user:user ./export2onnx.py /opt/app/

### ALGORITHM

# Copy custom trainers to docker
COPY --chown=user:user ./architecture/extensions/nnunetv2/ /opt/algorithm/nnunet/nnunetv2/

# Copy model checkpoint to docker (uncomment if you put the model weights directly in this repo)
COPY --chown=user:user ./architecture/nnUNet_results/ /opt/ml/model/

# Copy container testing data to docker (uncomment if you want to see if the model works and put a test image and spacing in this repo)
COPY --chown=user:user ./architecture/input/ /input/

# Set environment variable defaults
ENV nnUNet_raw="/opt/algorithm/nnunet/nnUNet_raw" \
    nnUNet_preprocessed="/opt/algorithm/nnunet/nnUNet_preprocessed" \
    nnUNet_results="/opt/algorithm/nnunet/nnUNet_results"

ENTRYPOINT [ "python3.10", "-m", "process_flexible", "--model_dir", \ 
"Dataset001_MIX/nnUNetTrainer_ULS_800_QuarterLR__nnUNetResEncUNetMPlans__3d_fullres"]
