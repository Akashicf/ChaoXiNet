# hash:sha256:3bd63b0bfdf010537d29f6020a53cbd776324366e9eda5e85e5472ca23422771
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx=21.2.6-0ubuntu0.1~20.04.2 \
        libglib2.0-0=2.64.6-1~ubuntu20.04.4 \
        libsm6=2:1.2.3-1 \
        libxext6=2:1.3.4-0ubuntu1 \
        libxrender-dev=1:0.9.10-1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    einops==0.4.1 \
    numpy==1.23.2 \
    opencv-python==4.6.0.66 \
    scikit-image==0.19.3 \
    tensorflow==2.9.1 \
    torch==1.12.1 \
    torchvision==0.13.1

COPY postInstall /
RUN /postInstall
