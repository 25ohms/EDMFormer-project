FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    libsndfile1 \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavdevice-dev \
    && rm -rf /var/lib/apt/lists/*

# Refresh dynamic linker cache so torchcodec can find FFmpeg libs
RUN ldconfig

# Ensure Python/FFmpeg shared libraries are discoverable
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

WORKDIR /app

COPY preprocessing/ /app/preprocessing/
COPY preprocessing/requirements_pre.txt /app/requirements_pre.txt

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r /app/requirements_pre.txt

# torchcodec for audio/video decode support (aligns with installed torch)
RUN python3 -m pip install torchcodec

# MusicFM is not pip-packaged; clone source and add to PYTHONPATH
ARG MUSICFM_REPO=https://github.com/minzwon/musicfm.git
ARG MUSICFM_REF=master
RUN git clone ${MUSICFM_REPO} /app/musicfm \
    && cd /app/musicfm \
    && git checkout ${MUSICFM_REF}

ENV PYTHONPATH=/app

CMD ["python3", "preprocessing/extract_muq.py", "--help"]
