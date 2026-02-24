FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN sed -i 's|http://|https://|g' /etc/apt/sources.list /etc/apt/sources.list.d/*.list 2>/dev/null || true \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        gnupg \
        ubuntu-keyring \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
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

RUN python - <<'PY'
import importlib.metadata as md
from pathlib import Path

lines = []
try:
    import torch
    lines.append(f"torch=={torch.__version__}")
except Exception:
    pass
for pkg in ("torchaudio", "torchvision"):
    try:
        lines.append(f"{pkg}=={md.version(pkg)}")
    except Exception:
        pass
Path("/app/torch_constraints.txt").write_text("\n".join(lines))
PY

RUN python -m pip install --upgrade pip \
    && grep -v -i -E '^[[:space:]]*(torch|torchaudio|torchvision)([=<>!~].*)?$' /app/requirements_pre.txt > /app/requirements_pre_filtered.txt \
    && python -m pip install -r /app/requirements_pre_filtered.txt -c /app/torch_constraints.txt

# torchcodec for audio/video decode support (aligns with installed torch)
RUN python -m pip install torchcodec

# MusicFM is not pip-packaged; clone source and add to PYTHONPATH
ARG MUSICFM_REPO=https://github.com/minzwon/musicfm.git
ARG MUSICFM_REF=master
RUN git clone ${MUSICFM_REPO} /app/musicfm \
    && cd /app/musicfm \
    && git checkout ${MUSICFM_REF}

ENV PYTHONPATH=/app

CMD ["python", "preprocessing/extract_muq.py", "--help"]
