FROM nvidia/cuda:12.1-base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY preprocessing/ /app/preprocessing/
COPY preprocessing/requirements_pre.txt /app/requirements_pre.txt

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r /app/requirements_pre.txt

CMD ["python3", "preprocessing/extract_muq.py", "--help"]
