FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY third_party/EDMFormer/requirements.txt /app/edmformer_requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/requirements.txt \
    && python -m pip install -r /app/edmformer_requirements.txt

COPY . /app

# Overlay a small patch to avoid multi-GPU crash without modifying the submodule.
RUN python /app/docker/patches/patch_edmformer_train.py

CMD ["python", "src/task.py", "--help"]
