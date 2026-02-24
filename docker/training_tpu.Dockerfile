FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.4.0_3.10_tpuvm

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
    && grep -v -E '^(torch|torchaudio|lightning|triton)=' /app/edmformer_requirements.txt > /app/edmformer_requirements_tpu.txt \
    && python -m pip install -r /app/edmformer_requirements_tpu.txt

COPY . /app

CMD ["python", "src/task.py", "--help"]
