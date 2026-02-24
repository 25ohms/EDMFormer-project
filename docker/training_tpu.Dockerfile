FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.4.0_3.10_tpuvm
ARG XLA_VERSION=2.4.0

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
    && grep -v -i -E '^[[:space:]]*(torch|torchvision|torchaudio|torch_xla|lightning|triton)([=<>!~].*)?$' /app/edmformer_requirements.txt > /app/edmformer_requirements_tpu.txt \
    && python -m pip install -r /app/edmformer_requirements_tpu.txt \
    && python -m pip install --no-cache-dir --upgrade --force-reinstall \
        "torch==${XLA_VERSION}" \
        "torch_xla==${XLA_VERSION}" \
        -f https://storage.googleapis.com/libtpu-releases/index.html

COPY . /app

CMD ["python", "src/task.py", "--help"]
