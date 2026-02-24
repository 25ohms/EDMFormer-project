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

# Pin whatever torch/torch_xla versions the base image ships with.
RUN python - <<'PY'
import importlib.metadata as md
import torch
from pathlib import Path

torch_ver = torch.__version__
xla_ver = md.version("torch_xla")
Path("/app/constraints_tpu.txt").write_text(
    f"torch=={torch_ver}\n"
    f"torch_xla=={xla_ver}\n"
)
print("Pinned torch:", torch_ver)
print("Pinned torch_xla:", xla_ver)
PY

RUN python -m pip install --upgrade pip \
    && grep -v -i -E '^[[:space:]]*(torch|torchvision|torchaudio|torch_xla|lightning|triton)([=<>!~].*)?$' /app/requirements.txt > /app/requirements_tpu.txt \
    && python -m pip install -r /app/requirements_tpu.txt -c /app/constraints_tpu.txt \
    && grep -v -i -E '^[[:space:]]*(torch|torchvision|torchaudio|torch_xla|lightning|triton)([=<>!~].*)?$' /app/edmformer_requirements.txt > /app/edmformer_requirements_tpu.txt \
    && python -m pip install -r /app/edmformer_requirements_tpu.txt -c /app/constraints_tpu.txt

# Fail fast if torch_xla is broken after dependency installs.
RUN python - <<'PY'
import importlib.metadata as md
import torch
import torch_xla

print("torch:", torch.__version__)
print("torch file:", torch.__file__)
print("torch_xla:", md.version("torch_xla"))
PY

COPY . /app

CMD ["python", "src/task.py", "--help"]
