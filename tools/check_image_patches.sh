#!/usr/bin/env bash
set -euo pipefail

IMAGE_URI="${1:-us-central1-docker.pkg.dev/edmformer-project/edmformer-repo/edmformer-train:latest}"

docker run --rm "${IMAGE_URI}" bash -lc "python - <<'PY'
from pathlib import Path
train = Path('/app/third_party/EDMFormer/src/SongFormer/train/train.py').read_text()
model = Path('/app/third_party/EDMFormer/src/SongFormer/models/SongFormer.py').read_text()
print('skip_batch patch:', 'skip_batch = torch.tensor' in train)
print('loss tensor init:', 'loss = torch.zeros' in model)
PY"
