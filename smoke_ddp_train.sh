#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

for cmd in docker; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 1
  fi
done

PYTHON_BIN="python3"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Missing required command: python3 (or python)" >&2
  exit 1
fi

GCP_ENV_CONFIG="${GCP_ENV_CONFIG:-config/gcp_env.yaml}"
if [[ -z "${REGION:-}" || -z "${GCP_PROJECT:-}" || -z "${ARTIFACT_REPO:-}" ]]; then
  if [[ -f "${GCP_ENV_CONFIG}" ]]; then
    eval "$("${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

cfg = Path(os.environ.get("GCP_ENV_CONFIG", "config/gcp_env.yaml"))
if not cfg.exists():
    raise SystemExit(0)

def sh_escape(value: str) -> str:
    return value.replace("\\\\", "\\\\\\\\").replace('"', '\\"')

data = {}
for line in cfg.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or ":" not in line:
        continue
    key, val = line.split(":", 1)
    key = key.strip()
    val = val.strip()
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        val = val[1:-1]
    data[key] = val

for key in ("REGION", "GCP_PROJECT", "ARTIFACT_REPO"):
    if os.environ.get(key):
        continue
    value = data.get(key)
    if value:
        print(f'export {key}="{sh_escape(value)}"')
PY
    )"
  fi
fi

if [[ -z "${REGION:-}" || -z "${GCP_PROJECT:-}" || -z "${ARTIFACT_REPO:-}" ]]; then
  cat <<'EOF'
Missing required environment variables.
Please set: REGION, GCP_PROJECT, ARTIFACT_REPO
Defaults can be loaded from config/gcp_env.yaml (or set GCP_ENV_CONFIG).
Optional: IMAGE_NAME, TAG, IMAGE_URI, CONFIG_PATH, NPROC, MAX_STEPS
EOF
  exit 1
fi

IMAGE_NAME="${IMAGE_NAME:-edmformer-train}"
TAG="${TAG:-}"
IMAGE_REPO="${REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACT_REPO}/${IMAGE_NAME}"

if [[ -z "${IMAGE_URI:-}" && -z "${TAG}" ]]; then
  if command -v gcloud >/dev/null 2>&1; then
    latest_tags="$(gcloud artifacts docker images list "${IMAGE_REPO}" \
      --project="${GCP_PROJECT}" \
      --include-tags \
      --sort-by=~UPDATE_TIME \
      --limit=1 \
      --format='value(TAGS)' 2>/dev/null || true)"
    if [[ -n "${latest_tags}" ]]; then
      TAG="${latest_tags%%,*}"
    fi
  fi
fi

if [[ -z "${IMAGE_URI:-}" && -z "${TAG}" ]]; then
  TAG="latest"
fi

IMAGE_URI="${IMAGE_URI:-${IMAGE_REPO}:${TAG}}"
CONFIG_PATH="${CONFIG_PATH:-/app/third_party/EDMFormer/src/SongFormer/configs/SongFormer.yaml}"
NPROC="${NPROC:-2}"
MAX_STEPS="${MAX_STEPS:-1}"

echo "Using image: ${IMAGE_URI}"
echo "Config: ${CONFIG_PATH}"
echo "DDP processes: ${NPROC}"
echo "Max steps: ${MAX_STEPS}"

docker run --rm --gpus all \
  -e WANDB_MODE=disabled \
  -e TORCH_DISTRIBUTED_DEBUG=DETAIL \
  "${IMAGE_URI}" \
  python -m torch.distributed.run --standalone --nproc_per_node "${NPROC}" \
    /app/third_party/EDMFormer/src/SongFormer/train/train.py \
    --config "${CONFIG_PATH}" \
    --init_seed 42 \
    --checkpoint_dir /tmp/edmformer-smoke \
    --max_steps "${MAX_STEPS}" \
    --max_epochs 1 \
    --log_interval 1

echo "DDP smoke test completed."
