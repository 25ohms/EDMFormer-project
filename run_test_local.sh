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
if [[ -z "${REGION:-}" || -z "${GCP_PROJECT:-}" || -z "${ARTIFACT_REPO:-}" || -z "${LABELS_JSONL_GCS:-}" || -z "${SPLIT_IDS_GCS:-}" || -z "${TEST_IDS_GCS:-}" || -z "${EMBEDDINGS_GCS_DIR:-}" ]]; then
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

for key in ("REGION", "GCP_PROJECT", "ARTIFACT_REPO", "LABELS_JSONL_GCS", "SPLIT_IDS_GCS", "TEST_IDS_GCS", "EMBEDDINGS_GCS_DIR", "EMBEDDING_SUBDIRS"):
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
EOF
  exit 1
fi

LABEL_PATH_GCS="${LABEL_PATH_GCS:-${LABELS_JSONL_GCS:-}}"
SPLIT_IDS_PATH_GCS="${SPLIT_IDS_PATH_GCS:-${SPLIT_IDS_GCS:-}}"
TEST_IDS_PATH_GCS="${TEST_IDS_PATH_GCS:-${TEST_IDS_GCS:-}}"
INPUT_EMBEDDING_DIR_GCS="${INPUT_EMBEDDING_DIR_GCS:-${EMBEDDINGS_GCS_DIR:-}}"
EMBEDDING_SUBDIRS="${EMBEDDING_SUBDIRS:-}"

if [[ -z "${LABEL_PATH_GCS}" || -z "${SPLIT_IDS_PATH_GCS}" || -z "${TEST_IDS_PATH_GCS}" || -z "${INPUT_EMBEDDING_DIR_GCS}" ]]; then
  cat <<'EOF' >&2
Missing required dataset paths.
Please set: LABEL_PATH_GCS, SPLIT_IDS_PATH_GCS, TEST_IDS_PATH_GCS, INPUT_EMBEDDING_DIR_GCS
EOF
  exit 1
fi

IMAGE_NAME="${IMAGE_NAME:-edmformer-train}"
TAG="${TAG:-latest}"
IMAGE_REPO="${REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACT_REPO}/${IMAGE_NAME}"
IMAGE_URI="${IMAGE_URI:-${IMAGE_REPO}:${TAG}}"
PULL_IMAGE="${PULL_IMAGE:-1}"
GPU_DEVICES="${GPU_DEVICES:-0}"

CONFIG_PATH="${CONFIG_PATH:-/app/third_party/EDMFormer/src/SongFormer/configs/SongFormer.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
CHECKPOINT="${CHECKPOINT:-}"
DATASET_TYPE="${DATASET_TYPE:-}"

if [[ -z "${CHECKPOINT_DIR}" && -z "${CHECKPOINT}" ]]; then
  cat <<'EOF' >&2
Missing checkpoint reference.
Please set CHECKPOINT_DIR or CHECKPOINT (can be gs://...).
EOF
  exit 1
fi

if [[ "${PULL_IMAGE}" == "1" ]]; then
  docker pull "${IMAGE_URI}"
fi

CUDA_ARGS=()
if [[ -n "${GPU_DEVICES}" ]]; then
  CUDA_ARGS+=("-e" "CUDA_VISIBLE_DEVICES=${GPU_DEVICES}")
fi

ENV_ARGS=(
  "-e" "LABEL_PATH_GCS=${LABEL_PATH_GCS}"
  "-e" "SPLIT_IDS_PATH_GCS=${SPLIT_IDS_PATH_GCS}"
  "-e" "TEST_IDS_PATH_GCS=${TEST_IDS_PATH_GCS}"
  "-e" "INPUT_EMBEDDING_DIR_GCS=${INPUT_EMBEDDING_DIR_GCS}"
)
if [[ -n "${EMBEDDING_SUBDIRS}" ]]; then
  ENV_ARGS+=("-e" "EMBEDDING_SUBDIRS=${EMBEDDING_SUBDIRS}")
fi
if [[ -n "${DATASET_TYPE}" ]]; then
  ENV_ARGS+=("-e" "DATASET_TYPE=${DATASET_TYPE}")
fi

CMD=(python /app/src/test.py --config "${CONFIG_PATH}")
if [[ -n "${CHECKPOINT_DIR}" ]]; then
  CMD+=(--checkpoint-dir "${CHECKPOINT_DIR}")
else
  CMD+=(--checkpoint "${CHECKPOINT}")
fi

echo "Using image: ${IMAGE_URI}"
echo "Running test in container..."
docker run --rm --gpus all \
  "${CUDA_ARGS[@]}" \
  "${ENV_ARGS[@]}" \
  "${IMAGE_URI}" \
  "${CMD[@]}"
