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
if [[ -z "${REGION:-}" || -z "${GCP_PROJECT:-}" || -z "${ARTIFACT_REPO:-}" || -z "${BUCKET_NAME:-}" ]]; then
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

for key in ("REGION", "GCP_PROJECT", "ARTIFACT_REPO", "BUCKET_NAME", "LABELS_JSONL_GCS", "SPLIT_IDS_GCS", "EMBEDDINGS_GCS_DIR"):
    if os.environ.get(key):
        continue
    value = data.get(key)
    if value:
        print(f'export {key}="{sh_escape(value)}"')
PY
    )"
  fi
fi

if [[ -z "${REGION:-}" || -z "${GCP_PROJECT:-}" || -z "${ARTIFACT_REPO:-}" || -z "${BUCKET_NAME:-}" ]]; then
  cat <<'EOF'
Missing required environment variables.
Please set: REGION, GCP_PROJECT, ARTIFACT_REPO, BUCKET_NAME
Defaults can be loaded from config/gcp_env.yaml (or set GCP_ENV_CONFIG).
Optional: LABELS_JSONL_GCS, SPLIT_IDS_GCS, EMBEDDINGS_GCS_DIR, OUTPUT_ROOT
EOF
  exit 1
fi

DOCKERFILE="${DOCKERFILE:-docker/preprocessing.Dockerfile}"
IMAGE_NAME="${IMAGE_NAME:-edmformer-preprocess}"
TAG="${TAG:-$(date +%Y%m%d-%H%M%S)}"
IMAGE_URI="${IMAGE_URI:-${REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACT_REPO}/${IMAGE_NAME}:${TAG}}"

OUTPUT_ROOT="${OUTPUT_ROOT:-}"
if [[ -z "${OUTPUT_ROOT}" && -n "${EMBEDDINGS_GCS_DIR:-}" ]]; then
  if [[ "${EMBEDDINGS_GCS_DIR}" == gs://* ]]; then
    # Extract path after bucket name
    EMB_PATH="${EMBEDDINGS_GCS_DIR#gs://}"
    EMB_BUCKET="${EMB_PATH%%/*}"
    EMB_PREFIX="${EMB_PATH#*/}"
    if [[ "${EMB_BUCKET}" != "${BUCKET_NAME}" ]]; then
      echo "Warning: EMBEDDINGS_GCS_DIR bucket (${EMB_BUCKET}) != BUCKET_NAME (${BUCKET_NAME})" >&2
    fi
    OUTPUT_ROOT="${EMB_PREFIX:-embeddings}"
  else
    OUTPUT_ROOT="${EMBEDDINGS_GCS_DIR}"
  fi
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-embeddings}"

ID_ARGS=""
if [[ -n "${SPLIT_IDS_GCS:-}" ]]; then
  ID_ARGS="--split-ids ${SPLIT_IDS_GCS}"
elif [[ -n "${LABELS_JSONL_GCS:-}" ]]; then
  ID_ARGS="--labels-jsonl ${LABELS_JSONL_GCS}"
else
  echo "Missing LABELS_JSONL_GCS or SPLIT_IDS_GCS for preprocessing." >&2
  exit 1
fi

WIPE_OUTPUT="${WIPE_OUTPUT:-1}"
WIPE_FLAG=""
if [[ "${WIPE_OUTPUT}" == "1" || "${WIPE_OUTPUT}" == "true" || "${WIPE_OUTPUT}" == "yes" ]]; then
  WIPE_FLAG="--wipe-output"
fi

echo "Using image: ${IMAGE_URI}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Building preprocessing image..."
docker build --no-cache -f "${DOCKERFILE}" -t "${IMAGE_URI}" .

CMD="python preprocessing/extract_muq.py --bucket ${BUCKET_NAME} ${ID_ARGS} --output-root ${OUTPUT_ROOT} ${WIPE_FLAG} && \
python preprocessing/extract_musicfm.py --bucket ${BUCKET_NAME} ${ID_ARGS} --output-root ${OUTPUT_ROOT} ${WIPE_FLAG}"

DOCKER_ENV_ARGS=()
for env_key in HF_TOKEN MUQ_MODEL_NAME MUQ_GCS_PREFIX MUSICFM_STAT_PATH MUSICFM_MODEL_PATH MUSICFM_VARIANT MUSICFM_DEVICE MUQ_DEVICE; do
  if [[ -n "${!env_key:-}" ]]; then
    DOCKER_ENV_ARGS+=("-e" "${env_key}=${!env_key}")
  fi
done

echo "Running preprocessing locally on this VM..."
docker run --rm --gpus all \
  "${DOCKER_ENV_ARGS[@]}" \
  "${IMAGE_URI}" \
  /bin/bash -c "${CMD}"

echo "Preprocessing completed."
