#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

for cmd in docker gcloud; do
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
Optional: IMAGE_NAME, TAG, DOCKERFILE, NO_CACHE, PUSH_LATEST
EOF
  exit 1
fi

IMAGE_NAME="${IMAGE_NAME:-edmformer-train}"
TAG="${TAG:-$(date +%Y%m%d-%H%M%S)}"
DOCKERFILE="${DOCKERFILE:-docker/training.Dockerfile}"
NO_CACHE="${NO_CACHE:-0}"
PUSH_LATEST="${PUSH_LATEST:-1}"

IMAGE_REPO="${REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACT_REPO}/${IMAGE_NAME}"
IMAGE_URI="${IMAGE_REPO}:${TAG}"

echo "Building image: ${IMAGE_URI}"
if [[ "${NO_CACHE}" == "1" ]]; then
  docker build --no-cache -f "${DOCKERFILE}" -t "${IMAGE_URI}" .
else
  docker build -f "${DOCKERFILE}" -t "${IMAGE_URI}" .
fi

echo "Pushing image: ${IMAGE_URI}"
docker push "${IMAGE_URI}"

if [[ "${PUSH_LATEST}" == "1" ]]; then
  echo "Tagging and pushing latest..."
  docker tag "${IMAGE_URI}" "${IMAGE_REPO}:latest"
  docker push "${IMAGE_REPO}:latest"
fi

echo "Running DDP smoke test on latest..."
SONGFORMER_CONFIG_PATH="${SONGFORMER_CONFIG_PATH:-/app/config/songformer_hr_tune.yaml}" \
SONGFORMER_TRAIN_SCRIPT="${SONGFORMER_TRAIN_SCRIPT:-/app/src/train_custom.py}" \
CONFIG_PATH="${CONFIG_PATH:-${SONGFORMER_CONFIG_PATH}}" \
USE_LATEST=1 PULL_IMAGE=1 ./smoke_ddp_train.sh
