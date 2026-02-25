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
import re
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
Optional: IMAGE_NAME, TAG, IMAGE_URI, JOB_CONFIG, DOCKERFILE, JOB_NAME
EOF
  exit 1
fi

JOB_CONFIG="${JOB_CONFIG:-config/vertex_gpu_job.yaml}"
DOCKERFILE="${DOCKERFILE:-docker/training.Dockerfile}"
IMAGE_NAME="${IMAGE_NAME:-edmformer-train}"
TAG="${TAG:-$(date +%Y%m%d-%H%M%S)}"
IMAGE_REPO="${REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACT_REPO}/${IMAGE_NAME}"
IMAGE_URI="${IMAGE_URI:-${IMAGE_REPO}:${TAG}}"
SONGFORMER_CONFIG_PATH="${SONGFORMER_CONFIG_PATH:-/app/config/songformer_hr_tune.yaml}"
SONGFORMER_TRAIN_SCRIPT="${SONGFORMER_TRAIN_SCRIPT:-/app/src/train_custom.py}"
JOB_NAME="${JOB_NAME:-edmformer-gpu-$(date +%Y%m%d-%H%M)}"
PUSH_LATEST="${PUSH_LATEST:-1}"
USE_LATEST="${USE_LATEST:-1}"

if [[ ! -f "${JOB_CONFIG}" ]]; then
  echo "Job config not found: ${JOB_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${DOCKERFILE}" ]]; then
  echo "Dockerfile not found: ${DOCKERFILE}" >&2
  exit 1
fi

echo "Using image: ${IMAGE_URI}"
echo "Using job config: ${JOB_CONFIG}"
echo "Building image..."
docker build --no-cache -f "${DOCKERFILE}" -t "${IMAGE_URI}" .

echo "Pushing image..."
docker push "${IMAGE_URI}"

if [[ "${USE_LATEST}" == "1" ]]; then
  PUSH_LATEST="1"
fi

if [[ "${PUSH_LATEST}" == "1" ]]; then
  echo "Tagging and pushing latest..."
  docker tag "${IMAGE_URI}" "${IMAGE_REPO}:latest"
  docker push "${IMAGE_REPO}:latest"
fi

JOB_IMAGE_URI="${IMAGE_URI}"
if [[ "${USE_LATEST}" == "1" ]]; then
  JOB_IMAGE_URI="${IMAGE_REPO}:latest"
fi

echo "Updating imageUri in ${JOB_CONFIG}..."
if sed --version >/dev/null 2>&1; then
  # GNU sed
  sed -i -E "s|^([[:space:]]*imageUri:).*|\\1 ${JOB_IMAGE_URI}|" "${JOB_CONFIG}"
else
  # BSD sed (macOS)
  sed -i '' -E "s|^([[:space:]]*imageUri:).*|\\1 ${JOB_IMAGE_URI}|" "${JOB_CONFIG}"
fi

if [[ -n "${SONGFORMER_CONFIG_PATH}" || -n "${SONGFORMER_TRAIN_SCRIPT}" ]]; then
  JOB_CONFIG="${JOB_CONFIG}" \
  SONGFORMER_CONFIG_PATH="${SONGFORMER_CONFIG_PATH}" \
  SONGFORMER_TRAIN_SCRIPT="${SONGFORMER_TRAIN_SCRIPT}" \
  "${PYTHON_BIN}" - <<'PY'
import os
import sys
from pathlib import Path

try:
    import yaml
except Exception as exc:
    print("Warning: PyYAML not available; skipping env patch in job config.", file=sys.stderr)
    sys.exit(0)

job_config = Path(os.environ["JOB_CONFIG"])
data = yaml.safe_load(job_config.read_text(encoding="utf-8")) or {}

def _get_specs(root):
    if isinstance(root, dict):
        job_spec = root.get("jobSpec")
        if isinstance(job_spec, dict):
            specs = job_spec.get("workerPoolSpecs")
            if isinstance(specs, list):
                return specs
        specs = root.get("workerPoolSpecs")
        if isinstance(specs, list):
            return specs
    return []

def _set_env(spec, name, value):
    if not value:
        return
    container = spec.setdefault("containerSpec", {})
    envs = container.setdefault("env", [])
    if envs is None:
        envs = []
        container["env"] = envs
    for item in envs:
        if item.get("name") == name:
            item["value"] = value
            return
    envs.append({"name": name, "value": value})

specs = _get_specs(data)
if not specs:
    print("Warning: no workerPoolSpecs found in job config; skipping env patch.", file=sys.stderr)
else:
    for spec in specs:
        _set_env(spec, "SONGFORMER_CONFIG_PATH", os.environ.get("SONGFORMER_CONFIG_PATH"))
        _set_env(spec, "SONGFORMER_TRAIN_SCRIPT", os.environ.get("SONGFORMER_TRAIN_SCRIPT"))
    job_config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"Updated env in {job_config}")
PY
fi

echo "Submitting job: ${JOB_NAME}"
gcloud ai custom-jobs create \
  --region="${REGION}" \
  --project="${GCP_PROJECT}" \
  --display-name="${JOB_NAME}" \
  --config="${JOB_CONFIG}"

echo "Job submitted. Stream logs with:"
echo "  gcloud ai custom-jobs stream-logs --region=${REGION} --project=${GCP_PROJECT} <JOB_ID>"
