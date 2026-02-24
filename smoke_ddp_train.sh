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
if [[ -z "${REGION:-}" || -z "${GCP_PROJECT:-}" || -z "${ARTIFACT_REPO:-}" || -z "${SPLIT_IDS_GCS:-}" || -z "${EVAL_SPLIT_IDS_GCS:-}" || -z "${LABELS_JSONL_GCS:-}" || -z "${EMBEDDINGS_GCS_DIR:-}" ]]; then
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

for key in ("REGION", "GCP_PROJECT", "ARTIFACT_REPO", "SPLIT_IDS_GCS", "EVAL_SPLIT_IDS_GCS", "LABELS_JSONL_GCS", "EMBEDDINGS_GCS_DIR", "EMBEDDING_SUBDIRS"):
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
Optional (task entrypoint): LABEL_PATH_GCS, SPLIT_IDS_PATH_GCS, EVAL_SPLIT_IDS_PATH_GCS,
  INPUT_EMBEDDING_DIR_GCS, EMBEDDING_SUBDIRS
EOF
  exit 1
fi

# Map legacy env names from config/gcp_env.yaml to task.py expectations.
LABEL_PATH_GCS="${LABEL_PATH_GCS:-${LABELS_JSONL_GCS:-}}"
SPLIT_IDS_PATH_GCS="${SPLIT_IDS_PATH_GCS:-${SPLIT_IDS_GCS:-}}"
EVAL_SPLIT_IDS_PATH_GCS="${EVAL_SPLIT_IDS_PATH_GCS:-${EVAL_SPLIT_IDS_GCS:-}}"
INPUT_EMBEDDING_DIR_GCS="${INPUT_EMBEDDING_DIR_GCS:-${EMBEDDINGS_GCS_DIR:-}}"
EMBEDDING_SUBDIRS="${EMBEDDING_SUBDIRS:-}"
PREFETCH_EMBEDDINGS="${PREFETCH_EMBEDDINGS:-}"

IMAGE_NAME="${IMAGE_NAME:-edmformer-train}"
TAG="${TAG:-}"
IMAGE_REPO="${REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACT_REPO}/${IMAGE_NAME}"
USE_LATEST="${USE_LATEST:-1}"
PULL_IMAGE="${PULL_IMAGE:-1}"

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
RUN_IMAGE_URI="${IMAGE_URI}"

if [[ "${USE_LATEST}" == "1" && "${IMAGE_URI}" != "${IMAGE_REPO}:latest" ]]; then
  if docker image inspect "${IMAGE_URI}" >/dev/null 2>&1; then
    docker tag "${IMAGE_URI}" "${IMAGE_REPO}:latest"
    RUN_IMAGE_URI="${IMAGE_REPO}:latest"
  else
    if docker pull "${IMAGE_URI}" >/dev/null 2>&1; then
      docker tag "${IMAGE_URI}" "${IMAGE_REPO}:latest"
      RUN_IMAGE_URI="${IMAGE_REPO}:latest"
    fi
  fi
fi
CONFIG_PATH="${CONFIG_PATH:-/app/third_party/EDMFormer/src/SongFormer/configs/SongFormer.yaml}"
NPROC="${NPROC:-}"
MAX_STEPS="${MAX_STEPS:-1}"
GPU_DEVICES="${GPU_DEVICES:-}"
USE_TASK_ENTRYPOINT="${USE_TASK_ENTRYPOINT:-1}"
DOCKER_WORKDIR="/app/third_party/EDMFormer/src/SongFormer"
if [[ "${USE_TASK_ENTRYPOINT}" == "1" ]]; then
  DOCKER_WORKDIR="/app"
fi
SHM_SIZE="${SHM_SIZE:-2g}"
USE_HOST_IPC="${USE_HOST_IPC:-0}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-}"

HOST_GPU_COUNT=0
if command -v nvidia-smi >/dev/null 2>&1; then
  HOST_GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
fi

GPU_LIST=()
if [[ -n "${GPU_DEVICES}" ]]; then
  IFS=',' read -r -a GPU_LIST <<< "${GPU_DEVICES}"
  GPU_LIST=("${GPU_LIST[@]/#/}" )
  GPU_LIST=("${GPU_LIST[@]/%/}" )
  GPU_COUNT=${#GPU_LIST[@]}
  if [[ "${HOST_GPU_COUNT}" -gt 0 && "${GPU_COUNT}" -gt "${HOST_GPU_COUNT}" ]]; then
    echo "Warning: GPU_DEVICES (${GPU_DEVICES}) exceeds host GPU count (${HOST_GPU_COUNT}); trimming." >&2
    GPU_LIST=("${GPU_LIST[@]:0:${HOST_GPU_COUNT}}")
    GPU_DEVICES="$(IFS=','; echo "${GPU_LIST[*]}")"
    GPU_COUNT=${#GPU_LIST[@]}
  fi
else
  GPU_COUNT=${HOST_GPU_COUNT}
fi

if [[ -z "${NPROC}" ]]; then
  if [[ "${GPU_COUNT}" -gt 0 ]]; then
    NPROC="${GPU_COUNT}"
  else
    NPROC="1"
  fi
elif [[ "${GPU_COUNT}" -gt 0 && "${NPROC}" -gt "${GPU_COUNT}" ]]; then
  echo "Warning: NPROC (${NPROC}) exceeds visible GPU count (${GPU_COUNT}); reducing." >&2
  NPROC="${GPU_COUNT}"
fi

echo "Using image: ${RUN_IMAGE_URI}"
echo "Config: ${CONFIG_PATH}"
echo "DDP processes: ${NPROC}"
echo "Max steps: ${MAX_STEPS}"
if [[ "${USE_TASK_ENTRYPOINT}" == "1" ]]; then
  if [[ -z "${LABEL_PATH_GCS}" || -z "${SPLIT_IDS_PATH_GCS}" || -z "${INPUT_EMBEDDING_DIR_GCS}" ]]; then
    cat <<'EOF' >&2
Missing required GCS paths for task entrypoint.
Please set: LABEL_PATH_GCS, SPLIT_IDS_PATH_GCS, INPUT_EMBEDDING_DIR_GCS
Optional: EVAL_SPLIT_IDS_PATH_GCS, EMBEDDING_SUBDIRS
EOF
    exit 1
  fi
fi

if [[ "${PULL_IMAGE}" == "1" ]]; then
  docker pull "${RUN_IMAGE_URI}"
fi

# Ensure eval split exists in GCS if configured.
if [[ "${ENSURE_VAL_SPLIT:-1}" == "1" && -n "${SPLIT_IDS_GCS:-}" && -n "${EVAL_SPLIT_IDS_GCS:-}" ]]; then
  if command -v gcloud >/dev/null 2>&1; then
    if ! gcloud storage ls "${EVAL_SPLIT_IDS_GCS}" >/dev/null 2>&1; then
      echo "Eval split not found at ${EVAL_SPLIT_IDS_GCS}; copying from ${SPLIT_IDS_GCS}"
      gcloud storage cp "${SPLIT_IDS_GCS}" "${EVAL_SPLIT_IDS_GCS}"
    fi
  else
    echo "gcloud not found; skipping eval split check."
  fi
fi

GCLOUD_MOUNT=()
if [[ "${MOUNT_GCLOUD:-1}" == "1" && -d "${HOME}/.config/gcloud" ]]; then
  GCLOUD_MOUNT=(-v "${HOME}/.config/gcloud:/root/.config/gcloud:ro")
fi

NVIDIA_DEVICES=()
if [[ -n "${GPU_DEVICES}" ]]; then
  NVIDIA_DEVICES=(-e "NVIDIA_VISIBLE_DEVICES=${GPU_DEVICES}" -e "CUDA_VISIBLE_DEVICES=${GPU_DEVICES}")
fi

DOCKER_GPU_ARGS=(--gpus all)
if [[ -n "${GPU_DEVICES}" ]]; then
  DOCKER_GPU_ARGS=(--gpus "device=${GPU_DEVICES}")
fi

DOCKER_IPC_ARGS=()
if [[ "${USE_HOST_IPC}" == "1" ]]; then
  DOCKER_IPC_ARGS=(--ipc=host)
fi

docker run --rm "${DOCKER_GPU_ARGS[@]}" "${DOCKER_IPC_ARGS[@]}" \
  --shm-size="${SHM_SIZE}" \
  -e WANDB_MODE=disabled \
  -e TORCH_DISTRIBUTED_DEBUG=DETAIL \
  -e PYTHONPATH=/app/src:/app/third_party/EDMFormer/src/SongFormer:/app/third_party/EDMFormer/src \
  -e LABEL_PATH_GCS="${LABEL_PATH_GCS}" \
  -e SPLIT_IDS_PATH_GCS="${SPLIT_IDS_PATH_GCS}" \
  -e EVAL_SPLIT_IDS_PATH_GCS="${EVAL_SPLIT_IDS_PATH_GCS}" \
  -e INPUT_EMBEDDING_DIR_GCS="${INPUT_EMBEDDING_DIR_GCS}" \
  -e EMBEDDING_SUBDIRS="${EMBEDDING_SUBDIRS}" \
  -e PREFETCH_EMBEDDINGS="${PREFETCH_EMBEDDINGS}" \
  -e DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS}" \
  -e DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR}" \
  -e DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS}" \
  -e DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY}" \
  -w "${DOCKER_WORKDIR}" \
  "${GCLOUD_MOUNT[@]}" \
  "${NVIDIA_DEVICES[@]}" \
  "${RUN_IMAGE_URI}" \
  bash -c "\
    if [[ '${USE_TASK_ENTRYPOINT}' == '1' ]]; then \
      python /app/src/task.py \
        --config-path '${CONFIG_PATH}' \
        --num-gpus '${NPROC}' \
        --checkpoint-dir /tmp/edmformer-smoke \
        --init-seed 42 \
        --train-args --max_steps '${MAX_STEPS}' --max_epochs 1 --log_interval 1; \
    else \
      python -m torch.distributed.run --standalone --nproc_per_node '${NPROC}' \
        /app/third_party/EDMFormer/src/SongFormer/train/train.py \
        --config '${CONFIG_PATH}' \
        --init_seed 42 \
        --checkpoint_dir /tmp/edmformer-smoke \
        --max_steps '${MAX_STEPS}' \
        --max_epochs 1 \
        --log_interval 1; \
    fi\
  "

echo "DDP smoke test completed."
