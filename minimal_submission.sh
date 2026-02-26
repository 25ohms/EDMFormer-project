#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export JOB_CONFIG="${JOB_CONFIG:-config/vertex_gpu_job.minimal.yaml}"
export JOB_NAME="${JOB_NAME:-edmformer-gpu-minimal-$(date +%Y%m%d-%H%M)}"

./submit_gpu_job.sh
