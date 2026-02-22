#!/usr/bin/env python3
"""
GCP Service: Vertex AI Custom Job, Cloud Storage
IAM Roles: roles/aiplatform.user, roles/storage.objectViewer
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from config_generator import update_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Vertex AI CustomJob entrypoint")
    parser.add_argument(
        "--config-path",
        default=os.environ.get(
            "SONGFORMER_CONFIG_PATH",
            "third_party/EDMFormer/configs/SongFormer.yaml",
        ),
    )
    parser.add_argument(
        "--label-path",
        default=os.environ.get("LABEL_PATH_GCS"),
    )
    parser.add_argument(
        "--split-ids-path",
        default=os.environ.get("SPLIT_IDS_PATH_GCS"),
    )
    parser.add_argument(
        "--input-embedding-dir",
        default=os.environ.get("INPUT_EMBEDDING_DIR_GCS"),
    )
    parser.add_argument(
        "--train-script",
        default=os.environ.get(
            "SONGFORMER_TRAIN_SCRIPT",
            "third_party/EDMFormer/train.py",
        ),
    )
    parser.add_argument(
        "--train-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments passed through to the SongFormer training script",
    )
    args = parser.parse_args()

    if not args.label_path or not args.split_ids_path or not args.input_embedding_dir:
        raise SystemExit(
            "Missing required GCS paths. Set LABEL_PATH_GCS, SPLIT_IDS_PATH_GCS, "
            "INPUT_EMBEDDING_DIR_GCS."
        )

    update_config(
        config_path=Path(args.config_path),
        label_path=args.label_path,
        split_ids_path=args.split_ids_path,
        input_embedding_dir=args.input_embedding_dir,
    )

    cmd = [sys.executable, args.train_script] + args.train_args
    print(f"Launching training: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
