#!/usr/bin/env python3
"""
GCP Service: Vertex AI Custom Job, Cloud Storage
IAM Roles: roles/aiplatform.user, roles/storage.objectViewer
"""

import argparse
import os
from pathlib import Path

import yaml


def update_config(
    config_path: Path,
    label_path: str,
    split_ids_path: str,
    input_embedding_dir: str,
) -> None:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if data is None:
        data = {}

    data["label_path"] = label_path
    data["split_ids_path"] = split_ids_path
    data["input_embedding_dir"] = input_embedding_dir
    data["dataset_type"] = "EDMFormer"

    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update SongFormer.yaml with GCS paths at job start"
    )
    parser.add_argument(
        "--config-path",
        default=os.environ.get(
            "SONGFORMER_CONFIG_PATH",
            "third_party/SongFormer/configs/SongFormer.yaml",
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
    args = parser.parse_args()

    if not args.label_path or not args.split_ids_path or not args.input_embedding_dir:
        raise SystemExit(
            "Missing required GCS paths. Set LABEL_PATH_GCS, SPLIT_IDS_PATH_GCS, "
            "INPUT_EMBEDDING_DIR_GCS."
        )

    update_config(
        Path(args.config_path),
        args.label_path,
        args.split_ids_path,
        args.input_embedding_dir,
    )
    print(f"Updated {args.config_path}")


if __name__ == "__main__":
    main()
