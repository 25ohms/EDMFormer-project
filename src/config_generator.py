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
    train_split_ids_path: str,
    eval_split_ids_path: str,
    input_embedding_dir: str,
    dataset_type: str = "EDMFormer",
) -> None:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if data is None:
        data = {}

    def ensure_dataset_section(key: str) -> None:
        if key not in data or data[key] is None:
            data[key] = {}
        data[key]["_target_"] = "edmformer_gcs_dataset.Dataset"
        if "dataset_abstracts" not in data[key] or data[key]["dataset_abstracts"] is None:
            data[key]["dataset_abstracts"] = []
        if "hparams" not in data[key] or data[key]["hparams"] is None:
            data[key]["hparams"] = {}

    train_item = {
        "internal_tmp_id": dataset_type,
        "dataset_type": dataset_type,
        "input_embedding_dir": input_embedding_dir,
        "label_path": label_path,
        "split_ids_path": train_split_ids_path,
        "multiplier": 1,
    }
    eval_item = {
        "internal_tmp_id": dataset_type,
        "dataset_type": dataset_type,
        "input_embedding_dir": input_embedding_dir,
        "label_path": label_path,
        "split_ids_path": eval_split_ids_path,
        "multiplier": 1,
    }

    ensure_dataset_section("train_dataset")
    data["train_dataset"]["dataset_abstracts"] = [train_item]

    ensure_dataset_section("eval_dataset")
    data["eval_dataset"]["dataset_abstracts"] = [eval_item]

    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update SongFormer.yaml with GCS paths at job start"
    )
    parser.add_argument(
        "--config-path",
        default=os.environ.get(
            "SONGFORMER_CONFIG_PATH",
            "third_party/EDMFormer/src/SongFormer/configs/SongFormer.yaml",
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
        "--eval-split-ids-path",
        default=os.environ.get("EVAL_SPLIT_IDS_PATH_GCS"),
        help="Optional eval split ids path (defaults to --split-ids-path)",
    )
    parser.add_argument(
        "--input-embedding-dir",
        default=os.environ.get("INPUT_EMBEDDING_DIR_GCS"),
    )
    parser.add_argument(
        "--dataset-type",
        default=os.environ.get("DATASET_TYPE", "EDMFormer"),
    )
    args = parser.parse_args()

    if not args.label_path or not args.split_ids_path or not args.input_embedding_dir:
        raise SystemExit(
            "Missing required GCS paths. Set LABEL_PATH_GCS, SPLIT_IDS_PATH_GCS, "
            "INPUT_EMBEDDING_DIR_GCS."
        )

    eval_split_ids_path = args.eval_split_ids_path or args.split_ids_path

    update_config(
        Path(args.config_path),
        args.label_path,
        args.split_ids_path,
        eval_split_ids_path,
        args.input_embedding_dir,
        dataset_type=args.dataset_type,
    )
    print(f"Updated {args.config_path}")


if __name__ == "__main__":
    main()
