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

from google.cloud import storage

from config_generator import update_config

DEFAULT_EMBEDDING_SUBDIRS = "musicfm_30s,muq_30s,musicfm_420s,muq_420s"


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    bucket, prefix = uri[5:].split("/", 1)
    return bucket, prefix


def download_gcs_prefix(
    client: storage.Client, gcs_prefix: str, dest_dir: Path
) -> Path:
    bucket_name, prefix = parse_gcs_uri(gcs_prefix)
    bucket = client.bucket(bucket_name)
    dest_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix.rstrip("/")
    for blob in bucket.list_blobs(prefix=prefix):
        rel = blob.name[len(prefix) :].lstrip("/")
        if not rel:
            continue
        out_path = dest_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(out_path)
    return dest_dir


def upload_dir_to_gcs(
    client: storage.Client, src_dir: Path, gcs_prefix: str
) -> None:
    bucket_name, prefix = parse_gcs_uri(gcs_prefix)
    bucket = client.bucket(bucket_name)
    prefix = prefix.rstrip("/")
    for path in src_dir.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(src_dir).as_posix()
        blob = bucket.blob(f"{prefix}/{rel}")
        blob.upload_from_filename(path)


def resolve_embedding_dirs(base_or_list: str, subdirs: list[str]) -> list[str]:
    if " " in base_or_list.strip():
        return [x for x in base_or_list.split() if x]
    base = base_or_list.rstrip("/")
    return [f"{base}/{subdir}" for subdir in subdirs]


def ensure_arg(args_list: list[str], flag: str, value: str) -> list[str]:
    if flag in args_list:
        return args_list
    return args_list + [flag, value]


def main() -> None:
    parser = argparse.ArgumentParser(description="Vertex AI CustomJob entrypoint")
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
        "--embedding-subdirs",
        default=os.environ.get("EMBEDDING_SUBDIRS", DEFAULT_EMBEDDING_SUBDIRS),
        help="Comma-separated subdir names used when input-embedding-dir is a root",
    )
    parser.add_argument(
        "--dataset-type",
        default=os.environ.get("DATASET_TYPE", "EDMFormer"),
    )
    parser.add_argument(
        "--train-script",
        default=os.environ.get(
            "SONGFORMER_TRAIN_SCRIPT",
            "third_party/EDMFormer/src/SongFormer/train/train.py",
        ),
    )
    parser.add_argument(
        "--init-seed",
        type=int,
        default=int(os.environ.get("INIT_SEED", "42")),
        help="Random seed for initialization",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=os.environ.get("CHECKPOINT_DIR_GCS"),
        help="Checkpoint dir (local path or gs://). If gs://, will sync to local and upload after training.",
    )
    parser.add_argument(
        "--local-data-dir",
        default=os.environ.get("LOCAL_DATA_DIR", "/tmp/edmformer-data"),
        help="Local staging directory inside the container",
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

    if "--checkpoint_dir" in args.train_args and args.checkpoint_dir:
        raise SystemExit(
            "Provide checkpoint dir via --checkpoint-dir or --train-args --checkpoint_dir, not both."
        )

    eval_split_ids_path = args.eval_split_ids_path or args.split_ids_path

    local_root = Path(args.local_data_dir)
    local_root.mkdir(parents=True, exist_ok=True)

    client = storage.Client()

    label_path = args.label_path
    train_split_path = args.split_ids_path
    eval_split_path = eval_split_ids_path

    embedding_subdirs_raw = args.embedding_subdirs.strip() or DEFAULT_EMBEDDING_SUBDIRS
    embedding_subdirs = [
        x.strip() for x in embedding_subdirs_raw.split(",") if x.strip()
    ]
    embedding_inputs = resolve_embedding_dirs(args.input_embedding_dir, embedding_subdirs)
    local_input_embedding_dir = " ".join(embedding_inputs)

    config_path = Path(args.config_path).resolve()

    update_config(
        config_path=config_path,
        label_path=str(label_path),
        train_split_ids_path=str(train_split_path),
        eval_split_ids_path=str(eval_split_path),
        input_embedding_dir=local_input_embedding_dir,
        dataset_type=args.dataset_type,
    )

    train_args = list(args.train_args)
    train_args = ensure_arg(train_args, "--config", str(config_path))
    train_args = ensure_arg(train_args, "--init_seed", str(args.init_seed))

    local_checkpoint_dir = None
    checkpoint_gcs = None
    if args.checkpoint_dir:
        if args.checkpoint_dir.startswith("gs://"):
            checkpoint_gcs = args.checkpoint_dir
            _, prefix = parse_gcs_uri(args.checkpoint_dir)
            checkpoint_local = (
                local_root / "checkpoints" / Path(prefix.rstrip("/")).name
            )
            if not checkpoint_local.exists():
                checkpoint_local.mkdir(parents=True, exist_ok=True)
            download_gcs_prefix(client, args.checkpoint_dir, checkpoint_local)
            local_checkpoint_dir = checkpoint_local
        else:
            local_checkpoint_dir = Path(args.checkpoint_dir)

    if local_checkpoint_dir is not None:
        train_args = ensure_arg(train_args, "--checkpoint_dir", str(local_checkpoint_dir))

    train_script = Path(args.train_script).resolve()
    workdir = Path("third_party/EDMFormer/src/SongFormer").resolve()
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"

    env = os.environ.copy()
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env["PYTHONPATH"] = f"{src_root}:{workdir}:{env.get('PYTHONPATH', '')}"

    cmd = [sys.executable, str(train_script)] + train_args
    print(f"Launching training: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=workdir, env=env)

    if checkpoint_gcs and local_checkpoint_dir is not None:
        upload_dir_to_gcs(client, local_checkpoint_dir, checkpoint_gcs)


if __name__ == "__main__":
    main()
