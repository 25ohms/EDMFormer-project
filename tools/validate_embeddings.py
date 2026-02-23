#!/usr/bin/env python3
"""
GCP Service: Cloud Storage (optional)
IAM Roles: roles/storage.objectViewer (if using GCS URIs)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

try:
    from google.cloud import storage
except ImportError:
    storage = None


def parse_gcs_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    bucket, prefix = uri[5:].split("/", 1)
    return bucket, prefix


def list_gcs_npy(client: "storage.Client", gcs_dir: str) -> set[str]:
    bucket_name, prefix = parse_gcs_uri(gcs_dir)
    prefix = prefix.rstrip("/") + "/"
    bucket = client.bucket(bucket_name)
    names: set[str] = set()
    for blob in bucket.list_blobs(prefix=prefix):
        name = Path(blob.name).name
        if name.endswith(".npy"):
            names.add(name)
    return names


def list_local_npy(dir_path: Path) -> set[str]:
    return {p.name for p in dir_path.glob("*.npy") if p.is_file()}


def load_labels(labels_jsonl: Path) -> set[str]:
    ids = set()
    with labels_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "id" not in obj:
                raise ValueError(f"Missing 'id' on line {line_no}")
            ids.add(str(obj["id"]))
    return ids


def validate_file_shapes(
    sample_paths: Iterable[Path], expected_rank: int = 3, expected_first_dim: int = 1
) -> list[str]:
    errors = []
    for path in sample_paths:
        arr = np.load(path)
        if arr.ndim != expected_rank:
            errors.append(f"{path}: expected rank {expected_rank}, got {arr.ndim}")
            continue
        if arr.shape[0] != expected_first_dim:
            errors.append(
                f"{path}: expected first dim {expected_first_dim}, got {arr.shape[0]}"
            )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate EDMFormer embedding outputs")
    parser.add_argument("--labels-jsonl", required=True, help="Path to labels.jsonl")
    parser.add_argument(
        "--embedding-dirs",
        nargs=4,
        required=True,
        help="Four embedding dirs (musicfm_30s muq_30s musicfm_420s muq_420s)",
    )
    parser.add_argument(
        "--split-ids",
        help="Optional split_ids.txt to restrict validation",
    )
    parser.add_argument(
        "--sample-shape-check",
        action="store_true",
        help="Load a small sample of files and validate array shape",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="Max files per dir to load when shape-checking",
    )
    args = parser.parse_args()

    labels_path = Path(args.labels_jsonl)
    ids = load_labels(labels_path)

    if args.split_ids:
        split_ids = {
            line.strip()
            for line in Path(args.split_ids).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        ids = ids.intersection(split_ids)

    use_gcs = any(x.startswith("gs://") for x in args.embedding_dirs)
    if use_gcs and storage is None:
        raise SystemExit(
            "google-cloud-storage is required for gs:// paths. Install it first."
        )

    client = storage.Client() if use_gcs else None

    dir_files: list[set[str]] = []
    for emb_dir in args.embedding_dirs:
        if emb_dir.startswith("gs://"):
            names = list_gcs_npy(client, emb_dir)
        else:
            names = list_local_npy(Path(emb_dir))
        dir_files.append(names)

    # Expect filenames like <id>_<start_sec>.npy
    missing = {}
    for emb_dir, names in zip(args.embedding_dirs, dir_files):
        expected = {f"{id_}_0.npy" for id_ in ids}
        missing_files = sorted(expected - names)
        if missing_files:
            missing[emb_dir] = missing_files[:20]

    if missing:
        print("Missing embeddings (showing up to 20 per dir):")
        for emb_dir, files in missing.items():
            print(f"- {emb_dir}: {len(files)} missing; examples: {files[:5]}")
        print("\nNote: This check only looks for <id>_0.npy. Ensure full 420s coverage.")
    else:
        print("All dirs contain <id>_0.npy for the provided IDs.")

    if args.sample_shape_check:
        print("\nSampling files to check shape...")
        shape_errors = []
        for emb_dir, names in zip(args.embedding_dirs, dir_files):
            sample = list(sorted(names))[: args.max_samples]
            if not sample:
                continue
            if emb_dir.startswith("gs://"):
                bucket_name, prefix = parse_gcs_uri(emb_dir)
                prefix = prefix.rstrip("/") + "/"
                bucket = client.bucket(bucket_name)
                tmp_dir = Path("/tmp/edmformer-validate")
                tmp_dir.mkdir(parents=True, exist_ok=True)
                paths = []
                for name in sample:
                    blob = bucket.blob(prefix + name)
                    local_path = tmp_dir / name
                    blob.download_to_filename(local_path)
                    paths.append(local_path)
            else:
                paths = [Path(emb_dir) / name for name in sample]
            shape_errors.extend(validate_file_shapes(paths))

        if shape_errors:
            print("Shape errors detected:")
            for err in shape_errors:
                print(f"- {err}")
        else:
            print("Shape check passed (rank 3 with first dim = 1).")


if __name__ == "__main__":
    main()
