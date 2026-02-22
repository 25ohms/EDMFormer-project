#!/usr/bin/env python3
"""
GCP Service: Cloud Storage
IAM Roles: roles/storage.objectAdmin (write embeddings), roles/storage.objectViewer (read audio)
"""

import argparse
import io
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from google.cloud import storage


def read_ids(split_ids_path: str | None, labels_jsonl_path: str | None) -> list[str]:
    if split_ids_path:
        return [
            line.strip()
            for line in Path(split_ids_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if labels_jsonl_path:
        ids: list[str] = []
        with Path(labels_jsonl_path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                ids.append(str(record["id"]))
        return ids
    raise ValueError("Provide --split-ids or --labels-jsonl")


def generate_embeddings_dummy(segment_seconds: int) -> Iterable[tuple[int, np.ndarray]]:
    return [(0, np.zeros((1,), dtype=np.float32))]


def upload_npy(
    client: storage.Client, bucket_name: str, blob_name: str, array: np.ndarray
) -> None:
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    buf = io.BytesIO()
    np.save(buf, array)
    buf.seek(0)
    blob.upload_from_file(buf, content_type="application/octet-stream")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract MusicFM embeddings (30s and 420s)"
    )
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument(
        "--output-root",
        default="embeddings",
        help="GCS prefix for embeddings (e.g., embeddings)",
    )
    parser.add_argument("--split-ids", help="Path to split_ids.txt")
    parser.add_argument("--labels-jsonl", help="Path to labels.jsonl")
    parser.add_argument(
        "--allow-dummy",
        action="store_true",
        help="Generate dummy embeddings (use only for wiring validation)",
    )
    args = parser.parse_args()

    ids = read_ids(args.split_ids, args.labels_jsonl)
    client = storage.Client()

    for audio_id in ids:
        for segment_seconds in (30, 420):
            if not args.allow_dummy:
                raise RuntimeError(
                    "MusicFM model loading not implemented. See preprocessing/README.md"
                )
            for start_sec, embedding in generate_embeddings_dummy(segment_seconds):
                subdir = f"musicfm_{segment_seconds}s"
                blob_name = (
                    f"{args.output_root.rstrip('/')}/{subdir}/{audio_id}_{start_sec}.npy"
                )
                upload_npy(client, args.bucket, blob_name, embedding)
                print(f"Wrote gs://{args.bucket}/{blob_name}")


if __name__ == "__main__":
    main()
