#!/usr/bin/env python3
"""
Validate embedding shapes across subdirs for each utterance id in GCS.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
from google.cloud import storage


def normalize_gcs_uri(uri: str) -> str:
    if uri.startswith("gs:/") and not uri.startswith("gs://"):
        return "gs://" + uri[4:]
    return uri


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    uri = normalize_gcs_uri(uri)
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    bucket, prefix = uri[5:].split("/", 1)
    return bucket, prefix


def read_text_uri(uri: str, client: storage.Client | None = None) -> str:
    uri = normalize_gcs_uri(uri)
    if uri.startswith("gs://"):
        bucket_name, blob_name = parse_gcs_uri(uri)
        client = client or storage.Client()
        blob = client.bucket(bucket_name).blob(blob_name)
        return blob.download_as_text(encoding="utf-8")
    return Path(uri).read_text(encoding="utf-8")


def read_ids(split_ids_path: str | None, labels_jsonl_path: str | None) -> set[str]:
    if split_ids_path:
        text = read_text_uri(split_ids_path)
        return {line.strip() for line in text.splitlines() if line.strip()}
    if labels_jsonl_path:
        text = read_text_uri(labels_jsonl_path)
        ids: set[str] = set()
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ids.add(str(record["id"]))
        return ids
    return set()


def list_embeddings(bucket: storage.Bucket, prefix: str) -> dict[str, str]:
    prefix = prefix.rstrip("/") + "/"
    result: dict[str, str] = {}
    for blob in bucket.list_blobs(prefix=prefix):
        if not blob.name.endswith(".npy"):
            continue
        utt = Path(blob.name).stem
        result[utt] = blob.name
    return result


def load_shape(bucket: storage.Bucket, blob_name: str) -> tuple[int, int]:
    data = bucket.blob(blob_name).download_as_bytes()
    arr = np.load(BytesIO(data))
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr.squeeze(axis=0)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected embedding shape {arr.shape} for {blob_name}")
    return int(arr.shape[0]), int(arr.shape[1])


def base_id_from_utt(utt: str) -> str:
    if "_" not in utt:
        return utt
    return utt.rsplit("_", 1)[0]


@dataclass
class CheckResult:
    missing: dict[str, int]
    mismatches: list[str]
    checked: int


def check_shapes(
    bucket: storage.Bucket,
    subdir_maps: dict[str, dict[str, str]],
    allowed_ids: set[str] | None,
    max_diff: int,
    max_utterances: int | None,
    seed: int,
) -> CheckResult:
    subdirs = list(subdir_maps.keys())
    all_utts: set[str] = set.intersection(*(set(m.keys()) for m in subdir_maps.values()))

    if allowed_ids:
        all_utts = {utt for utt in all_utts if base_id_from_utt(utt) in allowed_ids}

    utt_list = sorted(all_utts)
    if max_utterances is not None and max_utterances < len(utt_list):
        random.seed(seed)
        utt_list = random.sample(utt_list, max_utterances)

    missing_counts = {s: 0 for s in subdirs}
    mismatches: list[str] = []

    for utt in utt_list:
        shapes = {}
        for subdir in subdirs:
            blob_name = subdir_maps[subdir].get(utt)
            if not blob_name:
                missing_counts[subdir] += 1
                continue
            shapes[subdir] = load_shape(bucket, blob_name)
        if len(shapes) != len(subdirs):
            continue
        lengths = [shape[0] for shape in shapes.values()]
        if max(lengths) - min(lengths) > max_diff:
            detail = ", ".join(f"{k}={v}" for k, v in shapes.items())
            mismatches.append(f"{utt}: {detail}")

    return CheckResult(missing=missing_counts, mismatches=mismatches, checked=len(utt_list))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check embedding shape consistency in GCS.")
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument("--output-root", default="embeddings", help="Embedding root prefix")
    parser.add_argument("--subdirs", default="musicfm_30s,muq_30s,musicfm_420s,muq_420s")
    parser.add_argument("--split-ids", help="Path to split_ids.txt")
    parser.add_argument("--labels-jsonl", help="Path to labels.jsonl")
    parser.add_argument("--max-diff", type=int, default=4)
    parser.add_argument("--max-utterances", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on issues.")
    args = parser.parse_args()

    client = storage.Client()
    bucket = client.bucket(args.bucket)
    subdirs = [s.strip() for s in args.subdirs.split(",") if s.strip()]
    allowed_ids = read_ids(args.split_ids, args.labels_jsonl)
    allowed_ids = allowed_ids or None

    subdir_maps: dict[str, dict[str, str]] = {}
    for subdir in subdirs:
        prefix = f"{args.output_root.rstrip('/')}/{subdir}"
        subdir_maps[subdir] = list_embeddings(bucket, prefix)

    result = check_shapes(
        bucket=bucket,
        subdir_maps=subdir_maps,
        allowed_ids=allowed_ids,
        max_diff=args.max_diff,
        max_utterances=args.max_utterances,
        seed=args.seed,
    )

    print(f"Checked {result.checked} utterances.")
    for subdir, count in result.missing.items():
        if count:
            print(f"Missing in {subdir}: {count}")
    if result.mismatches:
        print(f"Shape mismatches (> {args.max_diff} frames): {len(result.mismatches)}")
        for line in result.mismatches[:25]:
            print(f"  {line}")
        if len(result.mismatches) > 25:
            print("  ...")
    else:
        print("No shape mismatches found.")

    if args.strict and (any(result.missing.values()) or result.mismatches):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
