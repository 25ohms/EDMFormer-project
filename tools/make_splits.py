#!/usr/bin/env python3
import argparse
import json
import random
from io import BytesIO
from pathlib import Path
from typing import Iterable

from google.cloud import storage


def _is_gcs(path: str) -> bool:
    return path.startswith("gs://")


def _parse_gcs(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    bucket, blob = uri[5:].split("/", 1)
    return bucket, blob


def _read_lines(path: str) -> Iterable[str]:
    if _is_gcs(path):
        bucket_name, blob_name = _parse_gcs(path)
        client = storage.Client()
        data = client.bucket(bucket_name).blob(blob_name).download_as_bytes()
        return data.decode("utf-8").splitlines()
    return Path(path).read_text(encoding="utf-8").splitlines()


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _upload_file(src: Path, gcs_uri: str) -> None:
    bucket_name, blob_name = _parse_gcs(gcs_uri)
    client = storage.Client()
    client.bucket(bucket_name).blob(blob_name).upload_from_filename(src)


def load_ids(input_path: str) -> list[str]:
    ids = []
    for line in _read_lines(input_path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        record = json.loads(line)
        if "id" not in record:
            raise ValueError("Each JSONL record must contain an 'id' field.")
        ids.append(str(record["id"]))
    return ids


def compute_counts(
    total: int,
    train_count: int | None,
    val_count: int | None,
    test_count: int | None,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    if train_count is not None or val_count is not None or test_count is not None:
        if train_count is None or val_count is None:
            raise ValueError("When using counts, provide both --train-count and --val-count.")
        if test_count is None:
            test_count = total - train_count - val_count
        if train_count + val_count + test_count != total:
            raise ValueError(
                f"Counts must sum to total={total}. "
                f"Got train={train_count}, val={val_count}, test={test_count}."
            )
        return train_count, val_count, test_count

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0.")
    train_count = int(round(total * train_ratio))
    val_count = int(round(total * val_ratio))
    if train_count + val_count > total:
        val_count = max(0, total - train_count)
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits from a dataset.jsonl"
    )
    parser.add_argument("--input", required=True, help="Path to dataset.jsonl (local or gs://)")
    parser.add_argument("--output-dir", default=".", help="Local output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--val-count", type=int, default=None)
    parser.add_argument("--test-count", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--gcs-output-prefix", default="", help="Optional gs:// prefix for uploads")
    args = parser.parse_args()

    ids = load_ids(args.input)
    if len(ids) == 0:
        raise ValueError("No ids found in input JSONL.")

    rng = random.Random(args.seed)
    rng.shuffle(ids)

    train_count, val_count, test_count = compute_counts(
        total=len(ids),
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    train_ids = ids[:train_count]
    val_ids = ids[train_count : train_count + val_count]
    test_ids = ids[train_count + val_count : train_count + val_count + test_count]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.txt"
    val_path = out_dir / "val.txt"
    test_path = out_dir / "test.txt"

    _write_lines(train_path, train_ids)
    _write_lines(val_path, val_ids)
    _write_lines(test_path, test_ids)

    if args.gcs_output_prefix:
        prefix = args.gcs_output_prefix.rstrip("/")
        _upload_file(train_path, f"{prefix}/train.txt")
        _upload_file(val_path, f"{prefix}/val.txt")
        _upload_file(test_path, f"{prefix}/test.txt")

    print(
        f"Created splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
    )
    print(f"Wrote: {train_path}, {val_path}, {test_path}")
    if args.gcs_output_prefix:
        print(f"Uploaded to: {args.gcs_output_prefix.rstrip('/')}/")


if __name__ == "__main__":
    main()
