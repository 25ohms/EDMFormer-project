#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Iterable

try:
    from google.cloud import storage
except Exception:  # pragma: no cover
    storage = None


def is_gcs_path(path: str) -> bool:
    return path.startswith("gs://")


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    bucket, prefix = uri[5:].split("/", 1)
    return bucket, prefix


def get_gcs_client():
    if storage is None:
        raise RuntimeError(
            "google-cloud-storage is not available. Install it or run in an environment "
            "with the dependency available."
        )
    return storage.Client()


def read_text(path: str) -> str:
    if is_gcs_path(path):
        client = get_gcs_client()
        bucket, prefix = parse_gcs_uri(path)
        return client.bucket(bucket).blob(prefix).download_as_text()
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str, text: str) -> None:
    if is_gcs_path(path):
        client = get_gcs_client()
        bucket, prefix = parse_gcs_uri(path)
        client.bucket(bucket).blob(prefix).upload_from_string(text)
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def parse_ids_from_labels(labels_text: str) -> list[str]:
    ids: list[str] = []
    for line in labels_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        data = json.loads(line)
        ids.append(str(data["id"]))
    return ids


def parse_ids_from_split(text: str) -> set[str]:
    ids = set()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        ids.add(line)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate train.txt from labels.jsonl minus test.txt."
    )
    parser.add_argument("--labels", required=True, help="dataset.jsonl path")
    parser.add_argument("--test-ids", required=True, help="test.txt path")
    parser.add_argument("--output", required=True, help="train.txt output path")
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort output ids (default preserves labels.jsonl order)",
    )
    args = parser.parse_args()

    labels_text = read_text(args.labels)
    test_text = read_text(args.test_ids)

    all_ids = parse_ids_from_labels(labels_text)
    test_ids = parse_ids_from_split(test_text)

    train_ids = [x for x in all_ids if x not in test_ids]
    if args.sort:
        train_ids = sorted(set(train_ids))

    output_text = "\n".join(train_ids) + "\n"
    write_text(args.output, output_text)

    print("==== Train split generation ====")
    print(f"Total label IDs: {len(set(all_ids))}")
    print(f"Test IDs: {len(test_ids)}")
    print(f"Train IDs: {len(train_ids)}")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
