#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

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


def parse_ids_from_split(text: str) -> list[str]:
    ids = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        ids.append(line)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split a train set into train/val (optionally deriving train from labels minus test)."
    )
    parser.add_argument("--train-ids", help="train.txt path (source ids)")
    parser.add_argument("--labels", help="dataset.jsonl path (used with --test-ids)")
    parser.add_argument("--test-ids", help="test.txt path (used with --labels)")
    parser.add_argument("--val-count", type=int, default=10, help="Number of ids for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--train-out", required=True, help="train.txt output path")
    parser.add_argument("--val-out", required=True, help="val.txt output path")
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort outputs (default preserves shuffled split order)",
    )
    args = parser.parse_args()

    if args.train_ids:
        train_text = read_text(args.train_ids)
        all_train_ids = parse_ids_from_split(train_text)
    else:
        if not args.labels or not args.test_ids:
            raise SystemExit("Provide --train-ids or both --labels and --test-ids.")
        labels_text = read_text(args.labels)
        test_text = read_text(args.test_ids)
        all_ids = parse_ids_from_labels(labels_text)
        test_ids = set(parse_ids_from_split(test_text))
        all_train_ids = [x for x in all_ids if x not in test_ids]

    if args.val_count < 1 or args.val_count >= len(all_train_ids):
        raise SystemExit(
            f"--val-count must be >=1 and < total train ids ({len(all_train_ids)})"
        )

    rng = random.Random(args.seed)
    shuffled = list(all_train_ids)
    rng.shuffle(shuffled)
    val_ids = shuffled[: args.val_count]
    train_ids = shuffled[args.val_count :]

    if args.sort:
        train_ids = sorted(set(train_ids))
        val_ids = sorted(set(val_ids))

    write_text(args.train_out, "\n".join(train_ids) + "\n")
    write_text(args.val_out, "\n".join(val_ids) + "\n")

    print("==== Train/Val split generation ====")
    print(f"Total train IDs (source): {len(all_train_ids)}")
    print(f"Val IDs: {len(val_ids)}")
    print(f"Train IDs: {len(train_ids)}")
    print(f"Wrote train: {args.train_out}")
    print(f"Wrote val: {args.val_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
