#!/usr/bin/env python3
import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np

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


def list_gcs_npy_stems(prefix_uri: str) -> set[str]:
    client = get_gcs_client()
    bucket_name, prefix = parse_gcs_uri(prefix_uri)
    bucket = client.bucket(bucket_name)
    prefix = prefix.rstrip("/") + "/"
    ids: set[str] = set()
    for blob in bucket.list_blobs(prefix=prefix):
        if not blob.name.endswith(".npy"):
            continue
        ids.add(Path(blob.name).stem)
    return ids


def list_local_npy_stems(dir_path: str) -> set[str]:
    ids: set[str] = set()
    for path in Path(dir_path).glob("*.npy"):
        ids.add(path.stem)
    return ids


def load_npy(path: str) -> np.ndarray:
    if is_gcs_path(path):
        client = get_gcs_client()
        bucket, prefix = parse_gcs_uri(path)
        data = client.bucket(bucket).blob(prefix).download_as_bytes()
        return np.load(BytesIO(data))
    return np.load(path)


@dataclass
class LabelInfo:
    times: list[float]
    labels: list[str]
    issues: list[str]


def load_labels(label_path: str) -> dict[str, LabelInfo]:
    text = read_text(label_path)
    label_map: dict[str, LabelInfo] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        data = json.loads(line)
        data_id = data.get("id")
        labels = data.get("labels", [])
        times = [float(x[0]) for x in labels]
        label_names = [x[1] for x in labels]
        issues: list[str] = []
        if not times:
            issues.append("empty_labels")
        else:
            diffs = np.diff(times)
            if np.any(diffs <= 0):
                issues.append("non_increasing_times")
            if label_names and label_names[-1] != "end":
                issues.append("missing_end_label")
        label_map[data_id] = LabelInfo(times=times, labels=label_names, issues=issues)
    return label_map


def parse_split_ids(split_path: str) -> set[str]:
    text = read_text(split_path)
    ids = set()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        ids.add(line)
    return ids


def parse_embedding_dirs(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []
    if " " in raw:
        return [x for x in raw.split() if x]
    return [x for x in raw.split(",") if x]


def extract_base_and_start(stem: str) -> tuple[str | None, int | None]:
    parts = stem.split("_")
    if len(parts) < 2:
        return None, None
    try:
        start = int(parts[-1])
    except ValueError:
        return None, None
    return "_".join(parts[:-1]), start


def audit(
    label_path: str,
    split_ids_path: str,
    embedding_dirs: list[str],
    slice_dur: float,
    eps: float,
    check_embeddings: bool,
    max_embeddings: int,
    output_path: str | None,
) -> int:
    split_ids = parse_split_ids(split_ids_path)
    labels = load_labels(label_path)

    label_issue_counts = Counter()
    for info in labels.values():
        label_issue_counts.update(info.issues)

    embedding_sets: list[set[str]] = []
    for embd in embedding_dirs:
        if is_gcs_path(embd):
            embedding_sets.append(list_gcs_npy_stems(embd))
        else:
            embedding_sets.append(list_local_npy_stems(embd))

    union_stems = set().union(*embedding_sets) if embedding_sets else set()

    issues_by_id: dict[str, list[str]] = defaultdict(list)
    details_by_id: dict[str, dict] = {}

    for stem in sorted(union_stems):
        base_id, start_time = extract_base_and_start(stem)
        if base_id is None or start_time is None:
            issues_by_id[stem].append("bad_stem_format")
            details_by_id[stem] = {"stem": stem}
            continue
        if base_id not in split_ids:
            continue

        missing_dirs = [
            embedding_dirs[i]
            for i, s in enumerate(embedding_sets)
            if stem not in s
        ]
        if missing_dirs:
            issues_by_id[stem].append("missing_embedding_dir")

        label_info = labels.get(base_id)
        if label_info is None:
            issues_by_id[stem].append("missing_labels")
            details_by_id[stem] = {
                "stem": stem,
                "base_id": base_id,
                "start": start_time,
                "missing_dirs": missing_dirs,
            }
            continue

        if label_info.issues:
            issues_by_id[stem].extend(label_info.issues)

        if label_info.times:
            local_times = np.array(label_info.times, dtype=float) - float(start_time)
            time_L = max(0.0, float(local_times.min()))
            time_R = min(float(slice_dur), float(local_times.max()))
            keep = (time_L + eps < local_times) & (local_times < time_R - eps)
            if time_R <= time_L + eps:
                issues_by_id[stem].append("empty_slice_window")
            if keep.sum() <= 0:
                issues_by_id[stem].append("no_boundaries_in_slice")

        details_by_id[stem] = {
            "stem": stem,
            "base_id": base_id,
            "start": start_time,
            "missing_dirs": missing_dirs,
        }

    checked = 0
    if check_embeddings:
        for stem in sorted(union_stems):
            base_id, start_time = extract_base_and_start(stem)
            if base_id is None or start_time is None:
                continue
            if base_id not in split_ids:
                continue
            if max_embeddings > 0 and checked >= max_embeddings:
                break
            checked += 1
            shapes = []
            for embd_dir in embedding_dirs:
                path = embd_dir.rstrip("/") + "/" + stem + ".npy"
                try:
                    arr = load_npy(path)
                except Exception:
                    issues_by_id[stem].append("embedding_load_failed")
                    continue
                if not np.isfinite(arr).all():
                    issues_by_id[stem].append("embedding_non_finite")
                shapes.append(tuple(arr.shape))
            if shapes and len(set(shapes)) > 1:
                issues_by_id[stem].append("embedding_shape_mismatch")

    output_lines = []
    issue_counts = Counter()
    for stem, issues in issues_by_id.items():
        issue_counts.update(issues)
        if not issues:
            continue
        row = details_by_id.get(stem, {"stem": stem})
        row["issues"] = sorted(set(issues))
        output_lines.append(row)

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in output_lines:
                f.write(json.dumps(row) + "\n")

    print("==== Dataset audit summary ====")
    print(f"Total split ids: {len(split_ids)}")
    print(f"Label entries: {len(labels)}")
    print(f"Embedding stems (union): {len(union_stems)}")
    print(f"Issues found: {len(output_lines)}")
    if issue_counts:
        print("Issue counts:")
        for k, v in issue_counts.most_common():
            print(f"  {k}: {v}")
    if label_issue_counts:
        print("Label-level issues:")
        for k, v in label_issue_counts.most_common():
            print(f"  {k}: {v}")

    return 1 if output_lines else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit EDMFormer dataset splits.")
    parser.add_argument("--labels", required=True, help="Path to dataset.jsonl")
    parser.add_argument("--split-ids", required=True, help="Split IDs file")
    parser.add_argument(
        "--embedding-dirs",
        required=True,
        help="Embedding dirs (space- or comma-separated).",
    )
    parser.add_argument("--slice-dur", type=float, default=420.0)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--check-embeddings", action="store_true")
    parser.add_argument("--max-embeddings", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    out_path = args.output.strip()
    if not out_path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out_path = f"reports/dataset_audit_{ts}.jsonl"

    embedding_dirs = parse_embedding_dirs(args.embedding_dirs)
    if not embedding_dirs:
        raise SystemExit("No embedding dirs provided.")

    return audit(
        label_path=args.labels,
        split_ids_path=args.split_ids,
        embedding_dirs=embedding_dirs,
        slice_dur=args.slice_dur,
        eps=args.eps,
        check_embeddings=args.check_embeddings,
        max_embeddings=args.max_embeddings,
        output_path=out_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
