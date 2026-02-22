#!/usr/bin/env python3
"""
GCP Service: None (local file generation only)
IAM Roles: None
"""

import argparse
import json
from pathlib import Path


def read_ids(labels_jsonl_path: Path) -> list[str]:
    ids: list[str] = []
    with labels_jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}") from exc
            if "id" not in record:
                raise ValueError(f"Missing 'id' field on line {line_no}")
            ids.append(str(record["id"]))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate split_ids.txt from labels.jsonl.")
    parser.add_argument("--labels-jsonl", required=True, help="Path to labels.jsonl")
    parser.add_argument(
        "--output",
        default="split_ids.txt",
        help="Output path for split_ids.txt",
    )
    args = parser.parse_args()

    labels_path = Path(args.labels_jsonl)
    output_path = Path(args.output)

    ids = read_ids(labels_path)
    output_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    print(f"Wrote {len(ids)} ids to {output_path}")


if __name__ == "__main__":
    main()
