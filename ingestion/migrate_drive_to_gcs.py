#!/usr/bin/env python3
"""
GCP Service: Google Drive API, Cloud Storage
IAM Roles: roles/drive.reader (or Drive Viewer), roles/storage.objectAdmin
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Optional
import google.auth

from google.cloud import storage

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    from google.oauth2 import service_account
except ImportError as exc:  # pragma: no cover - best effort guidance
    raise SystemExit(
        "Missing Google Drive dependencies. Install google-api-python-client and google-auth."
    ) from exc

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def load_labels(labels_jsonl_path: Path) -> list[dict]:
    records: list[dict] = []
    with labels_jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}") from exc
            if "id" not in record or "file_path" not in record:
                raise ValueError(f"Missing required fields on line {line_no}; need 'id' and 'file_path'")
            records.append(record)
    return records


def get_drive_service(unused_arg=None):
    # This automatically finds the credentials you created in Step 1
    credentials, project = google.auth.default()
    service = build("drive", "v3", credentials=credentials)
    return service


def resolve_path_to_file_id(service, file_path: str) -> str:
    parts = [p for p in file_path.split("/") if p]
    if not parts:
        raise ValueError("file_path is empty")

    parent_id = "root"
    for idx, part in enumerate(parts):
        is_last = idx == len(parts) - 1
        mime_filter = (
            " and mimeType = 'application/vnd.google-apps.folder'" if not is_last else ""
        )
        query = (
            f"name = '{part}' and '{parent_id}' in parents and trashed = false{mime_filter}"
        )
        response = (
            service.files()
            .list(
                q=query,
                fields="files(id, name)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        files = response.get("files", [])
        if not files:
            raise FileNotFoundError(f"Drive path segment not found: {part}")
        parent_id = files[0]["id"]
    return parent_id


def download_drive_file(service, file_id: str) -> io.BytesIO:
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return fh


def upload_to_gcs(bucket_name: str, blob_name: str, data: io.BytesIO) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_file(data, content_type="audio/mpeg")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate Google Drive audio to GCS using labels.jsonl"
    )
    parser.add_argument("--labels-jsonl", required=True,
                        help="Path to labels.jsonl")
    parser.add_argument("--bucket", required=True,
                        help="Target GCS bucket name")
    parser.add_argument(
        "--drive-sa-json",
        default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
        help="Service account JSON for Drive API access",
    )
    parser.add_argument(
        "--split-ids-out",
        default="split_ids.txt",
        help="Output path for split_ids.txt",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip upload if gs://bucket/audio/<id>.mp3 already exists",
    )
    args = parser.parse_args()

    labels_path = Path(args.labels_jsonl)
    records = load_labels(labels_path)

    drive_service = get_drive_service(args.drive_sa_json)
    storage_client = storage.Client()
    bucket = storage_client.bucket(args.bucket)

    split_ids: list[str] = []

    for record in records:
        audio_id = str(record["id"])
        split_ids.append(audio_id)
        gcs_blob_name = f"audio/{audio_id}.mp3"

        if args.skip_existing and bucket.blob(gcs_blob_name).exists():
            print(f"Skipping existing gs://{args.bucket}/{gcs_blob_name}")
            continue

        drive_file_id = resolve_path_to_file_id(
            drive_service, record["file_path"])
        audio_bytes = download_drive_file(drive_service, drive_file_id)
        upload_to_gcs(args.bucket, gcs_blob_name, audio_bytes)
        print(f"Uploaded gs://{args.bucket}/{gcs_blob_name}")

    Path(args.split_ids_out).write_text(
        "\n".join(split_ids) + "\n", encoding="utf-8")
    print(f"Wrote {len(split_ids)} ids to {args.split_ids_out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
