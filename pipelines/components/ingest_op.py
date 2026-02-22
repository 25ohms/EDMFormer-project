#!/usr/bin/env python3
"""
GCP Service: Vertex AI Pipelines, Google Drive API, Cloud Storage
IAM Roles: roles/aiplatform.user, roles/drive.reader, roles/storage.objectAdmin
"""

from kfp.v2 import dsl


@dsl.container_component
def ingest_op(
    labels_jsonl: str,
    bucket_name: str,
    split_ids_out: str,
    drive_sa_json: str = "",
    skip_existing_flag: str = "--skip-existing",
):
    return dsl.ContainerSpec(
        image="<PLACEHOLDER_REQUIRED_FIELD_MANUAL_FILL>",
        command=["python", "ingestion/migrate_drive_to_gcs.py"],
        args=[
            "--labels-jsonl",
            labels_jsonl,
            "--bucket",
            bucket_name,
            "--split-ids-out",
            split_ids_out,
            skip_existing_flag,
            "--drive-sa-json",
            drive_sa_json,
        ],
    )
