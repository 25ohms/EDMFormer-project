#!/usr/bin/env python3
"""
GCP Service: Vertex AI Pipelines, Cloud Storage
IAM Roles: roles/aiplatform.user, roles/storage.objectAdmin
"""

from kfp.v2 import dsl


@dsl.container_component
def preprocess_op(
    bucket_name: str,
    split_ids_path: str,
    output_root: str,
    dummy_flag: str = "",
):
    command = (
        "python preprocessing/extract_muq.py "
        "--bucket {bucket} --split-ids {split_ids} --output-root {output_root} {dummy} "
        "&& python preprocessing/extract_musicfm.py "
        "--bucket {bucket} --split-ids {split_ids} --output-root {output_root} {dummy}"
    ).format(
        bucket=bucket_name,
        split_ids=split_ids_path,
        output_root=output_root,
        dummy=dummy_flag,
    )

    return dsl.ContainerSpec(
        image="<PLACEHOLDER_REQUIRED_FIELD_MANUAL_FILL>",
        command=["/bin/bash", "-c"],
        args=[command],
    )
