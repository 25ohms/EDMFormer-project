#!/usr/bin/env python3
"""
GCP Service: Vertex AI Pipelines, Vertex AI Custom Job, Cloud Storage
IAM Roles: roles/aiplatform.user, roles/storage.objectViewer
"""

from kfp.v2 import dsl


@dsl.container_component
def train_op(
    label_path: str,
    split_ids_path: str,
    input_embedding_dir: str,
    config_path: str = "third_party/EDMFormer/configs/SongFormer.yaml",
):
    return dsl.ContainerSpec(
        image="<PLACEHOLDER_REQUIRED_FIELD_MANUAL_FILL>",
        command=["python", "src/task.py"],
        args=["--config-path", config_path],
        env=[
            dsl.EnvVar(name="LABEL_PATH_GCS", value=label_path),
            dsl.EnvVar(name="SPLIT_IDS_PATH_GCS", value=split_ids_path),
            dsl.EnvVar(name="INPUT_EMBEDDING_DIR_GCS", value=input_embedding_dir),
        ],
    )
