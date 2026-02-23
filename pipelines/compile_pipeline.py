#!/usr/bin/env python3
"""
GCP Service: Vertex AI Pipelines
IAM Roles: roles/aiplatform.user
"""

import argparse

from kfp.v2 import compiler, dsl

from components.ingest_op import ingest_op
from components.preprocess_op import preprocess_op
from components.train_op import train_op


@dsl.pipeline(name="edmformer-retrain")
def edmformer_pipeline(
    labels_jsonl: str,
    bucket_name: str,
    split_ids_path: str = "split_ids.txt",
    eval_split_ids_path: str = "",
    embeddings_root: str = "embeddings",
    preprocess_dummy_flag: str = "",
):
    split_ids_gcs = f"gs://{bucket_name}/{split_ids_path}"
    eval_split_ids_gcs = (
        f"gs://{bucket_name}/{eval_split_ids_path}" if eval_split_ids_path else ""
    )
    embeddings_gcs = f"gs://{bucket_name}/{embeddings_root}"

    ingest_task = ingest_op(
        labels_jsonl=labels_jsonl,
        bucket_name=bucket_name,
        split_ids_out=split_ids_gcs,
    )

    preprocess_task = preprocess_op(
        bucket_name=bucket_name,
        split_ids_path=split_ids_gcs,
        output_root=embeddings_root,
        dummy_flag=preprocess_dummy_flag,
    )
    preprocess_task.after(ingest_task)

    train_task = train_op(
        label_path=labels_jsonl,
        split_ids_path=split_ids_gcs,
        eval_split_ids_path=eval_split_ids_gcs,
        input_embedding_dir=embeddings_gcs,
    )
    train_task.after(preprocess_task)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile Vertex AI pipeline JSON")
    parser.add_argument("--output", default="pipeline.json")
    parser.add_argument(
        "--pipeline-root",
        required=True,
        help="GCS path for pipeline root (e.g., gs://bucket/pipeline-root)",
    )
    args = parser.parse_args()

    compiler.Compiler().compile(
        pipeline_func=edmformer_pipeline,
        package_path=args.output,
        pipeline_root=args.pipeline_root,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
