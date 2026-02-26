#!/usr/bin/env python3
"""
Evaluate a checkpoint on train and test splits and report all metrics.
Supports local paths and gs:// checkpoint URIs.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import hydra
import pandas as pd
import torch
from ema_pytorch import EMA
from google.cloud import storage
from omegaconf import OmegaConf


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    bucket, prefix = uri[5:].split("/", 1)
    return bucket, prefix


def download_gcs_blob(uri: str, dest_path: Path) -> Path:
    bucket_name, blob_name = parse_gcs_uri(uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    bucket.blob(blob_name).download_to_filename(dest_path)
    return dest_path


def resolve_embedding_dirs(base_or_list: str, subdirs: list[str]) -> list[str]:
    if " " in base_or_list.strip():
        return [x for x in base_or_list.split() if x]
    base = base_or_list.rstrip("/")
    return [f"{base}/{subdir}" for subdir in subdirs]


def build_dataset(hparams, split_ids_path, label_path, input_embedding_dir, dataset_type):
    dataset_cfg = OmegaConf.create(
        OmegaConf.to_container(hparams.train_dataset, resolve=True)
    )
    dataset_cfg["dataset_abstracts"] = [
        {
            "internal_tmp_id": dataset_type,
            "dataset_type": dataset_type,
            "input_embedding_dir": input_embedding_dir,
            "label_path": label_path,
            "split_ids_path": split_ids_path,
            "multiplier": 1,
        }
    ]
    return hydra.utils.instantiate(dataset_cfg)


def evaluate_dataset(model_ema, data_loader, device):
    model_ema.ema_model.eval()
    results_by_dataset = {}

    with torch.no_grad():
        for batch in data_loader:
            if batch is None:
                continue
            batch = {
                key: (val.to(device) if isinstance(val, torch.Tensor) else val)
                for key, val in batch.items()
            }
            dataset_id = int(batch["dataset_ids"].item())
            result = model_ema.ema_model.infer_with_metrics(batch, prefix="valid_")
            results_by_dataset.setdefault(dataset_id, []).append(result)

    flat_result = {}
    for dataset_id, result_list in results_by_dataset.items():
        df = pd.DataFrame(result_list)
        avg_metrics = df.mean().to_dict()
        for k, v in avg_metrics.items():
            flat_result[f"dataset_{dataset_id}_{k}"] = v

    all_results = [res for results in results_by_dataset.values() for res in results]
    if all_results:
        overall_df = pd.DataFrame(all_results)
        overall_metrics = overall_df.mean().to_dict()
        for k, v in overall_metrics.items():
            flat_result[f"overall_{k}"] = v

    return flat_result


def prefix_dict(d, prefix: str):
    return {prefix + key: value for key, value in d.items()}


def resolve_checkpoint(args) -> Path:
    if args.checkpoint:
        if args.checkpoint.startswith("gs://"):
            tmp_dir = Path(tempfile.mkdtemp(prefix="edmformer-ckpt-"))
            return download_gcs_blob(args.checkpoint, tmp_dir / Path(args.checkpoint).name)
        return Path(args.checkpoint)

    if not args.checkpoint_dir:
        raise SystemExit("Provide --checkpoint or --checkpoint-dir.")

    if args.checkpoint_dir.startswith("gs://"):
        tmp_dir = Path(tempfile.mkdtemp(prefix="edmformer-ckpt-dir-"))
        checkpoint_list_uri = args.checkpoint_dir.rstrip("/") + "/checkpoint"
        checkpoint_list = download_gcs_blob(
            checkpoint_list_uri, tmp_dir / "checkpoint"
        ).read_text(encoding="utf-8").strip()
        ckpt_uri = args.checkpoint_dir.rstrip("/") + "/" + checkpoint_list
        return download_gcs_blob(ckpt_uri, tmp_dir / checkpoint_list)

    checkpoint_list = Path(args.checkpoint_dir) / "checkpoint"
    if not checkpoint_list.exists():
        raise SystemExit(f"Checkpoint file not found: {checkpoint_list}")
    ckpt_name = checkpoint_list.read_text(encoding="utf-8").strip()
    return Path(args.checkpoint_dir) / ckpt_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint metrics")
    parser.add_argument(
        "--config",
        default=os.environ.get(
            "SONGFORMER_CONFIG_PATH",
            "third_party/EDMFormer/src/SongFormer/configs/SongFormer.yaml",
        ),
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument(
        "--label-path", default=os.environ.get("LABEL_PATH_GCS", "")
    )
    parser.add_argument(
        "--input-embedding-dir",
        default=os.environ.get("INPUT_EMBEDDING_DIR_GCS", ""),
    )
    parser.add_argument(
        "--embedding-subdirs",
        default=os.environ.get("EMBEDDING_SUBDIRS", ""),
    )
    parser.add_argument(
        "--train-split-ids-path",
        default=os.environ.get("SPLIT_IDS_PATH_GCS", ""),
    )
    parser.add_argument(
        "--test-split-ids-path",
        default=os.environ.get("TEST_IDS_PATH_GCS", ""),
    )
    parser.add_argument(
        "--dataset-type",
        default=os.environ.get("DATASET_TYPE", ""),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    if not args.label_path or not args.input_embedding_dir:
        raise SystemExit(
            "Missing LABEL_PATH_GCS or INPUT_EMBEDDING_DIR_GCS (or --label-path/--input-embedding-dir)."
        )
    if not args.train_split_ids_path:
        raise SystemExit(
            "Missing SPLIT_IDS_PATH_GCS (or --train-split-ids-path)."
        )
    if not args.test_split_ids_path:
        raise SystemExit(
            "Missing TEST_IDS_PATH_GCS (or --test-split-ids-path)."
        )

    hparams = OmegaConf.load(args.config)

    dataset_type = args.dataset_type
    if not dataset_type:
        try:
            dataset_type = (
                hparams.train_dataset.dataset_abstracts[0]["dataset_type"]
            )
        except Exception:
            dataset_type = "EDMFormer"

    embedding_dir = args.input_embedding_dir
    if args.embedding_subdirs:
        subdirs = [x.strip() for x in args.embedding_subdirs.split(",") if x.strip()]
        embedding_dir = " ".join(resolve_embedding_dirs(embedding_dir, subdirs))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = resolve_checkpoint(args)

    module = __import__(f"models.{hparams.args.model_name}", fromlist=["Model"])
    Model = getattr(module, "Model")
    model = Model(hparams)
    model.to(device)
    model_ema = EMA(model, include_online_model=False, **hparams.ema_kwargs)
    model_ema.to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    if "model_ema" in checkpoint:
        model_ema.load_state_dict(checkpoint["model_ema"])
    else:
        model_ema.ema_model.load_state_dict(model.state_dict())

    eval_loader_kwargs = dict(hparams.eval_dataloader)
    train_dataset = build_dataset(
        hparams,
        args.train_split_ids_path,
        args.label_path,
        embedding_dir,
        dataset_type,
    )
    test_dataset = build_dataset(
        hparams,
        args.test_split_ids_path,
        args.label_path,
        embedding_dir,
        dataset_type,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, **eval_loader_kwargs, collate_fn=train_dataset.collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, **eval_loader_kwargs, collate_fn=test_dataset.collate_fn
    )

    train_metrics = evaluate_dataset(model_ema, train_loader, device)
    test_metrics = evaluate_dataset(model_ema, test_loader, device)

    merged = {}
    merged.update(prefix_dict(train_metrics, "train_"))
    merged.update(prefix_dict(test_metrics, "test_"))

    print("Train metrics:")
    for k in sorted(train_metrics.keys()):
        print(f"  {k}: {train_metrics[k]}")
    print("Test metrics:")
    for k in sorted(test_metrics.keys()):
        print(f"  {k}: {test_metrics[k]}")

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
