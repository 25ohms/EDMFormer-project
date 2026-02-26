#!/usr/bin/env python3
"""
GCP Service: Vertex AI Custom Job, Cloud Storage
IAM Roles: roles/aiplatform.user, roles/storage.objectViewer
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import storage

from config_generator import update_config
import yaml

DEFAULT_EMBEDDING_SUBDIRS = "musicfm_30s,muq_30s,musicfm_420s,muq_420s"


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    bucket, prefix = uri[5:].split("/", 1)
    return bucket, prefix


def download_gcs_prefix(
    client: storage.Client, gcs_prefix: str, dest_dir: Path
) -> Path:
    bucket_name, prefix = parse_gcs_uri(gcs_prefix)
    bucket = client.bucket(bucket_name)
    dest_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix.rstrip("/")
    for blob in bucket.list_blobs(prefix=prefix):
        rel = blob.name[len(prefix) :].lstrip("/")
        if not rel:
            continue
        out_path = dest_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(out_path)
    return dest_dir


def upload_dir_to_gcs(
    client: storage.Client, src_dir: Path, gcs_prefix: str
) -> None:
    bucket_name, prefix = parse_gcs_uri(gcs_prefix)
    bucket = client.bucket(bucket_name)
    prefix = prefix.rstrip("/")
    for path in src_dir.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(src_dir).as_posix()
        blob = bucket.blob(f"{prefix}/{rel}")
        blob.upload_from_filename(path)


def resolve_embedding_dirs(base_or_list: str, subdirs: list[str]) -> list[str]:
    if " " in base_or_list.strip():
        return [x for x in base_or_list.split() if x]
    base = base_or_list.rstrip("/")
    return [f"{base}/{subdir}" for subdir in subdirs]


def _has_flag(args_list: list[str], flag: str) -> bool:
    if flag in args_list:
        return True
    flag_eq = f"{flag}="
    return any(arg.startswith(flag_eq) for arg in args_list)


def ensure_arg(args_list: list[str], flag: str, value: str) -> list[str]:
    if _has_flag(args_list, flag):
        return args_list
    return args_list + [flag, value]


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _default_checkpoint_run_id() -> str:
    explicit = (
        os.environ.get("CHECKPOINT_RUN_ID") or os.environ.get("RUN_ID") or ""
    ).strip()
    if explicit:
        return explicit
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"edmformer-{timestamp}"


def _resolve_checkpoint_dir(checkpoint_dir: str | None) -> str | None:
    if not checkpoint_dir:
        return checkpoint_dir
    run_id = _default_checkpoint_run_id()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    resolved = checkpoint_dir
    for placeholder, value in (
        ("{run_id}", run_id),
        ("<RUN_ID>", run_id),
        ("{timestamp}", timestamp),
        ("<TIMESTAMP>", timestamp),
    ):
        if placeholder in resolved:
            resolved = resolved.replace(placeholder, value)
    if resolved.endswith("/checkpoints") or resolved.endswith("/checkpoints/"):
        resolved = resolved.rstrip("/") + f"/{run_id}"
    return resolved


def _apply_config_overrides(config_path: Path) -> None:
    overrides: dict[str, str] = {}
    for key in (
        "ACCUMULATION_STEPS",
        "EARLY_STOPPING_STEP",
        "WARMUP_MAX_LR",
        "WARMUP_STEPS",
        "TOTAL_STEPS",
        "MAX_STEPS",
        "WEIGHT_DECAY",
        "TRAIN_BATCH_SIZE",
        "EVAL_BATCH_SIZE",
        "LABEL_FOCAL_LOSS_WEIGHT",
        "BOUNDARY_TVLOSS_WEIGHT",
        "LOSS_WEIGHT_SECTION",
        "LOSS_WEIGHT_FUNCTION",
        "LOCAL_MAXIMA_FILTER_SIZE",
        "NUM_NEIGHBORS",
        "DATALOADER_NUM_WORKERS",
        "DATALOADER_PREFETCH_FACTOR",
        "DATALOADER_PERSISTENT_WORKERS",
        "DATALOADER_PIN_MEMORY",
    ):
        value = os.environ.get(key)
        if value is not None and value != "":
            overrides[key] = value

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not overrides and not data:
        return

    if "ACCUMULATION_STEPS" in overrides:
        data["accumulation_steps"] = int(overrides["ACCUMULATION_STEPS"])
        print(f"Override: accumulation_steps={data['accumulation_steps']}")
    if "EARLY_STOPPING_STEP" in overrides:
        data["early_stopping_step"] = int(overrides["EARLY_STOPPING_STEP"])
        print(f"Override: early_stopping_step={data['early_stopping_step']}")

    if "WARMUP_MAX_LR" in overrides:
        data["warmup_max_lr"] = float(overrides["WARMUP_MAX_LR"])
        print(f"Override: warmup_max_lr={data['warmup_max_lr']}")
    if "WARMUP_STEPS" in overrides:
        data["warmup_steps"] = int(overrides["WARMUP_STEPS"])
        print(f"Override: warmup_steps={data['warmup_steps']}")
    if "TOTAL_STEPS" in overrides:
        data["total_steps"] = int(overrides["TOTAL_STEPS"])
        print(f"Override: total_steps={data['total_steps']}")
    if "MAX_STEPS" in overrides:
        if "args" not in data or data["args"] is None:
            data["args"] = {}
        data["args"]["max_steps"] = int(overrides["MAX_STEPS"])
        print(f"Override: max_steps={data['args']['max_steps']}")
    if "WEIGHT_DECAY" in overrides:
        if "optimizer" not in data or data["optimizer"] is None:
            data["optimizer"] = {}
        data["optimizer"]["weight_decay"] = float(overrides["WEIGHT_DECAY"])
        print(f"Override: weight_decay={data['optimizer']['weight_decay']}")

    if "TRAIN_BATCH_SIZE" in overrides:
        if "train_dataloader" not in data or data["train_dataloader"] is None:
            data["train_dataloader"] = {}
        data["train_dataloader"]["batch_size"] = int(overrides["TRAIN_BATCH_SIZE"])
        print(
            f"Override: train_dataloader.batch_size={data['train_dataloader']['batch_size']}"
        )
    if "EVAL_BATCH_SIZE" in overrides:
        if "eval_dataloader" not in data or data["eval_dataloader"] is None:
            data["eval_dataloader"] = {}
        data["eval_dataloader"]["batch_size"] = int(overrides["EVAL_BATCH_SIZE"])
        print(
            f"Override: eval_dataloader.batch_size={data['eval_dataloader']['batch_size']}"
        )

    if "LABEL_FOCAL_LOSS_WEIGHT" in overrides:
        data["label_focal_loss_weight"] = float(
            overrides["LABEL_FOCAL_LOSS_WEIGHT"]
        )
        print(f"Override: label_focal_loss_weight={data['label_focal_loss_weight']}")
    if "BOUNDARY_TVLOSS_WEIGHT" in overrides:
        data["boundary_tvloss_weight"] = float(
            overrides["BOUNDARY_TVLOSS_WEIGHT"]
        )
        print(f"Override: boundary_tvloss_weight={data['boundary_tvloss_weight']}")
    if "LOSS_WEIGHT_SECTION" in overrides:
        data["loss_weight_section"] = float(overrides["LOSS_WEIGHT_SECTION"])
        print(f"Override: loss_weight_section={data['loss_weight_section']}")
    if "LOSS_WEIGHT_FUNCTION" in overrides:
        data["loss_weight_function"] = float(overrides["LOSS_WEIGHT_FUNCTION"])
        print(f"Override: loss_weight_function={data['loss_weight_function']}")
    if "LOCAL_MAXIMA_FILTER_SIZE" in overrides:
        data["local_maxima_filter_size"] = int(
            overrides["LOCAL_MAXIMA_FILTER_SIZE"]
        )
        print(
            f"Override: local_maxima_filter_size={data['local_maxima_filter_size']}"
        )
    if "NUM_NEIGHBORS" in overrides:
        data["num_neighbors"] = int(overrides["NUM_NEIGHBORS"])
        print(f"Override: num_neighbors={data['num_neighbors']}")

    num_workers_override = (
        int(overrides["DATALOADER_NUM_WORKERS"])
        if "DATALOADER_NUM_WORKERS" in overrides
        else None
    )
    prefetch_override = (
        int(overrides["DATALOADER_PREFETCH_FACTOR"])
        if "DATALOADER_PREFETCH_FACTOR" in overrides
        else None
    )
    persistent_override = (
        _is_truthy(overrides["DATALOADER_PERSISTENT_WORKERS"])
        if "DATALOADER_PERSISTENT_WORKERS" in overrides
        else None
    )
    pin_memory_override = (
        _is_truthy(overrides["DATALOADER_PIN_MEMORY"])
        if "DATALOADER_PIN_MEMORY" in overrides
        else None
    )
    for section in ("train_dataloader", "eval_dataloader"):
        if section not in data or data[section] is None:
            data[section] = {}

        if num_workers_override is not None:
            data[section]["num_workers"] = num_workers_override

        effective_num_workers = data[section].get("num_workers", 0)

        if effective_num_workers == 0:
            # PyTorch forbids prefetch_factor when num_workers == 0.
            if "prefetch_factor" in data[section]:
                data[section].pop("prefetch_factor", None)
            if prefetch_override is not None:
                print(
                    f"Ignoring prefetch_factor override for {section} because num_workers=0."
                )
        elif prefetch_override is not None:
            data[section]["prefetch_factor"] = prefetch_override

        if persistent_override is not None:
            data[section]["persistent_workers"] = persistent_override
        elif effective_num_workers == 0:
            data[section]["persistent_workers"] = False

        if pin_memory_override is not None:
            data[section]["pin_memory"] = pin_memory_override

    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vertex AI CustomJob entrypoint")
    parser.add_argument(
        "--config-path",
        default=os.environ.get(
            "SONGFORMER_CONFIG_PATH",
            "third_party/EDMFormer/src/SongFormer/configs/SongFormer.yaml",
        ),
    )
    parser.add_argument(
        "--label-path",
        default=os.environ.get("LABEL_PATH_GCS"),
    )
    parser.add_argument(
        "--split-ids-path",
        default=os.environ.get("SPLIT_IDS_PATH_GCS"),
    )
    parser.add_argument(
        "--eval-split-ids-path",
        default=os.environ.get("EVAL_SPLIT_IDS_PATH_GCS"),
        help="Optional eval split ids path (defaults to --split-ids-path)",
    )
    parser.add_argument(
        "--input-embedding-dir",
        default=os.environ.get("INPUT_EMBEDDING_DIR_GCS"),
    )
    parser.add_argument(
        "--embedding-subdirs",
        default=os.environ.get("EMBEDDING_SUBDIRS", DEFAULT_EMBEDDING_SUBDIRS),
        help="Comma-separated subdir names used when input-embedding-dir is a root",
    )
    parser.add_argument(
        "--prefetch-embeddings",
        default=os.environ.get("PREFETCH_EMBEDDINGS", "0"),
        help="If true, download embedding prefixes from GCS to local-data-dir before training.",
    )
    parser.add_argument(
        "--dataset-type",
        default=os.environ.get("DATASET_TYPE", "EDMFormer"),
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=int(os.environ.get("NUM_GPUS", "1")),
        help="Number of GPUs to use on a single node (GPU backend only).",
    )
    train_backend = os.environ.get("TRAIN_BACKEND", "GPU").upper()
    default_train_script = (
        "src/tpu_train.py"
        if train_backend == "TPU"
        else "third_party/EDMFormer/src/SongFormer/train/train.py"
    )
    parser.add_argument(
        "--train-script",
        default=os.environ.get("SONGFORMER_TRAIN_SCRIPT", default_train_script),
    )
    parser.add_argument(
        "--init-seed",
        type=int,
        default=int(os.environ.get("INIT_SEED", "42")),
        help="Random seed for initialization",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=os.environ.get("CHECKPOINT_DIR_GCS"),
        help="Checkpoint dir (local path or gs://). If gs://, will sync to local and upload after training.",
    )
    parser.add_argument(
        "--local-data-dir",
        default=os.environ.get("LOCAL_DATA_DIR", "/tmp/edmformer-data"),
        help="Local staging directory inside the container",
    )
    parser.add_argument(
        "--train-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments passed through to the SongFormer training script",
    )
    args = parser.parse_args()

    if not args.label_path or not args.split_ids_path or not args.input_embedding_dir:
        raise SystemExit(
            "Missing required GCS paths. Set LABEL_PATH_GCS, SPLIT_IDS_PATH_GCS, "
            "INPUT_EMBEDDING_DIR_GCS."
        )

    args.checkpoint_dir = _resolve_checkpoint_dir(args.checkpoint_dir)
    if _has_flag(args.train_args, "--checkpoint_dir") and args.checkpoint_dir:
        raise SystemExit(
            "Provide checkpoint dir via --checkpoint-dir or --train-args --checkpoint_dir, not both."
        )

    eval_split_ids_path = args.eval_split_ids_path or args.split_ids_path

    local_root = Path(args.local_data_dir)
    local_root.mkdir(parents=True, exist_ok=True)

    client = storage.Client()

    label_path = args.label_path
    train_split_path = args.split_ids_path
    eval_split_path = eval_split_ids_path

    embedding_subdirs_raw = args.embedding_subdirs.strip() or DEFAULT_EMBEDDING_SUBDIRS
    embedding_subdirs = [
        x.strip() for x in embedding_subdirs_raw.split(",") if x.strip()
    ]
    embedding_inputs = resolve_embedding_dirs(args.input_embedding_dir, embedding_subdirs)

    prefetch_embeddings = _is_truthy(args.prefetch_embeddings)
    if prefetch_embeddings:
        local_embeddings_root = local_root / "embeddings"
        local_embedding_inputs = []
        for embd in embedding_inputs:
            if not embd.startswith("gs://"):
                local_embedding_inputs.append(embd)
                continue
            _, prefix = parse_gcs_uri(embd)
            subdir_name = Path(prefix.rstrip("/")).name or "embeddings"
            local_dir = local_embeddings_root / subdir_name
            if local_dir.exists() and any(local_dir.rglob("*.npy")):
                print(f"Using cached embeddings for {embd} at {local_dir}")
            else:
                print(f"Prefetching embeddings from {embd} to {local_dir} ...")
                download_gcs_prefix(client, embd, local_dir)
            local_embedding_inputs.append(str(local_dir))
        embedding_inputs = local_embedding_inputs

    local_input_embedding_dir = " ".join(embedding_inputs)

    config_path = Path(args.config_path).resolve()

    update_config(
        config_path=config_path,
        label_path=str(label_path),
        train_split_ids_path=str(train_split_path),
        eval_split_ids_path=str(eval_split_path),
        input_embedding_dir=local_input_embedding_dir,
        dataset_type=args.dataset_type,
    )
    _apply_config_overrides(config_path)

    train_args = list(args.train_args)
    train_args = ensure_arg(train_args, "--config", str(config_path))
    train_args = ensure_arg(train_args, "--init_seed", str(args.init_seed))
    cv_folds = os.environ.get("CV_FOLDS")
    if cv_folds:
        train_args = ensure_arg(train_args, "--cv_folds", str(cv_folds))
    cv_seed = os.environ.get("CV_SEED")
    if cv_seed:
        train_args = ensure_arg(train_args, "--cv_seed", str(cv_seed))

    local_checkpoint_dir = None
    checkpoint_gcs = None
    if args.checkpoint_dir:
        if args.checkpoint_dir.startswith("gs://"):
            checkpoint_gcs = args.checkpoint_dir
            _, prefix = parse_gcs_uri(args.checkpoint_dir)
            checkpoint_local = (
                local_root / "checkpoints" / Path(prefix.rstrip("/")).name
            )
            if not checkpoint_local.exists():
                checkpoint_local.mkdir(parents=True, exist_ok=True)
            download_gcs_prefix(client, args.checkpoint_dir, checkpoint_local)
            local_checkpoint_dir = checkpoint_local
        else:
            local_checkpoint_dir = Path(args.checkpoint_dir)

    if local_checkpoint_dir is not None:
        train_args = ensure_arg(train_args, "--checkpoint_dir", str(local_checkpoint_dir))

    repo_root = Path(__file__).resolve().parents[1]
    train_script = Path(args.train_script).resolve()
    if not train_script.exists() and not Path(args.train_script).is_absolute():
        candidate = (repo_root / args.train_script).resolve()
        if candidate.exists():
            train_script = candidate
    if not train_script.exists():
        raise SystemExit(f"Train script not found: {train_script}")
    if train_backend == "TPU" and "tpu_train.py" not in train_script.as_posix():
        raise SystemExit(
            "TRAIN_BACKEND=TPU requires --train-script to point to src/tpu_train.py."
        )
    if train_backend == "TPU":
        try:
            import torch_xla  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise SystemExit(
                "TRAIN_BACKEND=TPU requires an XLA runtime. "
                "Use docker/training_tpu.Dockerfile (runtime-xla base image)."
            ) from exc
    if train_backend == "GPU" and args.num_gpus < 1:
        raise SystemExit("--num-gpus must be >= 1 for GPU training.")
    if train_backend == "GPU" and args.num_gpus > 1:
        try:
            import torch

            gpu_count = torch.cuda.device_count()
            if gpu_count and args.num_gpus > gpu_count:
                raise SystemExit(
                    f"Requested {args.num_gpus} GPUs but only {gpu_count} detected."
                )
        except Exception:
            pass
    workdir = (repo_root / "third_party/EDMFormer/src/SongFormer").resolve()
    src_root = repo_root / "src"

    env = os.environ.copy()
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env["PYTHONPATH"] = f"{src_root}:{workdir}:{env.get('PYTHONPATH', '')}"
    # If no W&B key is provided, default to disabled to avoid no-tty failures.
    if not env.get("WANDB_API_KEY") and not _is_truthy(env.get("WANDB_FORCE_ENABLE")):
        env.setdefault("WANDB_DISABLED", "true")

    cmd = [sys.executable, str(train_script)] + train_args
    if train_backend == "GPU" and args.num_gpus > 1:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(args.num_gpus),
            str(train_script),
        ] + train_args
    print(f"Launching training: {' '.join(cmd)}")
    run_error = None
    try:
        subprocess.run(cmd, check=True, cwd=workdir, env=env)
    except subprocess.CalledProcessError as exc:
        run_error = exc
    else:
        if _is_truthy(os.environ.get("RUN_TEST_EVAL", "0")):
            eval_cmd = [sys.executable, str(repo_root / "src" / "test.py")]
            eval_cmd += ["--config", str(config_path)]
            if local_checkpoint_dir is not None:
                eval_cmd += ["--checkpoint-dir", str(local_checkpoint_dir)]
            elif args.checkpoint_dir:
                eval_cmd += ["--checkpoint-dir", str(args.checkpoint_dir)]
            print(f"Running test evaluation: {' '.join(eval_cmd)}")
            subprocess.run(eval_cmd, check=True, cwd=repo_root, env=env)
    finally:
        if checkpoint_gcs and local_checkpoint_dir is not None:
            try:
                upload_dir_to_gcs(client, local_checkpoint_dir, checkpoint_gcs)
            except Exception as exc:
                print(f"Failed to upload checkpoints to {checkpoint_gcs}: {exc}")

    if run_error is not None:
        raise run_error


if __name__ == "__main__":
    main()
