#!/usr/bin/env python3
"""
GCP Service: Cloud Storage
IAM Roles: roles/storage.objectAdmin (write embeddings), roles/storage.objectViewer (read audio)
"""

import argparse
import io
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from google.cloud import storage
import torch
import torchaudio
from huggingface_hub import snapshot_download

from muq import MuQ

WIN_SIZE = 30
HOP_SIZE = 30
WRAP_SIZE = 420
TARGET_SAMPLE_RATE = 24000


def normalize_gcs_uri(uri: str) -> str:
    if uri.startswith("gs:/") and not uri.startswith("gs://"):
        return "gs://" + uri[4:]
    return uri


def read_text_uri(uri: str, client: storage.Client | None = None) -> str:
    uri = normalize_gcs_uri(uri)
    if uri.startswith("gs://"):
        bucket_name, blob_name = parse_gcs_uri(uri)
        client = client or storage.Client()
        blob = client.bucket(bucket_name).blob(blob_name)
        return blob.download_as_text(encoding="utf-8")
    return Path(uri).read_text(encoding="utf-8")


def read_ids(split_ids_path: str | None, labels_jsonl_path: str | None) -> list[str]:
    if split_ids_path:
        text = read_text_uri(split_ids_path)
        return [line.strip() for line in text.splitlines() if line.strip()]
    if labels_jsonl_path:
        text = read_text_uri(labels_jsonl_path)
        ids: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ids.append(str(record["id"]))
        return ids
    raise ValueError("Provide --split-ids or --labels-jsonl")


def download_gcs_blob(bucket: storage.Bucket, blob_name: str) -> Path:
    blob = bucket.blob(blob_name)
    if not blob.exists():
        raise FileNotFoundError(f"GCS blob not found: gs://{bucket.name}/{blob_name}")
    data = blob.download_as_bytes()
    suffix = Path(blob_name).suffix or ".mp3"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    uri = normalize_gcs_uri(uri)
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    bucket, prefix = uri[5:].split("/", 1)
    return bucket, prefix


def sanitize_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", model_id)


def gcs_prefix_has_blobs(client: storage.Client, gcs_prefix: str) -> bool:
    bucket_name, prefix = parse_gcs_uri(gcs_prefix)
    bucket = client.bucket(bucket_name)
    return any(bucket.list_blobs(prefix=prefix, max_results=1))


def download_gcs_prefix(
    client: storage.Client, gcs_prefix: str, dest_dir: Path
) -> None:
    bucket_name, prefix = parse_gcs_uri(gcs_prefix)
    bucket = client.bucket(bucket_name)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for blob in bucket.list_blobs(prefix=prefix):
        rel = blob.name[len(prefix) :].lstrip("/")
        if not rel:
            continue
        out_path = dest_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(out_path)


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


def load_audio(path: Path, target_sr: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(str(path))
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze(0).to(torch.float32)


def segment_audio(
    waveform: torch.Tensor, segment_seconds: int, sample_rate: int
) -> Iterable[Tuple[int, torch.Tensor]]:
    segment_len = segment_seconds * sample_rate
    if waveform.numel() == 0:
        return []
    if waveform.numel() < segment_len:
        padded = torch.zeros(segment_len, dtype=waveform.dtype)
        padded[: waveform.numel()] = waveform
        return [(0, padded)]
    segments = []
    for start in range(0, waveform.numel() - segment_len + 1, segment_len):
        segments.append((start // sample_rate, waveform[start : start + segment_len]))
    return segments


def segment_audio_with_padding(
    waveform: torch.Tensor, segment_seconds: int, sample_rate: int
) -> Iterable[Tuple[int, torch.Tensor]]:
    segment_len = segment_seconds * sample_rate
    if waveform.numel() == 0:
        return []
    max_start = ((waveform.numel() - 1) // segment_len) * segment_len
    segments = []
    for start in range(0, max_start + 1, segment_len):
        end = start + segment_len
        segment = waveform[start:end]
        if segment.numel() < segment_len:
            padded = torch.zeros(segment_len, dtype=waveform.dtype)
            padded[: segment.numel()] = segment
            segment = padded
        segments.append((start // sample_rate, segment))
    return segments


def maybe_empty_cuda_cache(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_conformer_layer(
    conformer: torch.nn.Module,
    hidden_states: torch.Tensor,
    layer_ix: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    num_layers = len(conformer.layers)
    if layer_ix < 0 or layer_ix > num_layers:
        raise ValueError(
            f"layer_ix must be in [0, {num_layers}] for this model, got {layer_ix}"
        )

    captured: dict[str, torch.Tensor] = {}
    handle = None
    try:
        if layer_ix == num_layers:
            outputs = conformer(
                hidden_states,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )
            tensor = (
                outputs["last_hidden_state"]
                if isinstance(outputs, dict)
                else outputs.last_hidden_state
            )
            return tensor.detach()

        if layer_ix == 0:
            def _pre_hook(_module, inputs):
                if "tensor" not in captured:
                    captured["tensor"] = inputs[0].detach()

            handle = conformer.layers[0].register_forward_pre_hook(_pre_hook)
        else:
            target_layer = layer_ix - 1

            def _hook(_module, _inputs, output):
                out = output[0] if isinstance(output, tuple) else output
                captured["tensor"] = out.detach()

            handle = conformer.layers[target_layer].register_forward_hook(_hook)

        _ = conformer(
            hidden_states,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
    finally:
        if handle is not None:
            handle.remove()

    if "tensor" not in captured:
        raise RuntimeError(f"Failed to capture layer {layer_ix} output.")
    return captured["tensor"]


def extract_muq_layer(
    muq: MuQ, wavs: torch.Tensor, layer_ix: int
) -> torch.Tensor:
    features = muq.model.preprocessing(wavs, features=["melspec_2048"])
    features = muq.model.normalize(features)
    conv_out = muq.model.conv(features["melspec_2048"])
    return extract_conformer_layer(muq.model.conformer, conv_out, layer_ix)


def extract_muq_embedding(
    muq: MuQ,
    audio_seg: torch.Tensor,
    device: str,
    layer_ix: int,
) -> np.ndarray | None:
    if audio_seg.numel() < 1025:
        return None
    wavs = audio_seg.unsqueeze(0).to(device)
    with torch.no_grad():
        layer_tensor = extract_muq_layer(muq, wavs, layer_ix)
    embedding = layer_tensor.cpu().float().numpy()
    del wavs, layer_tensor
    maybe_empty_cuda_cache(device)
    return embedding


def upload_npy(
    client: storage.Client, bucket_name: str, blob_name: str, array: np.ndarray
) -> None:
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    buf = io.BytesIO()
    np.save(buf, array)
    buf.seek(0)
    blob.upload_from_file(buf, content_type="application/octet-stream")


def delete_gcs_prefix(bucket: storage.Bucket, prefix: str) -> int:
    prefix = prefix.rstrip("/") + "/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        return 0
    bucket.delete_blobs(blobs)
    return len(blobs)


def gcs_blob_exists(bucket: storage.Bucket, blob_name: str) -> bool:
    return bucket.blob(blob_name).exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MuQ embeddings (30s and 420s)")
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument(
        "--audio-prefix",
        default="audio",
        help="GCS prefix for audio files (default: audio)",
    )
    parser.add_argument(
        "--audio-extension",
        default="mp3",
        help="Audio file extension in GCS (default: mp3)",
    )
    parser.add_argument(
        "--output-root",
        default="embeddings",
        help="GCS prefix for embeddings (e.g., embeddings)",
    )
    parser.add_argument("--split-ids", help="Path to split_ids.txt")
    parser.add_argument("--labels-jsonl", help="Path to labels.jsonl")
    parser.add_argument(
        "--muq-model",
        default=os.environ.get("MUQ_MODEL_NAME", "OpenMuQ/MuQ-large-msd-iter"),
        help="HuggingFace model id or local path for MuQ",
    )
    parser.add_argument(
        "--muq-gcs-prefix",
        default=os.environ.get("MUQ_GCS_PREFIX"),
        help="GCS prefix to cache MuQ weights (default: gs://edmformer-data/metadata/muq/<model-id>)",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token (if required)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("MUQ_DEVICE"),
        help="Device override (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--wipe-output",
        action="store_true",
        help="Delete existing MuQ embeddings under the output root before writing.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip writing embeddings that already exist in GCS.",
    )
    args = parser.parse_args()

    ids = read_ids(args.split_ids, args.labels_jsonl)
    client = storage.Client()
    bucket = client.bucket(args.bucket)

    if args.wipe_output:
        removed_30s = delete_gcs_prefix(
            bucket, f"{args.output_root.rstrip('/')}/muq_30s"
        )
        removed_420s = delete_gcs_prefix(
            bucket, f"{args.output_root.rstrip('/')}/muq_420s"
        )
        print(
            f"Deleted {removed_30s} blobs under muq_30s and {removed_420s} under muq_420s."
        )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.muq_model
    local_dir = None
    if model_path.startswith("gs://"):
        local_dir = Path("/tmp/muq")
        download_gcs_prefix(client, model_path, local_dir)
        model_path = str(local_dir)
    elif Path(model_path).exists():
        model_path = str(Path(model_path).resolve())
    else:
        gcs_prefix = args.muq_gcs_prefix
        if not gcs_prefix:
            safe_name = sanitize_model_id(model_path)
            gcs_prefix = f"gs://edmformer-data/metadata/muq/{safe_name}"

        local_dir = Path("/tmp/muq")
        if gcs_prefix_has_blobs(client, gcs_prefix):
            download_gcs_prefix(client, gcs_prefix, local_dir)
        else:
            snapshot_download(
                repo_id=model_path,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                token=args.hf_token,
            )
            upload_dir_to_gcs(client, local_dir, gcs_prefix)
        model_path = str(local_dir)

    muq = MuQ.from_pretrained(model_path).to(device).eval()

    for audio_id in ids:
        blob_name = f"{args.audio_prefix.rstrip('/')}/{audio_id}.{args.audio_extension}"
        local_path = download_gcs_blob(bucket, blob_name)
        try:
            waveform = load_audio(local_path, TARGET_SAMPLE_RATE)
        finally:
            local_path.unlink(missing_ok=True)
        if waveform.numel() < WRAP_SIZE * TARGET_SAMPLE_RATE:
            padded = torch.zeros(
                WRAP_SIZE * TARGET_SAMPLE_RATE, dtype=waveform.dtype
            )
            padded[: waveform.numel()] = waveform
            waveform = padded

        # 30s embeddings wrapped into 420s windows
        wrap_subdir = "muq_30s"
        for wrap_start in range(0, 10000, WRAP_SIZE):
            if wrap_start * TARGET_SAMPLE_RATE >= waveform.numel():
                break
            out_blob = (
                f"{args.output_root.rstrip('/')}/{wrap_subdir}/{audio_id}_{wrap_start}.npy"
            )
            if args.skip_existing and gcs_blob_exists(bucket, out_blob):
                print(f"Skip existing gs://{args.bucket}/{out_blob}")
                continue
            layer_embeds: list[np.ndarray] = []
            for j in range(0, WRAP_SIZE, HOP_SIZE):
                start_idx = (wrap_start + j) * TARGET_SAMPLE_RATE
                end_idx = min(
                    (wrap_start + j + WIN_SIZE) * TARGET_SAMPLE_RATE, waveform.numel()
                )
                if start_idx >= waveform.numel():
                    break
                audio_seg = waveform[start_idx:end_idx]
                if audio_seg.numel() < 1025:
                    break
                embedding = extract_muq_embedding(muq, audio_seg, device, layer_ix=10)
                if embedding is None:
                    break
                layer_embeds.append(embedding)

            if not layer_embeds:
                continue

            wrap_embedding = np.concatenate(layer_embeds, axis=1)
            upload_npy(client, args.bucket, out_blob, wrap_embedding)
            print(f"Wrote gs://{args.bucket}/{out_blob}")

        # 420s embeddings (single window)
        full_subdir = "muq_420s"
        for start_sec, segment in segment_audio_with_padding(
            waveform, WRAP_SIZE, TARGET_SAMPLE_RATE
        ):
            out_blob = (
                f"{args.output_root.rstrip('/')}/{full_subdir}/{audio_id}_{start_sec}.npy"
            )
            if args.skip_existing and gcs_blob_exists(bucket, out_blob):
                print(f"Skip existing gs://{args.bucket}/{out_blob}")
                continue
            if segment.numel() < 1025:
                break
            layer_embeds: list[np.ndarray] = []
            for j in range(0, WRAP_SIZE, HOP_SIZE):
                start_idx = j * TARGET_SAMPLE_RATE
                end_idx = min((j + WIN_SIZE) * TARGET_SAMPLE_RATE, segment.numel())
                if start_idx >= segment.numel():
                    break
                audio_seg = segment[start_idx:end_idx]
                embedding = extract_muq_embedding(muq, audio_seg, device, layer_ix=10)
                if embedding is None:
                    break
                layer_embeds.append(embedding)

            if not layer_embeds:
                continue

            embedding = np.concatenate(layer_embeds, axis=1)
            upload_npy(client, args.bucket, out_blob, embedding)
            print(f"Wrote gs://{args.bucket}/{out_blob}")


if __name__ == "__main__":
    main()
