#!/usr/bin/env python3
"""
GCP Service: Cloud Storage
IAM Roles: roles/storage.objectAdmin (write embeddings), roles/storage.objectViewer (read audio)
"""

import argparse
import io
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from google.cloud import storage
import torch
import torchaudio

SEGMENT_SECONDS = (30, 420)
TARGET_SAMPLE_RATE = 24000
MUSICFM_HF_BASE = "https://huggingface.co/minzwon/MusicFM/resolve/main"


def read_ids(split_ids_path: str | None, labels_jsonl_path: str | None) -> list[str]:
    if split_ids_path:
        return [
            line.strip()
            for line in Path(split_ids_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if labels_jsonl_path:
        ids: list[str] = []
        with Path(labels_jsonl_path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                ids.append(str(record["id"]))
        return ids
    raise ValueError("Provide --split-ids or --labels-jsonl")


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    bucket, blob = uri[5:].split("/", 1)
    return bucket, blob


def download_to_cache(uri: str, cache_dir: Path, client: storage.Client) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(uri).name
    dest = cache_dir / filename
    if dest.exists():
        return dest

    if uri.startswith("gs://"):
        bucket_name, blob_name = parse_gcs_uri(uri)
        blob = client.bucket(bucket_name).blob(blob_name)
        blob.download_to_filename(dest)
    elif uri.startswith("http://") or uri.startswith("https://"):
        urllib.request.urlretrieve(uri, dest)  # nosec - controlled URL
    else:
        raise ValueError(f"Unsupported URI: {uri}")
    return dest


def ensure_gcs_blob(
    uri: str,
    fallback_url: str,
    cache_dir: Path,
    client: storage.Client,
) -> Path:
    bucket_name, blob_name = parse_gcs_uri(uri)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / Path(blob_name).name

    if blob.exists():
        blob.download_to_filename(dest)
        return dest

    if not fallback_url:
        raise FileNotFoundError(
            f"Missing gs://{bucket_name}/{blob_name} and no fallback URL provided"
        )

    print(f"Downloading {fallback_url} and uploading to gs://{bucket_name}/{blob_name}")
    urllib.request.urlretrieve(fallback_url, dest)  # nosec - controlled URL
    blob.upload_from_filename(dest)
    return dest


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


def upload_npy(
    client: storage.Client, bucket_name: str, blob_name: str, array: np.ndarray
) -> None:
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    buf = io.BytesIO()
    np.save(buf, array)
    buf.seek(0)
    blob.upload_from_file(buf, content_type="application/octet-stream")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract MusicFM embeddings (30s and 420s)"
    )
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
        "--musicfm-home",
        default=os.environ.get("MUSICFM_HOME", "/app"),
        help="Path that contains the musicfm repo (default: /app)",
    )
    parser.add_argument(
        "--musicfm-stat-path",
        default=os.environ.get("MUSICFM_STAT_PATH"),
        help="Path or URI to MusicFM stat json (msd_stats.json or fma_stats.json)",
    )
    parser.add_argument(
        "--musicfm-model-path",
        default=os.environ.get("MUSICFM_MODEL_PATH"),
        help="Path or URI to MusicFM checkpoint (pretrained_msd.pt or pretrained_fma.pt)",
    )
    parser.add_argument(
        "--variant",
        choices=["msd", "fma"],
        default=os.environ.get("MUSICFM_VARIANT", "msd"),
        help="Default MusicFM variant if paths are not provided",
    )
    parser.add_argument(
        "--layer-ix",
        type=int,
        default=7,
        help="Layer index for get_latent (default: 7)",
    )
    parser.add_argument(
        "--use-flash",
        action="store_true",
        help="Enable flash attention (if supported)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use half precision for model and audio",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("MUSICFM_DEVICE"),
        help="Device override (default: cuda if available else cpu)",
    )
    args = parser.parse_args()

    ids = read_ids(args.split_ids, args.labels_jsonl)
    client = storage.Client()
    bucket = client.bucket(args.bucket)

    sys.path.append(args.musicfm_home)
    from musicfm.model.musicfm_25hz import MusicFM25Hz

    stat_path = args.musicfm_stat_path
    model_path = args.musicfm_model_path
    if not stat_path or not model_path:
        stat_file = "msd_stats.json" if args.variant == "msd" else "fma_stats.json"
        model_file = (
            "pretrained_msd.pt" if args.variant == "msd" else "pretrained_fma.pt"
        )
        default_stat = f"gs://edmformer-data/metadata/{stat_file}"
        default_model = f"gs://edmformer-data/metadata/{model_file}"
        stat_path = stat_path or default_stat
        model_path = model_path or default_model

    hf_stat_url = f"{MUSICFM_HF_BASE}/{Path(stat_path).name}"
    hf_model_url = f"{MUSICFM_HF_BASE}/{Path(model_path).name}"

    if stat_path.startswith("gs://") or stat_path.startswith("http"):
        if stat_path.startswith("gs://"):
            stat_path = str(
                ensure_gcs_blob(
                    stat_path, hf_stat_url, Path("/tmp/musicfm"), client
                )
            )
        else:
            stat_path = str(download_to_cache(stat_path, Path("/tmp/musicfm"), client))
    if model_path.startswith("gs://") or model_path.startswith("http"):
        if model_path.startswith("gs://"):
            model_path = str(
                ensure_gcs_blob(
                    model_path, hf_model_url, Path("/tmp/musicfm"), client
                )
            )
        else:
            model_path = str(download_to_cache(model_path, Path("/tmp/musicfm"), client))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    musicfm = MusicFM25Hz(
        is_flash=args.use_flash,
        stat_path=stat_path,
        model_path=model_path,
    )
    musicfm = musicfm.to(device).eval()
    if args.fp16:
        musicfm = musicfm.half()

    for audio_id in ids:
        blob_name = f"{args.audio_prefix.rstrip('/')}/{audio_id}.{args.audio_extension}"
        local_path = download_gcs_blob(bucket, blob_name)
        try:
            waveform = load_audio(local_path, TARGET_SAMPLE_RATE)
        finally:
            local_path.unlink(missing_ok=True)

        for segment_seconds in SEGMENT_SECONDS:
            for start_sec, segment in segment_audio(
                waveform, segment_seconds, TARGET_SAMPLE_RATE
            ):
                wavs = segment.unsqueeze(0).to(device)
                if args.fp16:
                    wavs = wavs.half()
                with torch.no_grad():
                    emb = musicfm.get_latent(wavs, layer_ix=args.layer_ix)
                embedding = emb.squeeze(0).detach().cpu().numpy().astype(np.float32)
                subdir = f"musicfm_{segment_seconds}s"
                out_blob = (
                    f"{args.output_root.rstrip('/')}/{subdir}/{audio_id}_{start_sec}.npy"
                )
                upload_npy(client, args.bucket, out_blob, embedding)
                print(f"Wrote gs://{args.bucket}/{out_blob}")


if __name__ == "__main__":
    main()
