# Preprocessing: MuQ and MusicFM Embeddings

This folder contains GPU-focused extraction scripts for MuQ and MusicFM embeddings. The scripts generate four directories of `.npy` files in GCS:

- `muq_30s`
- `muq_420s`
- `musicfm_30s`
- `musicfm_420s`

Each file is named `<id>_<start_sec>.npy`.

Note: the `*_30s` directories store embeddings for 420s windows that are built by concatenating 30s sub-windows (wrap-420 format), matching EDMFormer’s expected input.

## Loading Model Weights in the Container

There are two common options for making MuQ and MusicFM weights available inside the preprocessing container:

1. Bake weights into the image
   - Copy weights into the image during `docker build` and set env vars like `MUQ_WEIGHTS_PATH` and `MUSICFM_WEIGHTS_PATH`.

2. Pull weights at runtime
   - Store weights in a private GCS bucket and download them at container start.
   - Example pattern in your entrypoint or script:
     - `gsutil cp gs://<BUCKET>/weights/muq.pt /opt/weights/muq.pt`
     - `gsutil cp gs://<BUCKET>/weights/musicfm.pt /opt/weights/musicfm.pt`

Update `extract_muq.py` and `extract_musicfm.py` to load the model checkpoints from those paths before running extraction.

## Source Code Installation

- MuQ is installed via pip from the GitHub repo in `requirements_pre.txt`.
- MusicFM is cloned into the image in `docker/preprocessing.Dockerfile` and exposed via `PYTHONPATH=/app`.

To pin MusicFM to a commit or tag, build with:
- `--build-arg MUSICFM_REF=<commit-or-tag>`

## Runtime Configuration

MuQ (Hugging Face model id):
- `MUQ_MODEL_NAME` (default: `OpenMuQ/MuQ-large-msd-iter`)
- If the model is not available in `MUQ_GCS_PREFIX`, the script downloads from Hugging Face and uploads to `gs://edmformer-data/metadata/muq/<model-id>`.

MusicFM (checkpoint paths):
- `MUSICFM_STAT_PATH` and `MUSICFM_MODEL_PATH` can be local paths or `gs://` URIs.
- If not provided, defaults to `gs://edmformer-data/metadata/msd_stats.json` and `gs://edmformer-data/metadata/pretrained_msd.pt`.
- If the default GCS files are missing, the script downloads them from Hugging Face and uploads them to GCS automatically.

Both models expect 24kHz audio input.

Embedding layer defaults:
- MuQ: layer 10 (hard-coded to match EDMFormer scripts).
- MusicFM: layer 10 (`--layer-ix`).
