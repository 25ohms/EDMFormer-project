# Preprocessing: MuQ and MusicFM Embeddings

This folder contains GPU-focused extraction scripts for MuQ and MusicFM embeddings. The scripts generate four directories of `.npy` files in GCS:

- `muq_30s`
- `muq_420s`
- `musicfm_30s`
- `musicfm_420s`

Each file is named `<id>_<start_sec>.npy`.

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
