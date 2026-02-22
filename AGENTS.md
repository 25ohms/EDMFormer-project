# Agents.md: Project Generation Instructions

## 1. Objective
Construct a cloud-native MLOps repository for retraining **SongFormer** (EDMFormer variant) on Google Cloud Vertex AI. The architecture must facilitate a seamless transition from Google Drive audio files to precomputed SSL embeddings (MuQ/MusicFM) used for TPU-based training.

---

## 2. Target Directory Structure
Generate the following hierarchy. Use modular scripts for ingestion, preprocessing, and training orchestration.

```text
songformer-cloud-retrain/
├── .gitmodules                 # Points to [https://github.com/25ohms/EDMFormer](https://github.com/25ohms/EDMFormer)
├── README.md                   # Instructions for GCP setup and Artifact Registry push
├── requirements.txt            # Local dev deps: google-cloud-storage, gspread, kfp
│
├── ingestion/
│   ├── migrate_drive_to_gcs.py # Maps Drive audio -> GCS using labels.jsonl IDs
│   └── generate_split_ids.py   # Creates split_ids.txt from the labels.jsonl
│
├── preprocessing/              # GPU-based Feature Extraction (Pre-training)
│   ├── extract_muq.py          # Extracts 30s & 420s MuQ .npy embeddings
│   ├── extract_musicfm.py      # Extracts 30s & 420s MusicFM .npy embeddings
│   └── requirements_pre.txt    # Deps for SSL models (MuQ, MusicFM, torchaudio)
│
├── src/                        # Training Orchestration
│   ├── task.py                 # Main entrypoint for Vertex AI CustomJob
│   ├── config_generator.py     # Dynamically overwrites SongFormer.yaml with GCS paths
│   └── trainer_adapter.py      # PyTorch/XLA wrapper for TPU support
│
├── pipelines/                  # Vertex AI (Kubeflow) Pipeline Definitions
│   ├── compile_pipeline.py     # Compiles pipeline to JSON for Vertex AI
│   └── components/             # Reusable Pipeline Components
│       ├── ingest_op.py
│       ├── preprocess_op.py
│       └── train_op.py
│
├── docker/                     # Environment Definitions
│   ├── preprocessing.Dockerfile # GPU Image for SSL extraction
│   └── training.Dockerfile      # TPU Image (PyTorch-XLA base)
│
└── config/
    └── gcp_env.yaml            # Environment variables (GCP_PROJECT, BUCKET_NAME)

```

---

## 3. Detailed Logic & Functional Requirements

### A. Data Ingestion (`ingestion/`)

* **Logic**: `migrate_drive_to_gcs.py` must read the provided `labels.jsonl`.
* **Action**:
1. Resolve the source file in Google Drive via the `file_path` attribute.
2. Upload to `gs://<PLACEHOLDER_BUCKET_NAME>/audio/<id>.mp3`.
3. Generate a `split_ids.txt` (newline-separated IDs) to meet submodule requirements.



### B. Preprocessing Step (`preprocessing/`)

* **Hardware Requirement**: Must be optimized for **NVIDIA L4/T4 GPUs**.
* **Task**: Generate four directories of `.npy` files: `muq_30s`, `muq_420s`, `musicfm_30s`, `musicfm_420s`.
* **Naming Convention**: Files must be named `<id>_<start_sec>.npy`.
* **Documentation**: Include a README in this folder explaining how to load the MuQ/MusicFM weights into the container.

### C. Config Adaptation (`src/config_generator.py`)

* **Problem**: `third_party/SongFormer/configs/SongFormer.yaml` requires absolute local paths.
* **Agent Task**: Create a utility that runs at the start of the Training Job. It must:
1. Read GCS paths from environment variables (passed by Vertex AI).
2. Dynamically update the `SongFormer.yaml` fields: `label_path`, `split_ids_path`, and `input_embedding_dir`.
3. Set `dataset_type: "EDMFormer"`.



### D. Training Adapter (`src/trainer_adapter.py`)

* **Requirement**: Wrap the SongFormer training loop to ensure it utilizes `torch_xla`.
* **Functionality**: Use `xm.optimizer_step(optimizer)` and implement the **BCE + Total Variation (TV) loss** mentioned in the SongFormer paper.

---

## 4. Hardware & Environment Specs

| Component | Device | Base Image |
| --- | --- | --- |
| **Preprocessing** | GPU (L4) | `nvidia/cuda:12.1-base` |
| **Training** | TPU (v3-8) | `us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-xla.2-1.py310` |

---

## 5. Placeholders & Variables

All sensitive or environment-specific values should use the following placeholder:
`"<PLACEHOLDER_REQUIRED_FIELD_MANUAL_FILL>"`

---

**Documentation Note**: For every generated script, the AI Agent must include a header comment describing the specific **GCP Service** (e.g., Vertex Custom Job, Cloud Storage) it interacts with and any required **IAM Roles** (e.g., Storage Object Admin).
