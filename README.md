# EDMFormer Project (SongFormer Retraining on Vertex AI)

This repository provisions ingestion, preprocessing, and training orchestration for retraining EDMFormer (SongFormer variant) on Google Cloud Vertex AI. It assumes audio lives in Google Drive and is migrated to Cloud Storage before SSL embedding extraction (MuQ/MusicFM) and TPU training.

## Quick Start

1. Initialize the EDMFormer submodule
   - `git submodule add <EDMFORMER_REPO_URL> third_party/EDMFormer`
   - `git submodule update --init --recursive`

2. Fill in environment placeholders
   - Edit `config/gcp_env.yaml` and replace all `<PLACEHOLDER_REQUIRED_FIELD_MANUAL_FILL>` values.

3. Enable required APIs
   - `gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com storage.googleapis.com iam.googleapis.com` 

4. Create a Cloud Storage bucket
   - `gsutil mb -l ${REGION} gs://${BUCKET_NAME}`

5. Create an Artifact Registry repository
   - `gcloud artifacts repositories create ${ARTIFACT_REPO} --repository-format=docker --location=${REGION}`

6. Build and push images
   - Preprocessing (GPU):
   - `docker build -f docker/preprocessing.Dockerfile -t ${REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACT_REPO}/edmformer-preprocess:latest .`
   - `docker push ${REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACT_REPO}/edmformer-preprocess:latest`
   - Training (GPU):
   - `docker build -f docker/training.Dockerfile -t ${REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACT_REPO}/edmformer-train:latest .`
   - `docker push ${REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACT_REPO}/edmformer-train:latest`
   - Training (TPU):
   - `docker build -f docker/training_tpu.Dockerfile -t ${REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACT_REPO}/edmformer-train-tpu:latest .`
   - `docker push ${REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACT_REPO}/edmformer-train-tpu:latest`

7. Compile the Vertex AI pipeline
   - `python pipelines/compile_pipeline.py --output pipeline.json`

8. Submit the pipeline in Vertex AI
   - Upload `pipeline.json` in the Vertex AI Pipelines UI or via `gcloud ai pipelines run`.
   - For TPU jobs, use the TPU image and set `TRAIN_BACKEND=TPU` in the environment.

## Service Accounts and IAM

Recommended minimum roles for the pipeline service account:
- `roles/storage.objectAdmin` (Cloud Storage read/write)
- `roles/aiplatform.user` (Vertex AI Custom Jobs and Pipelines)
- `roles/artifactregistry.reader` (pull images)
- `roles/iam.serviceAccountUser` (impersonation)

## Inputs

- `labels.jsonl` with fields: `id`, `file_path` (Google Drive path)
- Drive access for ingestion (OAuth or service account with shared Drive access)

## Outputs

- Cloud Storage audio: `gs://<BUCKET>/audio/<id>.mp3`
- SSL embeddings: `gs://<BUCKET>/embeddings/{muq_30s,muq_420s,musicfm_30s,musicfm_420s}/<id>_<start_sec>.npy`

## Notes

- `src/config_generator.py` rewrites `third_party/EDMFormer/configs/SongFormer.yaml` at job start based on GCS paths.
- `src/trainer_adapter.py` is not used by the current pipeline. TPU training uses `src/tpu_train.py` with `torch_xla`.
- `src/edmformer_gcs_dataset.py` streams labels, splits, and embeddings directly from GCS (`gs://`).

## GPU / TPU Toggle

Use `TRAIN_BACKEND` to control the training backend.

- GPU (default):
  - Image: `docker/training.Dockerfile`
  - `TRAIN_BACKEND=GPU`
  - Vertex AI machine type: `g2-standard-8` + `NVIDIA_L4`
- TPU:
  - Image: `docker/training_tpu.Dockerfile`
  - `TRAIN_BACKEND=TPU`
  - Vertex AI machine type: `cloud-tpu` + `TPU_V3`

## Example: GPU Custom Job

```bash
gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name=<JOB_NAME> \
  --worker-pool-spec=machine-type=g2-standard-8,accelerator-type=NVIDIA_L4,accelerator-count=1,replica-count=1,container-image-uri=${REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACT_REPO}/edmformer-train:latest,env=TRAIN_BACKEND=GPU,env=LABEL_PATH_GCS=gs://<BUCKET>/metadata/dataset.jsonl,env=SPLIT_IDS_PATH_GCS=gs://<BUCKET>/metadata/train.txt,env=EVAL_SPLIT_IDS_PATH_GCS=gs://<BUCKET>/metadata/val.txt,env=INPUT_EMBEDDING_DIR_GCS=gs://<BUCKET>/embeddings,env=EMBEDDING_SUBDIRS=musicfm_30s,muq_30s,musicfm_420s,muq_420s,env=CHECKPOINT_DIR_GCS=gs://<BUCKET>/checkpoints/<RUN_ID>,env=INIT_SEED=42
```

## Example: TPU Custom Job

```bash
gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name=<JOB_NAME> \
  --worker-pool-spec=machine-type=cloud-tpu,accelerator-type=TPU_V3,accelerator-count=8,replica-count=1,container-image-uri=${REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACT_REPO}/edmformer-train-tpu:latest,env=TRAIN_BACKEND=TPU,env=LABEL_PATH_GCS=gs://<BUCKET>/metadata/dataset.jsonl,env=SPLIT_IDS_PATH_GCS=gs://<BUCKET>/metadata/train.txt,env=EVAL_SPLIT_IDS_PATH_GCS=gs://<BUCKET>/metadata/val.txt,env=INPUT_EMBEDDING_DIR_GCS=gs://<BUCKET>/embeddings,env=EMBEDDING_SUBDIRS=musicfm_30s,muq_30s,musicfm_420s,muq_420s,env=CHECKPOINT_DIR_GCS=gs://<BUCKET>/checkpoints/<RUN_ID>,env=INIT_SEED=42
```
