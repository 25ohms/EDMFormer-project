# SongFormer Cloud Retrain (EDMFormer on Vertex AI)

This repository provisions ingestion, preprocessing, and training orchestration for retraining SongFormer (EDMFormer variant) on Google Cloud Vertex AI. It assumes audio lives in Google Drive and is migrated to Cloud Storage before SSL embedding extraction (MuQ/MusicFM) and TPU training.

## Quick Start

1. Initialize the SongFormer submodule
   - `git submodule add https://github.com/25ohms/EDMFormer third_party/SongFormer`
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
     - `docker build -f docker/preprocessing.Dockerfile -t ${REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACT_REPO}/songformer-preprocess:latest .`
     - `docker push ${REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACT_REPO}/songformer-preprocess:latest`
   - Training (TPU):
     - `docker build -f docker/training.Dockerfile -t ${REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACT_REPO}/songformer-train:latest .`
     - `docker push ${REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACT_REPO}/songformer-train:latest`

7. Compile the Vertex AI pipeline
   - `python pipelines/compile_pipeline.py --output pipeline.json`

8. Submit the pipeline in Vertex AI
   - Upload `pipeline.json` in the Vertex AI Pipelines UI or via `gcloud ai pipelines run`.

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

- `src/config_generator.py` rewrites `third_party/SongFormer/configs/SongFormer.yaml` at job start based on GCS paths.
- `src/trainer_adapter.py` wraps SongFormer training to use `torch_xla` and the BCE + TV loss.
