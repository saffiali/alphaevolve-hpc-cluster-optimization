#!/usr/bin/env bash
# Deploy the AlphaEvolve HPC Optimization Workbench & Evaluator image to Google Cloud.
#
# Required environment variable:
#   export GCP_PROJECT_ID="<YOUR_GCP_PROJECT_ID>"
# Optional overrides:
#   export GCP_REGION="us-central1"
#   export AE_AR_REPO="alphaevolve-hpc-poc"
#   export AE_SERVICE_NAME="alphaevolve-hpc-workbench"

set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID to your target Google Cloud project ID}"
REGION="${GCP_REGION:-us-central1}"
AR_REPO="${AE_AR_REPO:-alphaevolve-hpc-poc}"
SERVICE_NAME="${AE_SERVICE_NAME:-alphaevolve-hpc-workbench}"
BQ_DATASET="${AE_BQ_DATASET:-alphaevolve_hpc_poc}"
ARCHIVE_BUCKET="${AE_ARCHIVE_BUCKET:-${PROJECT_ID}-alphaevolve-hpc-poc}"
SERVICE_ACCOUNT="${AE_SERVICE_ACCOUNT:-ae-hpc-poc-sa@${PROJECT_ID}.iam.gserviceaccount.com}"
BATCH_JOB_NAME="${AE_BATCH_JOB_NAME:-}"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${SERVICE_NAME}:latest"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "==> Building container image via Cloud Build: ${IMAGE_URI}"
gcloud builds submit "${REPO_ROOT}" \
  --project="${PROJECT_ID}" \
  --tag="${IMAGE_URI}" \
  --quiet

echo "==> Deploying Cloud Run service: ${SERVICE_NAME} (${REGION})"
gcloud run deploy "${SERVICE_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${IMAGE_URI}" \
  --service-account="${SERVICE_ACCOUNT}" \
  --no-invoker-iam-check \
  --cpu=2 \
  --memory=2Gi \
  --timeout=120 \
  --max-instances=10 \
  --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCP_REGION=${REGION},AE_BQ_DATASET=${BQ_DATASET},AE_ARCHIVE_BUCKET=${ARCHIVE_BUCKET},AE_BATCH_JOB_NAME=${BATCH_JOB_NAME}" \
  --quiet

SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" --project="${PROJECT_ID}" --region="${REGION}" --format='value(status.url)')"
echo "==> Deployed successfully: ${SERVICE_URL}"
