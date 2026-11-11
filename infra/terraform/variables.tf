variable "project_id" {
  description = "Target Google Cloud Project ID for AlphaEvolve HPC Optimization platform."
  type        = string
}

variable "region" {
  description = "Primary Google Cloud region for Cloud Run, Artifact Registry, and Cloud Batch."
  type        = string
  default     = "us-central1"
}

variable "bq_location" {
  description = "BigQuery multi-region or regional location for telemetry tables."
  type        = string
  default     = "US"
}

variable "bq_dataset_id" {
  description = "BigQuery dataset ID storing AlphaEvolve evolutionary trajectories and trace replays."
  type        = string
  default     = "alphaevolve_hpc_poc"
}

variable "artifact_repo_id" {
  description = "Artifact Registry Docker repository ID for evaluator and workbench images."
  type        = string
  default     = "alphaevolve-hpc-poc"
}

variable "archive_bucket_suffix" {
  description = "Suffix appended to project_id for the Cloud Storage experiment archive bucket."
  type        = string
  default     = "alphaevolve-hpc-poc"
}

variable "workbench_service_name" {
  description = "Cloud Run service name for the Interactive AlphaEvolve HPC Workbench."
  type        = string
  default     = "alphaevolve-hpc-workbench"
}

variable "workbench_image" {
  description = "Container image URI deployed to the Cloud Run Workbench service."
  type        = string
  default     = "us-docker.pkg.dev/cloudrun/container/hello"
}

variable "invoker_iam_disabled" {
  description = "Disable Cloud Run IAM invoker check where organization policies restrict allUsers."
  type        = bool
  default     = true
}
