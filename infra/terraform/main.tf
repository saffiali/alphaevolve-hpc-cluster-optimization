locals {
  required_apis = toset([
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "batch.googleapis.com",
    "bigquery.googleapis.com",
    "cloudbuild.googleapis.com",
    "compute.googleapis.com",
    "logging.googleapis.com",
    "pubsub.googleapis.com",
    "run.googleapis.com",
    "storage.googleapis.com",
  ])

  archive_bucket_name = "${var.project_id}-${var.archive_bucket_suffix}"

  common_labels = {
    managed_by = "terraform"
    platform   = "alphaevolve-hpc-optimization"
  }
}

resource "google_project_service" "enabled_apis" {
  for_each           = local.required_apis
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# ------------------------------------------------------------------------------
# Cloud Storage Experiment Archive & Read-Only Dataset Bucket
# ------------------------------------------------------------------------------
resource "google_storage_bucket" "experiment_archive" {
  name                        = local.archive_bucket_name
  project                     = var.project_id
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false
  labels                      = local.common_labels

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      num_newer_versions = 5
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.enabled_apis]
}

# ------------------------------------------------------------------------------
# Artifact Registry Docker Repository (Evaluator & Workbench Images)
# ------------------------------------------------------------------------------
resource "google_artifact_registry_repository" "containers" {
  project       = var.project_id
  location      = var.region
  repository_id = var.artifact_repo_id
  description   = "AlphaEvolve HPC evaluator and interactive workbench container images"
  format        = "DOCKER"
  labels        = local.common_labels

  depends_on = [google_project_service.enabled_apis]
}

# ------------------------------------------------------------------------------
# BigQuery Telemetry Dataset & Analytical Tables
# ------------------------------------------------------------------------------
resource "google_bigquery_dataset" "telemetry" {
  dataset_id  = var.bq_dataset_id
  project     = var.project_id
  location    = var.bq_location
  description = "AlphaEvolve 3-Pillar HPC Optimization evolutionary runs and benchmark telemetry"
  labels      = local.common_labels

  depends_on = [google_project_service.enabled_apis]
}

resource "google_bigquery_table" "evolution_runs" {
  dataset_id          = google_bigquery_dataset.telemetry.dataset_id
  table_id            = "evolution_runs"
  project             = var.project_id
  deletion_protection = false
  labels              = local.common_labels

  schema = jsonencode([
    { name = "run_id", type = "STRING", mode = "REQUIRED" },
    { name = "pillar", type = "STRING", mode = "REQUIRED" },
    { name = "pillar_title", type = "STRING", mode = "NULLABLE" },
    { name = "model_name", type = "STRING", mode = "NULLABLE" },
    { name = "generations", type = "INTEGER", mode = "NULLABLE" },
    { name = "baseline_score", type = "FLOAT", mode = "REQUIRED" },
    { name = "evolved_score", type = "FLOAT", mode = "REQUIRED" },
    { name = "improvement_pct", type = "FLOAT", mode = "REQUIRED" },
    { name = "primary_kpi_name", type = "STRING", mode = "NULLABLE" },
    { name = "baseline_kpi_value", type = "FLOAT", mode = "NULLABLE" },
    { name = "evolved_kpi_value", type = "FLOAT", mode = "NULLABLE" },
    { name = "holdout_verified", type = "BOOLEAN", mode = "REQUIRED" },
    { name = "batch_job_name", type = "STRING", mode = "NULLABLE" },
    { name = "created_at", type = "TIMESTAMP", mode = "REQUIRED" }
  ])
}

resource "google_bigquery_table" "candidate_evaluations" {
  dataset_id          = google_bigquery_dataset.telemetry.dataset_id
  table_id            = "candidate_evaluations"
  project             = var.project_id
  deletion_protection = false
  labels              = local.common_labels

  schema = jsonencode([
    { name = "run_id", type = "STRING", mode = "REQUIRED" },
    { name = "pillar", type = "STRING", mode = "REQUIRED" },
    { name = "generation", type = "INTEGER", mode = "REQUIRED" },
    { name = "candidate_id", type = "STRING", mode = "REQUIRED" },
    { name = "stage1_ast_valid", type = "BOOLEAN", mode = "REQUIRED" },
    { name = "stage2_hard_gate_passed", type = "BOOLEAN", mode = "REQUIRED" },
    { name = "overall_score", type = "FLOAT", mode = "REQUIRED" },
    { name = "best_so_far", type = "FLOAT", mode = "REQUIRED" },
    { name = "eval_latency_ms", type = "FLOAT", mode = "NULLABLE" },
    { name = "mutation_summary", type = "STRING", mode = "NULLABLE" },
    { name = "evaluated_at", type = "TIMESTAMP", mode = "REQUIRED" }
  ])
}

resource "google_bigquery_table" "cluster_replay_metrics" {
  dataset_id          = google_bigquery_dataset.telemetry.dataset_id
  table_id            = "cluster_replay_metrics"
  project             = var.project_id
  deletion_protection = false
  labels              = local.common_labels

  schema = jsonencode([
    { name = "scenario_id", type = "STRING", mode = "REQUIRED" },
    { name = "pillar", type = "STRING", mode = "REQUIRED" },
    { name = "policy_name", type = "STRING", mode = "REQUIRED" },
    { name = "split", type = "STRING", mode = "REQUIRED" },
    { name = "metric_name", type = "STRING", mode = "REQUIRED" },
    { name = "metric_value", type = "FLOAT", mode = "REQUIRED" },
    { name = "unit", type = "STRING", mode = "NULLABLE" },
    { name = "recorded_at", type = "TIMESTAMP", mode = "REQUIRED" }
  ])
}

# ------------------------------------------------------------------------------
# Least-Privilege IAM Service Accounts (Secure Split Architecture)
# ------------------------------------------------------------------------------
resource "google_service_account" "evaluator_worker" {
  account_id   = "ae-hpc-eval-worker-sa"
  display_name = "AlphaEvolve Sandboxed Cloud Batch Evaluator SA"
  description  = "Isolated identity for Cloud Batch candidate evaluators (no LLM or controller access)."
  project      = var.project_id
}

resource "google_service_account" "workbench_controller" {
  account_id   = "ae-hpc-workbench-sa"
  display_name = "AlphaEvolve HPC Workbench & Controller SA"
  description  = "Service identity for Cloud Run Workbench querying BigQuery and invoking Vertex AI."
  project      = var.project_id
}

resource "google_project_iam_member" "evaluator_roles" {
  for_each = toset([
    "roles/batch.agentReporter",
    "roles/artifactregistry.reader",
    "roles/logging.logWriter",
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.evaluator_worker.email}"
}

resource "google_storage_bucket_iam_member" "evaluator_bucket_access" {
  bucket = google_storage_bucket.experiment_archive.name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.evaluator_worker.email}"
}

resource "google_project_iam_member" "workbench_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/batch.jobsViewer",
    "roles/bigquery.dataViewer",
    "roles/bigquery.jobUser",
    "roles/logging.logWriter",
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.workbench_controller.email}"
}

resource "google_storage_bucket_iam_member" "workbench_bucket_reader" {
  bucket = google_storage_bucket.experiment_archive.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.workbench_controller.email}"
}

# ------------------------------------------------------------------------------
# Cloud Run v2 Interactive Workbench Service
# ------------------------------------------------------------------------------
resource "google_cloud_run_v2_service" "workbench" {
  name                 = var.workbench_service_name
  location             = var.region
  project              = var.project_id
  ingress              = "INGRESS_TRAFFIC_ALL"
  invoker_iam_disabled = var.invoker_iam_disabled
  labels               = local.common_labels

  template {
    service_account = google_service_account.workbench_controller.email
    timeout         = "120s"

    scaling {
      min_instance_count = 0
      max_instance_count = 10
    }

    containers {
      image = var.workbench_image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }

      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }

      env {
        name  = "GCP_REGION"
        value = var.region
      }

      env {
        name  = "AE_BQ_DATASET"
        value = google_bigquery_dataset.telemetry.dataset_id
      }

      env {
        name  = "AE_ARCHIVE_BUCKET"
        value = google_storage_bucket.experiment_archive.name
      }
    }
  }

  depends_on = [
    google_project_service.enabled_apis,
    google_bigquery_dataset.telemetry,
  ]
}
