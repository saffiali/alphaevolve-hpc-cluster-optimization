output "workbench_url" {
  description = "HTTPS URL of the deployed AlphaEvolve HPC Cloud Run Workbench."
  value       = google_cloud_run_v2_service.workbench.uri
}

output "archive_bucket_url" {
  description = "Cloud Storage URI for AlphaEvolve candidate archives and trace datasets."
  value       = google_storage_bucket.experiment_archive.url
}

output "bigquery_dataset_id" {
  description = "Fully qualified BigQuery telemetry dataset identifier."
  value       = "${var.project_id}.${google_bigquery_dataset.telemetry.dataset_id}"
}

output "artifact_registry_uri" {
  description = "Artifact Registry Docker repository base URI."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.containers.repository_id}"
}

output "evaluator_service_account_email" {
  description = "Sandboxed Cloud Batch evaluator worker service account email."
  value       = google_service_account.evaluator_worker.email
}

output "workbench_service_account_email" {
  description = "Cloud Run Workbench & controller service account email."
  value       = google_service_account.workbench_controller.email
}
