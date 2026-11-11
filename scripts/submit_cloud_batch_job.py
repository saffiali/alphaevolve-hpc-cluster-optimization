#!/usr/bin/env python3
"""Submit and monitor a containerized Google Cloud Batch 3-Pillar Evaluator Job.

Creates an isolated Cloud Batch job (`ae-hpc-poc-eval-<ts>`) with:
  - `noExternalIpAddress: true` (private VPC + Cloud NAT compliance)
  - Shielded VM enforcement (`enableSecureBoot`, `enableVtpm`, `enableIntegrityMonitoring`)
  - Containerized execution of `scripts/cloud_batch_eval_runner.py`
  - Verified output uploaded to `gs://<AE_ARCHIVE_BUCKET>/batch-validations/`
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import tempfile
import time
from typing import Any

from google.cloud import storage

from pillars.common.vertex_client import resolve_gcp_project_id


def _run_gcloud(args: list[str]) -> str:
    env = dict(os.environ)
    env.pop("CLOUDSDK_AUTH_ACCESS_TOKEN_FILE", None)
    proc = subprocess.run(
        ["gcloud", *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gcloud {' '.join(args[:4])} failed: {proc.stderr.strip()}")
    return proc.stdout


def submit_and_wait(
    *,
    project_id: str,
    region: str,
    image_uri: str,
    bucket_name: str,
    service_account: str,
    network: str,
    subnetwork: str,
    machine_type: str = "e2-standard-4",
    boot_disk_image: str = "projects/debian-cloud/global/images/family/debian-12",
    timeout_sec: int = 600,
) -> dict[str, Any]:
    job_id = f"ae-hpc-poc-eval-{int(time.time())}"
    batch_config = {
        "taskGroups": [
            {
                "taskSpec": {
                    "runnables": [
                        {
                            "container": {
                                "imageUri": image_uri,
                                "commands": ["python3", "/app/scripts/cloud_batch_eval_runner.py"],
                            }
                        }
                    ],
                    "computeResource": {"cpuMilli": 4000, "memoryMib": 8192},
                    "maxRunDuration": "600s",
                    "maxRetryCount": 1,
                    "environment": {
                        "variables": {
                            "GCP_PROJECT_ID": project_id,
                            "GCP_REGION": region,
                            "AE_ARCHIVE_BUCKET": bucket_name,
                            "AE_BATCH_JOB_ID": job_id,
                        }
                    },
                },
                "taskCount": 1,
                "parallelism": 1,
            }
        ],
        "allocationPolicy": {
            "instances": [
                {
                    "policy": {
                        "machineType": machine_type,
                        "provisioningModel": "STANDARD",
                        "bootDisk": {
                            "image": boot_disk_image,
                        },
                    }
                }
            ],
            "network": {
                "networkInterfaces": [
                    {
                        "network": f"projects/{project_id}/global/networks/{network}",
                        "subnetwork": f"projects/{project_id}/regions/{region}/subnetworks/{subnetwork}",
                        "noExternalIpAddress": True,
                    }
                ]
            },
            "serviceAccount": {"email": service_account},
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(batch_config, tf, indent=2)
        cfg_path = tf.name

    try:
        submit_out = _run_gcloud(
            [
                "batch",
                "jobs",
                "submit",
                job_id,
                f"--project={project_id}",
                f"--location={region}",
                f"--config={cfg_path}",
                "--format=json",
            ]
        )
        submitted = json.loads(submit_out)
    finally:
        if os.path.exists(cfg_path):
            os.remove(cfg_path)

    job_name = submitted.get("name", f"projects/{project_id}/locations/{region}/jobs/{job_id}")
    job_uid = submitted.get("uid", "")
    print(f"Submitted Cloud Batch job: {job_name} (uid={job_uid})")

    deadline = time.time() + timeout_sec
    last_state = "QUEUED"
    while time.time() < deadline:
        desc_out = _run_gcloud(
            [
                "batch",
                "jobs",
                "describe",
                job_id,
                f"--project={project_id}",
                f"--location={region}",
                "--format=json",
            ]
        )
        desc = json.loads(desc_out)
        state = desc.get("status", {}).get("state", "STATE_UNSPECIFIED")
        if state != last_state:
            print(f"[{dt.datetime.now(dt.timezone.utc).strftime('%H:%M:%S')}] Cloud Batch {job_id} -> {state}")
            last_state = state
        if state == "SUCCEEDED":
            meta = {
                "job_id": job_id,
                "job_name": desc.get("name", job_name),
                "job_uid": desc.get("uid", job_uid),
                "state": state,
                "region": region,
                "machine_type": machine_type,
                "gcs_summary_uri": f"gs://{bucket_name}/batch-validations/{job_id}/eval_summary.json",
                "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            storage_client = storage.Client(project=project_id)
            storage_client.bucket(bucket_name).blob("batch-validations/latest_job_metadata.json").upload_from_string(
                json.dumps(meta, indent=2),
                content_type="application/json",
            )
            return meta
        if state in ("FAILED", "DELETION_IN_PROGRESS"):
            events = desc.get("status", {}).get("statusEvents", [])
            raise RuntimeError(f"Cloud Batch job {job_id} ended in {state}: {json.dumps(events[-3:], indent=2)}")
        time.sleep(10)

    raise TimeoutError(f"Cloud Batch job {job_id} did not finish within {timeout_sec}s (last_state={last_state})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit & verify Google Cloud Batch AlphaEvolve evaluator job.")
    parser.add_argument("--project-id", default="")
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "us-central1"))
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--bucket-name", default="")
    parser.add_argument("--service-account", default="")
    parser.add_argument("--network", default="default")
    parser.add_argument("--subnetwork", default="default")
    args = parser.parse_args()

    proj = resolve_gcp_project_id(args.project_id or None)
    bucket = args.bucket_name or f"{proj}-alphaevolve-hpc-poc"
    sa = args.service_account or f"ae-hpc-poc-sa@{proj}.iam.gserviceaccount.com"

    meta = submit_and_wait(
        project_id=proj,
        region=args.region,
        image_uri=args.image_uri,
        bucket_name=bucket,
        service_account=sa,
        network=args.network,
        subnetwork=args.subnetwork,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
