#!/usr/bin/env python3
"""Containerized 3-Pillar AlphaEvolve Evaluator Harness for Google Cloud Batch.

Executes inside an isolated Google Cloud Batch worker container to verify:
1. Canonical seed program evaluation across all 3 pillars (`cluster_scheduling`,
   `hpc_code_optimization`, `demand_forecasting`).
2. Winning evolved `# EVOLVE-BLOCK` candidate verification on held-out test splits.
3. Anti-reward-hacking hard-gate enforcement (verifying sabotaged candidates receive
   `HARD_GATE_PENALTY` == -1e12).
4. Uploads structured JSON validation proof to `gs://<AE_ARCHIVE_BUCKET>/batch-proofs/`.
"""

from __future__ import annotations

import datetime as dt
import importlib
import json
import os
from pathlib import Path
import socket
import time
from typing import Any

from google.cloud import storage

from pillars.common.models import EVOLVE_END_MARKER, EVOLVE_START_MARKER, HARD_GATE_PENALTY
from pillars.common.sandbox import replace_evolve_block

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "pillars" / "results"

PILLAR_SPECS = [
    ("cluster_scheduling", "pillars.cluster_scheduling"),
    ("hpc_code_optimization", "pillars.hpc_code_optimization"),
    ("demand_forecasting", "pillars.demand_forecasting"),
]


def _invoke_pillar_evaluator(pillar_pkg: str, source_code: str, split: str = "test") -> dict[str, Any]:
    """Dynamically invoke a pillar's `evaluator.py` entrypoint with flexible signature handling."""
    eval_mod = importlib.import_module(f"{pillar_pkg}.evaluator")
    for fn_name in ("evaluate_candidate_source", "evaluate_source", "evaluate_program", "evaluate"):
        fn = getattr(eval_mod, fn_name, None)
        if callable(fn):
            try:
                res = fn(source_code, split=split)
            except TypeError:
                res = fn(source_code)
            if isinstance(res, dict):
                return res
    raise RuntimeError(f"No compatible evaluation entrypoint found in {pillar_pkg}.evaluator")


def _extract_winning_block(result_payload: dict[str, Any]) -> str | None:
    """Extract the evolved code block from a pillar result artifact."""
    for key in (
        "winning_evolved_code_block",
        "evolved_block",
        "winning_evolved_block",
        "best_evolved_block",
        "best_block",
        "winning_code_block",
        "evolved_code_block",
    ):
        val = result_payload.get(key)
        if isinstance(val, str) and val.strip():
            return val
    for full_key in ("winning_full_program", "evolved_source", "winning_source", "best_program_source"):
        val = result_payload.get(full_key)
        if isinstance(val, str) and EVOLVE_START_MARKER in val:
            return val
    return None


def main() -> None:
    t_start = time.perf_counter()
    job_id = os.environ.get("BATCH_JOB_ID") or os.environ.get("AE_BATCH_JOB_ID") or f"local-{int(time.time())}"
    job_uid = os.environ.get("BATCH_JOB_UID", "")
    task_index = os.environ.get("BATCH_TASK_INDEX", "0")
    project_id = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT") or ""
    region = os.environ.get("GCP_REGION") or "us-central1"
    bucket_name = os.environ.get("AE_ARCHIVE_BUCKET") or (f"{project_id}-alphaevolve-hpc-poc" if project_id else "")

    pillar_verifications: list[dict[str, Any]] = []

    for pillar_id, pillar_pkg in PILLAR_SPECS:
        p_t0 = time.perf_counter()
        seed_path = REPO_ROOT / "pillars" / pillar_id / "main.py"
        seed_source = seed_path.read_text(encoding="utf-8")

        seed_eval = _invoke_pillar_evaluator(pillar_pkg, seed_source, split="test")
        seed_score = float(seed_eval.get("overall_score", 0.0))

        # Load evolved block from results JSON if present
        result_file = RESULTS_DIR / f"{pillar_id}_results.json"
        evolved_score = seed_score
        evolved_gate = bool(seed_eval.get("gate_passed", True))
        if result_file.exists():
            saved = json.loads(result_file.read_text(encoding="utf-8"))
            eblock = _extract_winning_block(saved)
            if eblock:
                evolved_source = (
                    eblock
                    if (EVOLVE_START_MARKER in eblock and "def " in eblock and len(eblock) > len(seed_source) * 0.7)
                    else replace_evolve_block(seed_source, eblock)
                )
                evolved_eval = _invoke_pillar_evaluator(pillar_pkg, evolved_source, split="test")
                evolved_score = float(evolved_eval.get("overall_score", seed_score))
                evolved_gate = bool(evolved_eval.get("gate_passed", False))

        # Verify anti-reward-hacking gate rejects forbidden AST import
        sabotaged_source = replace_evolve_block(
            seed_source,
            "import subprocess\nsubprocess.run(['echo', 'hack'])\n",
        )
        sabotaged_eval = _invoke_pillar_evaluator(pillar_pkg, sabotaged_source, split="train")
        sabotage_blocked = float(sabotaged_eval.get("overall_score", 0.0)) <= HARD_GATE_PENALTY * 0.99

        pillar_verifications.append(
            {
                "pillar": pillar_id,
                "seed_holdout_score": round(seed_score, 6),
                "evolved_holdout_score": round(evolved_score, 6),
                "evolved_beats_seed": bool(evolved_score > seed_score),
                "evolved_hard_gate_passed": evolved_gate,
                "anti_reward_hacking_verified": sabotage_blocked,
                "duration_ms": round((time.perf_counter() - p_t0) * 1000.0, 1),
            }
        )

    summary = {
        "status": "SUCCEEDED",
        "batch_job_id": job_id,
        "batch_job_uid": job_uid,
        "task_index": task_index,
        "worker_hostname": socket.gethostname(),
        "region": region,
        "evaluated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "total_duration_ms": round((time.perf_counter() - t_start) * 1000.0, 1),
        "all_pillars_verified": all(
            p["evolved_beats_seed"] and p["evolved_hard_gate_passed"] and p["anti_reward_hacking_verified"]
            for p in pillar_verifications
        ),
        "pillars": pillar_verifications,
    }

    print(json.dumps(summary, indent=2))

    output_local = os.environ.get("AE_LOCAL_BATCH_OUT")
    if output_local:
        Path(output_local).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if bucket_name:
        client = storage.Client(project=project_id or None)
        bucket = client.bucket(bucket_name)
        for blob_path in (
            f"batch-validations/{job_id}/eval_summary.json",
            "batch-validations/latest_eval_summary.json",
        ):
            blob = bucket.blob(blob_path)
            blob.upload_from_string(
                json.dumps(summary, indent=2),
                content_type="application/json",
            )
            print(f"Uploaded Cloud Batch verification artifact to gs://{bucket_name}/{blob_path}")


if __name__ == "__main__":
    main()
