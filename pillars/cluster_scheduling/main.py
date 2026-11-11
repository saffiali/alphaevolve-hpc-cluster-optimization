"""Seed program for Pillar 1: Enterprise HPC & AI Cluster Scheduling Policy.

Optimizes job priority ordering, shape-guardrailed node pool placement,
multi-tier capacity spillover (Committed Reservations -> DWS Flex Start ->
Gen-4 On-Demand -> Spot), historical memory/walltime right-sizing, and
I/O-aware storage tier routing across Slurm, GKE Custom Compute Classes,
and Cloud Batch.
"""

from __future__ import annotations

import math
from typing import Any

VALID_POOLS = ("cpu_general", "ultramem_pool", "gpu_pool", "batch_elastic")
VALID_TIERS = ("committed_reservation", "dws_flex_start", "gen4_on_demand", "spot")
VALID_STORAGE = ("shared_nfs_home", "nvme_parallel_scratch", "object_storage_tier")


# EVOLVE-BLOCK-START
def plan_job_dispatch(
    job: dict[str, Any],
    pool_utilization: dict[str, float],
    queue_backlog: dict[str, int],
) -> dict[str, Any]:
    """Return scheduling dispatch plan for an arriving or queued HPC workload.

    Args:
        job: Observable job attributes (`job_id`, `arrival_sec`, `workload_class`,
             `orchestrator_target`, `req_cpus`, `req_mem_gb`, `req_gpus`,
             `req_walltime_sec`, `hist_p95_mem_ratio`, `hist_p95_runtime_sec`,
             `io_profile`, `checkpointable`, `priority_tier`, `wait_so_far_sec`).
        pool_utilization: Current active vCPU/GPU utilization ratio [0.0..1.0] per pool.
        queue_backlog: Count of queued jobs currently waiting per workload_class.

    Returns:
        Dictionary containing:
          - `priority_score`: float (higher dispatches earlier in scheduling epoch)
          - `preferred_pools`: ordered list of target node pools from `VALID_POOLS`
          - `tier_preference`: ordered list of capacity tiers from `VALID_TIERS`
          - `storage_tier`: target storage mount from `VALID_STORAGE`
          - `alloc_mem_gb`: memory reservation in GB assigned to the job
          - `alloc_walltime_sec`: walltime limit in seconds assigned to the job
    """
    # Naive Seed Policy:
    # 1. Pure FIFO arrival ordering (except mild interactive tie-break) with no SJF or aging.
    # 2. First-fit pool list without shape guardrails: small CPU/interactive jobs freely
    #    spill onto 64-core Ultramem and GPU nodes when general CPU pools fill up,
    #    fragmenting whole-node GPU capacity and causing >7h P95 GPU queue starvation.
    # 3. Static On-Demand fallback (ignores DWS Flex Start for GPUs and Spot for checkpointable batch).
    # 4. Raw user memory/walltime requests without P95 historical right-sizing, causing OOM/timeouts.
    # 5. Default shared NFS `/home` mount for all jobs, creating metadata IOPS contention.
    priority_score = -float(job["arrival_sec"]) * 1e-4
    if job["workload_class"] == "interactive_ide":
        priority_score += 5.0

    if job["req_gpus"] > 0:
        preferred_pools = ["gpu_pool"]
    else:
        preferred_pools = ["cpu_general", "gpu_pool", "ultramem_pool", "batch_elastic"]

    tier_preference = ["committed_reservation", "gen4_on_demand"]
    storage_tier = "shared_nfs_home"

    return {
        "priority_score": priority_score,
        "preferred_pools": preferred_pools,
        "tier_preference": tier_preference,
        "storage_tier": storage_tier,
        "alloc_mem_gb": float(job["req_mem_gb"]),
        "alloc_walltime_sec": float(job["req_walltime_sec"]),
    }
# EVOLVE-BLOCK-END
