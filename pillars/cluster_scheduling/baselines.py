"""Reference scheduling policies (Naive FIFO vs. Tuned Fairshare) for simulator benchmarking."""

from __future__ import annotations

from typing import Any


def tuned_fairshare_dispatch(
    job: dict[str, Any],
    pool_utilization: dict[str, float],
    queue_backlog: dict[str, int],
) -> dict[str, Any]:
    """Hand-tuned static Slurm/Kubernetes fairshare baseline for 3-way UI comparison."""
    wclass = job["workload_class"]
    wait_min = float(job.get("wait_so_far_sec", 0.0)) / 60.0

    if wclass == "interactive_ide":
        prio = 500.0 + wait_min * 2.0
        pools = ["cpu_general", "batch_elastic"]
        tiers = ["committed_reservation", "gen4_on_demand"]
        storage = "shared_nfs_home"
        mem_gb = float(job["req_mem_gb"]) * 1.05
        wall_sec = float(job["req_walltime_sec"]) * 1.10
    elif wclass == "gpu_training":
        prio = 80.0 + wait_min * 0.5 - float(job["req_gpus"]) * 2.0
        pools = ["gpu_pool"]
        tiers = ["committed_reservation", "gen4_on_demand"]
        storage = "nvme_parallel_scratch" if job["req_gpus"] >= 4 else "shared_nfs_home"
        mem_gb = min(512.0, float(job["req_mem_gb"]) * 1.08)
        wall_sec = float(job["req_walltime_sec"]) * 1.10
    elif wclass == "high_mem_analytics":
        prio = 60.0 + wait_min * 0.4
        pools = ["ultramem_pool"]
        tiers = ["committed_reservation", "gen4_on_demand"]
        storage = "shared_nfs_home"
        mem_gb = min(1024.0, float(job["req_mem_gb"]) * 1.10)
        wall_sec = float(job["req_walltime_sec"]) * 1.10
    else:
        prio = 20.0 + wait_min * 0.3
        pools = ["batch_elastic", "cpu_general"]
        tiers = ["committed_reservation", "gen4_on_demand"]
        storage = "shared_nfs_home"
        mem_gb = min(64.0, float(job["req_mem_gb"]) * 1.05)
        wall_sec = float(job["req_walltime_sec"]) * 1.10

    return {
        "priority_score": prio,
        "preferred_pools": pools,
        "tier_preference": tiers,
        "storage_tier": storage,
        "alloc_mem_gb": mem_gb,
        "alloc_walltime_sec": wall_sec,
    }
