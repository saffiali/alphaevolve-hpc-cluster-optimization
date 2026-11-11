"""Deterministic 10,000-job enterprise scientific & AI HPC trace generator.

Models four heterogeneous workload classes across Slurm partitions, GKE Custom
Compute Classes (CCC), and Cloud Batch:
  1. `interactive_ide`: 2-8 vCPU interactive notebook/IDE sessions (< 2 min SLA).
  2. `gpu_training`: 1-8 GPU distributed AI training & molecular simulations.
  3. `high_mem_analytics`: 16-64 vCPU, 192-960 GB RAM genomics & structural solvers.
  4. `batch_array`: High-throughput fault-tolerant CPU parameter sweeps.

Captures real-world operational pathologies without exposing hidden ground-truth
execution durations or peak memory footprints to candidate scheduling policies.
"""

from __future__ import annotations

from typing import Any
import numpy as np

CLUSTER_NODE_POOLS: dict[str, dict[str, Any]] = {
    "cpu_general": {
        "machine_family": "c4d-standard-32",
        "orchestrator": "slurm_partition",
        "cpus_per_node": 32,
        "mem_gb_per_node": 128.0,
        "gpus_per_node": 0,
        "reserved_nodes": 4,
        "dws_max_nodes": 6,
        "on_demand_max_nodes": 28,
        "spot_max_nodes": 36,
        "base_hourly_usd": 1.52,
    },
    "ultramem_pool": {
        "machine_family": "m3-ultramem-64",
        "orchestrator": "slurm_partition",
        "cpus_per_node": 64,
        "mem_gb_per_node": 1024.0,
        "gpus_per_node": 0,
        "reserved_nodes": 5,
        "dws_max_nodes": 8,
        "on_demand_max_nodes": 10,
        "spot_max_nodes": 6,
        "base_hourly_usd": 9.80,
    },
    "gpu_pool": {
        "machine_family": "a3-highgpu-8g",
        "orchestrator": "gke_ccc",
        "cpus_per_node": 64,
        "mem_gb_per_node": 512.0,
        "gpus_per_node": 8,
        "reserved_nodes": 4,
        "dws_max_nodes": 10,
        "on_demand_max_nodes": 2,
        "spot_max_nodes": 6,
        "base_hourly_usd": 31.20,
    },
    "batch_elastic": {
        "machine_family": "n4-standard-16",
        "orchestrator": "cloud_batch",
        "cpus_per_node": 16,
        "mem_gb_per_node": 64.0,
        "gpus_per_node": 0,
        "reserved_nodes": 5,
        "dws_max_nodes": 6,
        "on_demand_max_nodes": 32,
        "spot_max_nodes": 64,
        "base_hourly_usd": 0.76,
    },
}

PROVISIONING_TIERS: dict[str, dict[str, float]] = {
    "committed_reservation": {
        "cost_multiplier": 0.42,  # Effective amortized CUD/Reservation rate
        "startup_overhead_sec": 8.0,
        "preemption_multiplier": 0.0,
    },
    "dws_flex_start": {
        "cost_multiplier": 0.68,  # Dynamic Workload Scheduler Flex Start discount
        "startup_overhead_sec": 42.0,
        "preemption_multiplier": 0.0,
    },
    "gen4_on_demand": {
        "cost_multiplier": 1.00,
        "startup_overhead_sec": 28.0,
        "preemption_multiplier": 0.0,
    },
    "spot": {
        "cost_multiplier": 0.32,
        "startup_overhead_sec": 24.0,
        "preemption_multiplier": 1.0,
    },
}


def generate_hpc_trace(
    num_jobs: int = 10_000,
    seed: int = 42,
    split: str = "full",
    *,
    burst_multiplier: float = 1.0,
    gpu_share_override: float | None = None,
    spot_preemption_rate: float = 0.08,
    dws_availability: float = 0.85,
) -> list[dict[str, Any]]:
    """Generate a deterministic 10,000-job enterprise HPC trace with chronological splits.

    Splits (chronological by arrival timestamp to prevent temporal leakage):
      - `train`: first 60% of jobs (0 .. 5,999 for 10,000 jobs)
      - `val`:   next 20% of jobs  (6,000 .. 7,999)
      - `test`:  final 20% of jobs (8,000 .. 9,999, strictly held out)
      - `full`:  all 10,000 jobs
    """
    if split not in ("full", "train", "val", "test"):
        raise ValueError(f"Invalid split '{split}'. Expected full, train, val, or test.")

    rng = np.random.default_rng(seed)

    base_gpu_share = 0.16 if gpu_share_override is None else float(np.clip(gpu_share_override, 0.05, 0.55))
    rem = 1.0 - base_gpu_share
    # Workload mix: interactive_ide (~24%), gpu_training (~16%), high_mem_analytics (~15%), batch_array (~45%)
    probs = np.array(
        [
            0.285 * rem,
            base_gpu_share,
            0.180 * rem,
            (1.0 - 0.285 - 0.180) * rem,
        ],
        dtype=np.float64,
    )
    probs /= probs.sum()

    classes = ("interactive_ide", "gpu_training", "high_mem_analytics", "batch_array")
    class_idx = rng.choice(4, size=num_jobs, p=probs)

    # Diurnal + bursty inhomogeneous Poisson arrival process (~14 days across 10,000 jobs)
    # Higher burst_multiplier compresses inter-arrival times to stress-test spillover.
    effective_rate_scale = max(0.35, float(burst_multiplier))
    t_sec = 0.0
    jobs: list[dict[str, Any]] = []

    for jid in range(num_jobs):
        hour_of_day = (t_sec / 3600.0) % 24.0
        # Business-hour surge between 08:00 and 19:00 UTC + periodic submission bursts
        diurnal = 1.75 if 8.0 <= hour_of_day <= 19.0 else 0.55
        wave_burst = 2.1 if (jid % 500) < 85 else 1.0
        mean_ia = 122.0 / (diurnal * wave_burst * effective_rate_scale)
        t_sec += float(rng.exponential(mean_ia))

        wtype = classes[int(class_idx[jid])]

        if wtype == "interactive_ide":
            req_cpus = int(rng.choice([2, 4, 8], p=[0.45, 0.40, 0.15]))
            req_mem_gb = float(req_cpus * int(rng.choice([4, 8], p=[0.65, 0.35])))
            req_gpus = 0
            orchestrator = "gke_ccc" if rng.random() < 0.55 else "slurm_partition"
            true_runtime = float(rng.uniform(900.0, 4200.0))
            req_walltime = true_runtime * float(rng.uniform(1.15, 1.60))
            true_mem_ratio = float(rng.uniform(0.65, 1.03))
            hist_p95_mem_ratio = min(1.35, max(0.85, true_mem_ratio * float(rng.uniform(0.96, 1.08))))
            io_profile = "light_interactive" if rng.random() < 0.72 else "metadata_heavy"
            checkpointable = False
            priority_tier = 3

        elif wtype == "gpu_training":
            req_gpus = int(rng.choice([1, 4, 8], p=[0.42, 0.36, 0.22]))
            req_cpus = req_gpus * 8
            req_mem_gb = float(req_gpus * int(rng.choice([32, 48, 60], p=[0.35, 0.40, 0.25])))
            orchestrator = "gke_ccc" if rng.random() < 0.60 else "slurm_partition"
            true_runtime = float(rng.uniform(2400.0, 11800.0))
            # ~22% of GPU jobs under-request walltime or memory without historical right-sizing
            under_sized = rng.random() < 0.24
            req_walltime = true_runtime * (float(rng.uniform(0.82, 0.98)) if under_sized else float(rng.uniform(1.15, 1.45)))
            true_mem_ratio = float(rng.uniform(1.04, 1.24)) if under_sized else float(rng.uniform(0.72, 0.96))
            hist_p95_mem_ratio = min(1.35, max(0.90, true_mem_ratio * float(rng.uniform(0.98, 1.07))))
            io_profile = "high_iops_scratch" if rng.random() < 0.75 else "metadata_heavy"
            checkpointable = bool(rng.random() < 0.35)
            priority_tier = 2

        elif wtype == "high_mem_analytics":
            req_cpus = int(rng.choice([16, 32, 64], p=[0.30, 0.45, 0.25]))
            req_mem_gb = float(rng.choice([160.0, 256.0, 384.0, 640.0], p=[0.25, 0.35, 0.25, 0.15]))
            req_gpus = 0
            orchestrator = "slurm_partition"
            true_runtime = float(rng.uniform(1800.0, 8400.0))
            # ~34% of high-memory genomics/structural jobs under-estimate peak RAM or walltime
            under_sized = rng.random() < 0.34
            req_walltime = true_runtime * (float(rng.uniform(0.80, 0.97)) if under_sized else float(rng.uniform(1.12, 1.50)))
            true_mem_ratio = float(rng.uniform(1.05, 1.32)) if under_sized else float(rng.uniform(0.75, 0.97))
            hist_p95_mem_ratio = min(1.42, max(0.92, true_mem_ratio * float(rng.uniform(0.99, 1.08))))
            io_profile = "high_iops_scratch" if rng.random() < 0.68 else "metadata_heavy"
            checkpointable = False
            priority_tier = 2

        else:  # batch_array
            req_cpus = int(rng.choice([2, 4, 8, 16], p=[0.35, 0.35, 0.20, 0.10]))
            req_mem_gb = float(req_cpus * int(rng.choice([2, 4], p=[0.55, 0.45])))
            req_gpus = 0
            orchestrator = "cloud_batch" if rng.random() < 0.65 else "slurm_partition"
            true_runtime = float(rng.uniform(420.0, 3200.0))
            under_sized = rng.random() < 0.12
            req_walltime = true_runtime * (float(rng.uniform(0.84, 0.98)) if under_sized else float(rng.uniform(1.20, 1.65)))
            true_mem_ratio = float(rng.uniform(1.03, 1.18)) if under_sized else float(rng.uniform(0.68, 0.94))
            hist_p95_mem_ratio = min(1.30, max(0.88, true_mem_ratio * float(rng.uniform(0.98, 1.06))))
            io_profile = "streaming_throughput" if rng.random() < 0.74 else "metadata_heavy"
            checkpointable = bool(rng.random() < 0.88)
            priority_tier = 1

        hist_p95_runtime = true_runtime * float(rng.uniform(0.95, 1.12))
        spot_hazard_u = float(rng.random())

        jobs.append(
            {
                "job_id": jid,
                "arrival_sec": round(t_sec, 2),
                "workload_class": wtype,
                "orchestrator_target": orchestrator,
                "req_cpus": req_cpus,
                "req_mem_gb": round(req_mem_gb, 2),
                "req_gpus": req_gpus,
                "req_walltime_sec": round(req_walltime, 1),
                "hist_p95_mem_ratio": round(hist_p95_mem_ratio, 3),
                "hist_p95_runtime_sec": round(hist_p95_runtime, 1),
                "io_profile": io_profile,
                "checkpointable": checkpointable,
                "priority_tier": priority_tier,
                # Hidden ground-truth fields stripped before invoking candidate policy:
                "_true_runtime_sec": round(true_runtime, 2),
                "_true_peak_mem_gb": round(req_mem_gb * true_mem_ratio, 2),
                "_spot_u": spot_hazard_u,
                "_spot_preemption_rate": float(spot_preemption_rate),
                "_dws_availability": float(dws_availability),
            }
        )

    n_train = int(0.60 * num_jobs)
    n_val = int(0.80 * num_jobs)

    if split == "train":
        selected = jobs[:n_train]
    elif split == "val":
        selected = jobs[n_train:n_val]
    elif split == "test":
        selected = jobs[n_val:]
    else:
        selected = jobs

    if not selected:
        return []

    # Re-anchor split start time to 0.0 so each evaluation window starts cleanly
    t0 = selected[0]["arrival_sec"]
    reanchored: list[dict[str, Any]] = []
    for rec in selected:
        item = dict(rec)
        item["arrival_sec"] = round(rec["arrival_sec"] - t0, 2)
        reanchored.append(item)
    return reanchored
