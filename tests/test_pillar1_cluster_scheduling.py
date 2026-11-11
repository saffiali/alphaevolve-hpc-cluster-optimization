"""Tests for Pillar 1 Cluster Scheduling simulator, trace splits, and hard SLA gates."""

from __future__ import annotations

from pillars.cluster_scheduling.evaluator import evaluate_program
from pillars.cluster_scheduling.trace_generator import generate_hpc_trace
from pillars.common.models import HARD_GATE_PENALTY


def test_trace_chronological_splits_and_determinism() -> None:
    t1 = generate_hpc_trace(num_jobs=10_000, seed=42, split="full")
    t2 = generate_hpc_trace(num_jobs=10_000, seed=42, split="full")
    assert len(t1) == 10_000
    assert t1[0] == t2[0] and t1[-1] == t2[-1]

    train = generate_hpc_trace(num_jobs=10_000, seed=42, split="train")
    val = generate_hpc_trace(num_jobs=10_000, seed=42, split="val")
    test = generate_hpc_trace(num_jobs=10_000, seed=42, split="test")
    assert len(train) == 6000
    assert len(val) == 2000
    assert len(test) == 2000
    assert train[0]["job_id"] == 0
    assert val[0]["job_id"] == 6000
    assert test[0]["job_id"] == 8000


def test_seed_program_passes_feasibility_and_captures_gpu_starvation() -> None:
    res = evaluate_program(split="test")
    assert res["valid"] is True
    assert res["gate_passed"] is True
    assert res["overall_score"] > HARD_GATE_PENALTY
    # Seed captures >7h (420m) GPU queue wait pathology while keeping interactive <120s
    assert res["metrics"]["gpu_p95_wait_min"] > 420.0
    assert res["metrics"]["interactive_p95_wait_sec"] < 120.0
    assert res["metrics"]["starved_or_dropped_jobs"] == 0


def test_sabotaged_candidate_dropping_gpu_jobs_returns_hard_penalty() -> None:
    sabotaged = '''# EVOLVE-BLOCK-START
def plan_job_dispatch(job, pool_utilization, queue_backlog):
    # Cheating attempt: route GPU jobs onto CPU-only pool so they get dropped!
    return {
        "priority_score": 1.0,
        "preferred_pools": ["batch_elastic"],
        "tier_preference": ["committed_reservation"],
        "storage_tier": "shared_nfs_home",
        "alloc_mem_gb": float(job["req_mem_gb"]),
        "alloc_walltime_sec": float(job["req_walltime_sec"]),
    }
# EVOLVE-BLOCK-END
'''
    res = evaluate_program(candidate_code=sabotaged, split="val")
    assert res["valid"] is False
    assert res["overall_score"] == HARD_GATE_PENALTY
    assert res["stage_reached"] == "stage_2_hard_gates"


def test_sabotaged_candidate_violating_interactive_sla_returns_hard_penalty() -> None:
    sabotaged = '''# EVOLVE-BLOCK-START
def plan_job_dispatch(job, pool_utilization, queue_backlog):
    # Violates interactive <120s SLA by forcing interactive IDE onto tiny congested pool at lowest priority
    if job["workload_class"] == "interactive_ide":
        return {
            "priority_score": -99999.0,
            "preferred_pools": ["gpu_pool"],
            "tier_preference": ["committed_reservation"],
            "storage_tier": "shared_nfs_home",
            "alloc_mem_gb": 512.0,
            "alloc_walltime_sec": float(job["req_walltime_sec"]),
        }
    return {
        "priority_score": 100.0,
        "preferred_pools": ["gpu_pool"] if job["req_gpus"] > 0 else ["cpu_general", "ultramem_pool", "batch_elastic"],
        "tier_preference": ["committed_reservation", "gen4_on_demand"],
        "storage_tier": "shared_nfs_home",
        "alloc_mem_gb": float(job["req_mem_gb"]),
        "alloc_walltime_sec": float(job["req_walltime_sec"]),
    }
# EVOLVE-BLOCK-END
'''
    res = evaluate_program(candidate_code=sabotaged, split="val")
    assert res["valid"] is False
    assert res["overall_score"] == HARD_GATE_PENALTY


def test_sabotaged_candidate_under_provisioning_memory_returns_hard_penalty() -> None:
    sabotaged = '''# EVOLVE-BLOCK-START
def plan_job_dispatch(job, pool_utilization, queue_backlog):
    # Cheating bin-packing by shrinking memory 50%, triggering OOM cascade > 22% floor
    pools = ["gpu_pool"] if job["req_gpus"] > 0 else ["cpu_general", "ultramem_pool", "batch_elastic"]
    return {
        "priority_score": 50.0 if job["workload_class"] == "interactive_ide" else 1.0,
        "preferred_pools": pools,
        "tier_preference": ["committed_reservation", "gen4_on_demand"],
        "storage_tier": "shared_nfs_home",
        "alloc_mem_gb": max(1.0, float(job["req_mem_gb"]) * 0.45),
        "alloc_walltime_sec": float(job["req_walltime_sec"]),
    }
# EVOLVE-BLOCK-END
'''
    res = evaluate_program(candidate_code=sabotaged, split="val")
    assert res["valid"] is False
    assert res["overall_score"] == HARD_GATE_PENALTY
