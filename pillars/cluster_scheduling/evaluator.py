"""Discrete-event cluster simulator and 3-stage anti-reward-hacking evaluator for Pillar 1."""

from __future__ import annotations

import copy
import heapq
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from pillars.cluster_scheduling.trace_generator import (
    CLUSTER_NODE_POOLS,
    PROVISIONING_TIERS,
    generate_hpc_trace,
)
from pillars.common.models import HARD_GATE_PENALTY, make_evaluation_result
from pillars.common.sandbox import (
    CandidateTimeoutError,
    SandboxViolationError,
    execute_candidate_module,
    run_with_timeout,
    validate_candidate_source,
)

PILLAR_DIR = Path(__file__).resolve().parent
SEED_FILE_PATH = PILLAR_DIR / "main.py"
ALLOWED_IMPORTS = frozenset({"math", "typing", "__future__", "collections", "statistics"})
REQUIRED_FUNCTIONS = ("plan_job_dispatch",)

# Hard anti-reward-hacking constraint floors
MAX_STARVED_OR_DROPPED_JOBS = 0
MAX_INTERACTIVE_P95_WAIT_SEC = 120.0  # Strict < 2 min SLA for interactive IDE startups
MAX_OOM_RATE_FLOOR = 0.22             # Hard OOM rate ceiling (22%) blocking memory under-provisioning
MAX_SPOT_KILL_RATE_FLOOR = 0.04       # Hard Spot kill ceiling (4%) blocking unsafe Spot placement
STARVATION_CEILING_SEC = 86_400.0     # 24-hour starvation cutoff


class _NodeSlot:
    """Tracks active resource allocations on a single provisioned cluster node."""

    __slots__ = (
        "pool_name",
        "tier_name",
        "cpus_total",
        "mem_total",
        "gpus_total",
        "cpus_used",
        "mem_used",
        "gpus_used",
        "active_jobs",
        "squatter_cpu_jobs",
    )

    def __init__(
        self,
        pool_name: str,
        tier_name: str,
        cpus_total: int,
        mem_total: float,
        gpus_total: int,
    ) -> None:
        self.pool_name = pool_name
        self.tier_name = tier_name
        self.cpus_total = cpus_total
        self.mem_total = mem_total
        self.gpus_total = gpus_total
        self.cpus_used = 0
        self.mem_used = 0.0
        self.gpus_used = 0
        self.active_jobs = 0
        self.squatter_cpu_jobs = 0

    def can_fit(self, cpus: int, mem_gb: float, gpus: int) -> bool:
        return (
            (self.cpus_total - self.cpus_used) >= cpus
            and (self.mem_total - self.mem_used) >= mem_gb - 1e-6
            and (self.gpus_total - self.gpus_used) >= gpus
        )

    def allocate(self, cpus: int, mem_gb: float, gpus: int) -> None:
        self.cpus_used += cpus
        self.mem_used += mem_gb
        self.gpus_used += gpus
        self.active_jobs += 1
        if gpus == 0 and cpus <= 8 and self.cpus_total >= 64:
            self.squatter_cpu_jobs += 1

    def release(self, cpus: int, mem_gb: float, gpus: int) -> None:
        self.cpus_used = max(0, self.cpus_used - cpus)
        self.mem_used = max(0.0, self.mem_used - mem_gb)
        self.gpus_used = max(0, self.gpus_used - gpus)
        self.active_jobs = max(0, self.active_jobs - 1)


def _strip_hidden_fields(job: dict[str, Any], wait_so_far_sec: float) -> dict[str, Any]:
    """Return observable job metadata with zero hidden ground-truth leakage."""
    return {
        "job_id": job["job_id"],
        "arrival_sec": job["arrival_sec"],
        "workload_class": job["workload_class"],
        "orchestrator_target": job["orchestrator_target"],
        "req_cpus": job["req_cpus"],
        "req_mem_gb": job["req_mem_gb"],
        "req_gpus": job["req_gpus"],
        "req_walltime_sec": job["req_walltime_sec"],
        "hist_p95_mem_ratio": job["hist_p95_mem_ratio"],
        "hist_p95_runtime_sec": job["hist_p95_runtime_sec"],
        "io_profile": job["io_profile"],
        "checkpointable": job["checkpointable"],
        "priority_tier": job["priority_tier"],
        "wait_so_far_sec": round(wait_so_far_sec, 2),
    }


def run_policy_simulation(
    dispatch_fn: Callable[[dict[str, Any], dict[str, float], dict[str, int]], dict[str, Any]],
    jobs: list[dict[str, Any]],
    *,
    dws_availability_override: float | None = None,
    spot_preemption_override: float | None = None,
) -> dict[str, Any]:
    """Execute discrete-event cluster simulation for a candidate scheduling policy."""
    if not jobs:
        raise ValueError("Job trace cannot be empty.")

    pools_cfg = copy.deepcopy(CLUSTER_NODE_POOLS)
    dws_avail = (
        float(dws_availability_override)
        if dws_availability_override is not None
        else float(jobs[0].get("_dws_availability", 0.85))
    )
    spot_rate = (
        float(spot_preemption_override)
        if spot_preemption_override is not None
        else float(jobs[0].get("_spot_preemption_rate", 0.08))
    )

    # Pre-provision baseline committed reservation nodes; elastic nodes spin up on demand
    nodes_by_pool_tier: dict[tuple[str, str], list[_NodeSlot]] = {}
    max_nodes_by_pool_tier: dict[tuple[str, str], int] = {}

    for pname, pspec in pools_cfg.items():
        res_n = int(pspec["reserved_nodes"])
        dws_n = max(0, int(round(pspec["dws_max_nodes"] * np.clip(dws_avail, 0.0, 1.5))))
        od_n = int(pspec["on_demand_max_nodes"])
        sp_n = int(pspec["spot_max_nodes"])

        nodes_by_pool_tier[(pname, "committed_reservation")] = [
            _NodeSlot(pname, "committed_reservation", pspec["cpus_per_node"], pspec["mem_gb_per_node"], pspec["gpus_per_node"])
            for _ in range(res_n)
        ]
        for tname, cap in (
            ("dws_flex_start", dws_n),
            ("gen4_on_demand", od_n),
            ("spot", sp_n),
        ):
            nodes_by_pool_tier[(pname, tname)] = []
            max_nodes_by_pool_tier[(pname, tname)] = cap
        max_nodes_by_pool_tier[(pname, "committed_reservation")] = res_n

    # Event heap: (timestamp_sec, event_type_order, tie_id, payload)
    # event_type_order: 0 = job_release, 1 = job_arrival_or_retry
    event_heap: list[tuple[float, int, int, Any]] = []
    tie_counter = 0
    for j in jobs:
        heapq.heappush(event_heap, (float(j["arrival_sec"]), 1, tie_counter, j))
        tie_counter += 1

    waiting_queue: list[dict[str, Any]] = []
    active_nfs_heavy_jobs = 0

    wait_times_all: list[float] = []
    wait_by_class: dict[str, list[float]] = {
        "interactive_ide": [],
        "gpu_training": [],
        "high_mem_analytics": [],
        "batch_array": [],
    }
    total_compute_cost_usd = 0.0
    total_storage_cost_usd = 0.0
    oom_failures = 0
    timeout_failures = 0
    spot_failures = 0
    starved_or_dropped = 0
    fragmentation_samples: list[float] = []
    squatter_placements = 0
    tier_job_counts: dict[str, int] = {k: 0 for k in PROVISIONING_TIERS}
    storage_job_counts: dict[str, int] = {
        "shared_nfs_home": 0,
        "nvme_parallel_scratch": 0,
        "object_storage_tier": 0,
    }

    def _compute_pool_util() -> dict[str, float]:
        util: dict[str, float] = {}
        for pname, pspec in pools_cfg.items():
            res_nodes = nodes_by_pool_tier[(pname, "committed_reservation")]
            if not res_nodes:
                util[pname] = 0.0
                continue
            if pspec["gpus_per_node"] > 0:
                used = sum(n.gpus_used for n in res_nodes)
                tot = len(res_nodes) * pspec["gpus_per_node"]
            else:
                used = sum(n.cpus_used for n in res_nodes)
                tot = len(res_nodes) * pspec["cpus_per_node"]
            util[pname] = round(used / max(1, tot), 3)
        return util

    def _sample_fragmentation() -> None:
        # Measure stranded GPU and Ultramem capacity caused by mismatched bin-packing
        stranded_ratios: list[float] = []
        for pname in ("gpu_pool", "ultramem_pool", "cpu_general"):
            for tier_key in PROVISIONING_TIERS:
                for node in nodes_by_pool_tier[(pname, tier_key)]:
                    if node.active_jobs == 0:
                        continue
                    cpu_frac = node.cpus_used / node.cpus_total
                    mem_frac = node.mem_used / node.mem_total
                    if node.gpus_total > 0:
                        gpu_frac = node.gpus_used / node.gpus_total
                        # If CPUs/RAM are consumed on a GPU node while GPUs sit idle, GPUs are stranded!
                        imbalance = max(cpu_frac, mem_frac) - gpu_frac
                        idle_waste = 1.0 - 0.5 * (gpu_frac + max(cpu_frac, mem_frac))
                        stranded_ratios.append(max(0.0, 0.65 * max(0.0, imbalance) + 0.35 * idle_waste))
                    else:
                        dominant = max(cpu_frac, mem_frac)
                        sub = min(cpu_frac, mem_frac)
                        stranded_ratios.append(max(0.0, 0.55 * (dominant - sub) + 0.45 * (1.0 - dominant)))
        if stranded_ratios:
            fragmentation_samples.append(float(np.mean(stranded_ratios)))

    def _try_dispatch_waiting(now_sec: float) -> float | None:
        nonlocal tie_counter, active_nfs_heavy_jobs, total_compute_cost_usd
        nonlocal total_storage_cost_usd, oom_failures, timeout_failures
        nonlocal spot_failures, starved_or_dropped, squatter_placements

        if not waiting_queue:
            return None

        pool_util = _compute_pool_util()
        backlog: dict[str, int] = {
            "interactive_ide": 0,
            "gpu_training": 0,
            "high_mem_analytics": 0,
            "batch_array": 0,
        }
        for wj in waiting_queue:
            backlog[wj["workload_class"]] = backlog.get(wj["workload_class"], 0) + 1

        scored_plans: list[tuple[float, int, dict[str, Any], dict[str, Any]]] = []
        still_waiting: list[dict[str, Any]] = []

        for idx, wj in enumerate(waiting_queue):
            wait_so_far = max(0.0, now_sec - float(wj["arrival_sec"]))
            if wait_so_far > STARVATION_CEILING_SEC:
                starved_or_dropped += 1
                continue
            view = _strip_hidden_fields(wj, wait_so_far)
            plan = dispatch_fn(view, pool_util, backlog)
            if not isinstance(plan, dict):
                starved_or_dropped += 1
                continue
            prio = float(plan.get("priority_score", 0.0))
            if not math.isfinite(prio):
                starved_or_dropped += 1
                continue
            scored_plans.append((-prio, idx, wj, plan))

        scored_plans.sort(key=lambda item: (item[0], item[1]))

        for _, _, wj, plan in scored_plans:
            pref_pools = [p for p in plan.get("preferred_pools", []) if p in pools_cfg]
            pref_tiers = [t for t in plan.get("tier_preference", []) if t in PROVISIONING_TIERS]
            storage_tier = str(plan.get("storage_tier", "shared_nfs_home"))
            if storage_tier not in storage_job_counts:
                storage_tier = "shared_nfs_home"

            req_cpus = int(wj["req_cpus"])
            req_gpus = int(wj["req_gpus"])
            alloc_mem = float(plan.get("alloc_mem_gb", wj["req_mem_gb"]))
            alloc_wall = float(plan.get("alloc_walltime_sec", wj["req_walltime_sec"]))

            if not pref_pools or not pref_tiers or not math.isfinite(alloc_mem) or not math.isfinite(alloc_wall):
                starved_or_dropped += 1
                continue
            if alloc_mem <= 0.0 or alloc_wall <= 0.0:
                starved_or_dropped += 1
                continue

            # Check if ANY requested pool can ever physically fit this job shape
            physically_feasible = False
            chosen_node: _NodeSlot | None = None
            chosen_pool: str = ""
            chosen_tier: str = ""

            # Evaluate tier-outer, pool-inner so warm committed reservations across candidate's
            # allowed pools are checked before spinning up cold elastic burst VMs.
            feasible_pools = [
                pname
                for pname in pref_pools
                if req_cpus <= pools_cfg[pname]["cpus_per_node"]
                and req_gpus <= pools_cfg[pname]["gpus_per_node"]
                and alloc_mem <= pools_cfg[pname]["mem_gb_per_node"]
            ]
            physically_feasible = len(feasible_pools) > 0

            for tname in pref_tiers:
                for pname in feasible_pools:
                    if tname == "dws_flex_start" and pname not in ("gpu_pool", "ultramem_pool", "batch_elastic"):
                        continue
                    pspec = pools_cfg[pname]
                    slot_list = nodes_by_pool_tier[(pname, tname)]
                    for existing in slot_list:
                        if existing.can_fit(req_cpus, alloc_mem, req_gpus):
                            chosen_node = existing
                            chosen_pool = pname
                            chosen_tier = tname
                            break
                    if chosen_node is not None:
                        break

                    if len(slot_list) < max_nodes_by_pool_tier[(pname, tname)]:
                        new_node = _NodeSlot(
                            pname,
                            tname,
                            pspec["cpus_per_node"],
                            pspec["mem_gb_per_node"],
                            pspec["gpus_per_node"],
                        )
                        slot_list.append(new_node)
                        chosen_node = new_node
                        chosen_pool = pname
                        chosen_tier = tname
                        break
                if chosen_node is not None:
                    break

            if not physically_feasible:
                # Candidate routed job to incompatible pools or requested impossible memory > node max
                starved_or_dropped += 1
                continue

            if chosen_node is None:
                still_waiting.append(wj)
                continue

            # Dispatch job onto chosen_node!
            chosen_node.allocate(req_cpus, alloc_mem, req_gpus)
            if req_gpus == 0 and req_cpus <= 8 and pools_cfg[chosen_pool]["cpus_per_node"] >= 64:
                squatter_placements += 1

            tier_spec = PROVISIONING_TIERS[chosen_tier]
            startup_delay = tier_spec["startup_overhead_sec"]
            # Committed reservation nodes are warm; elastic nodes only incur cold-boot on first job
            if chosen_tier != "committed_reservation" and chosen_node.active_jobs > 1:
                startup_delay *= 0.25

            total_wait_sec = max(0.0, now_sec - float(wj["arrival_sec"])) + startup_delay
            wait_times_all.append(total_wait_sec)
            wait_by_class[wj["workload_class"]].append(total_wait_sec)
            tier_job_counts[chosen_tier] += 1
            storage_job_counts[storage_tier] += 1

            # Storage & metadata contention physics
            io_prof = wj["io_profile"]
            is_nfs_heavy = storage_tier == "shared_nfs_home" and io_prof in ("metadata_heavy", "high_iops_scratch")
            if is_nfs_heavy:
                active_nfs_heavy_jobs += 1

            io_speed_factor = 1.0
            if storage_tier == "shared_nfs_home":
                if io_prof in ("metadata_heavy", "high_iops_scratch"):
                    contention = min(1.8, active_nfs_heavy_jobs / 18.0)
                    io_speed_factor = 1.0 + 0.34 * contention
                storage_rate_hr = 0.012
            elif storage_tier == "nvme_parallel_scratch":
                if io_prof in ("high_iops_scratch", "metadata_heavy"):
                    io_speed_factor = 0.86
                storage_rate_hr = 0.048
            else:  # object_storage_tier
                if io_prof == "streaming_throughput":
                    io_speed_factor = 0.92
                elif io_prof in ("light_interactive", "metadata_heavy"):
                    io_speed_factor = 1.32
                storage_rate_hr = 0.009

            eff_true_runtime = float(wj["_true_runtime_sec"]) * io_speed_factor
            true_peak_mem = float(wj["_true_peak_mem_gb"])

            # Check OOM, Spot preemption, or Walltime Timeout
            failed_oom = alloc_mem < true_peak_mem
            spot_prob = 0.0
            if chosen_tier == "spot":
                job_hrs = eff_true_runtime / 3600.0
                spot_prob = 1.0 - math.exp(-spot_rate * tier_spec["preemption_multiplier"] * job_hrs * 1.6)
            spot_interrupted = chosen_tier == "spot" and float(wj["_spot_u"]) < spot_prob

            if failed_oom:
                oom_failures += 1
                billed_sec = startup_delay + min(eff_true_runtime * 0.55, alloc_wall)
            elif spot_interrupted and not wj["checkpointable"]:
                spot_failures += 1
                billed_sec = startup_delay + eff_true_runtime * 0.60
            else:
                if spot_interrupted and wj["checkpointable"]:
                    # Checkpoint restart adds 16% wall-time overhead instead of failing
                    eff_true_runtime *= 1.16
                if eff_true_runtime > alloc_wall:
                    timeout_failures += 1
                    billed_sec = startup_delay + alloc_wall
                else:
                    billed_sec = startup_delay + eff_true_runtime

            pspec = pools_cfg[chosen_pool]
            if pspec["gpus_per_node"] > 0 and req_gpus > 0:
                share_frac = max(req_gpus / pspec["gpus_per_node"], req_cpus / pspec["cpus_per_node"])
            else:
                share_frac = max(req_cpus / pspec["cpus_per_node"], alloc_mem / pspec["mem_gb_per_node"])
                # Small squatter jobs on GPU/Ultramem nodes strand disproportionate host value
                if pspec["gpus_per_node"] > 0 and req_gpus == 0:
                    share_frac = max(share_frac, 0.28)

            billed_hrs = billed_sec / 3600.0
            job_compute_cost = billed_hrs * pspec["base_hourly_usd"] * share_frac * tier_spec["cost_multiplier"]
            total_compute_cost_usd += job_compute_cost
            total_storage_cost_usd += billed_hrs * storage_rate_hr

            finish_sec = now_sec + billed_sec
            heapq.heappush(
                event_heap,
                (
                    finish_sec,
                    0,
                    tie_counter,
                    (chosen_node, req_cpus, alloc_mem, req_gpus, is_nfs_heavy),
                ),
            )
            tie_counter += 1

        waiting_queue[:] = still_waiting
        _sample_fragmentation()
        if waiting_queue:
            return now_sec + 60.0
        return None

    last_dispatch_sec = -1e9
    while event_heap:
        ev_time, ev_kind, _, payload = heapq.heappop(event_heap)
        if ev_kind == 0:
            node_slot, c_rel, m_rel, g_rel, was_nfs_heavy = payload
            node_slot.release(c_rel, m_rel, g_rel)
            if was_nfs_heavy:
                active_nfs_heavy_jobs = max(0, active_nfs_heavy_jobs - 1)
            # Trigger dispatch when resources free up (coalesced to >= 5s apart)
            if waiting_queue and (ev_time - last_dispatch_sec) >= 5.0:
                _try_dispatch_waiting(ev_time)
                last_dispatch_sec = ev_time
        elif ev_kind == 1:
            waiting_queue.append(payload)
            # Immediate fast-path check for interactive jobs or batch epoch
            if payload["workload_class"] == "interactive_ide" or (ev_time - last_dispatch_sec) >= 30.0 or not event_heap:
                nxt = _try_dispatch_waiting(ev_time)
                last_dispatch_sec = ev_time
                if nxt is not None and not event_heap:
                    heapq.heappush(event_heap, (nxt, 2, tie_counter, None))
                    tie_counter += 1
        else:
            nxt = _try_dispatch_waiting(ev_time)
            last_dispatch_sec = ev_time
            if nxt is not None and waiting_queue and not event_heap:
                heapq.heappush(event_heap, (nxt, 2, tie_counter, None))
                tie_counter += 1

    if waiting_queue:
        starved_or_dropped += len(waiting_queue)

    n_total = len(jobs)
    n_dispatched = len(wait_times_all)

    def _p95(vals: list[float]) -> float:
        return float(np.percentile(vals, 95)) if vals else 0.0

    p95_all_min = _p95(wait_times_all) / 60.0
    p95_gpu_min = _p95(wait_by_class["gpu_training"]) / 60.0
    p95_highmem_min = _p95(wait_by_class["high_mem_analytics"]) / 60.0
    p95_batch_min = _p95(wait_by_class["batch_array"]) / 60.0
    p95_interactive_sec = _p95(wait_by_class["interactive_ide"])

    weighted_p95_wait_min = (
        0.45 * p95_gpu_min
        + 0.25 * p95_highmem_min
        + 0.20 * p95_batch_min
        + 0.10 * (p95_interactive_sec / 60.0)
    )

    total_failures = oom_failures + timeout_failures + spot_failures
    oom_rate = oom_failures / max(1, n_total)
    spot_kill_rate = spot_failures / max(1, n_total)
    failure_rate = total_failures / max(1, n_total)
    avg_cost_per_job = (total_compute_cost_usd + total_storage_cost_usd) / max(1, n_dispatched)
    frag_waste_pct = (float(np.mean(fragmentation_samples)) * 100.0) if fragmentation_samples else 0.0

    # Monotonic maximization objective (negated composite operational penalty)
    raw_score = -(
        0.52 * weighted_p95_wait_min
        + 12.5 * avg_cost_per_job
        + 1.85 * frag_waste_pct
        + 2.80 * (failure_rate * 100.0)
    )

    return {
        "raw_overall_score": round(raw_score, 4),
        "total_jobs": n_total,
        "dispatched_jobs": n_dispatched,
        "starved_or_dropped_jobs": starved_or_dropped,
        "interactive_p95_wait_sec": round(p95_interactive_sec, 2),
        "gpu_p95_wait_min": round(p95_gpu_min, 2),
        "high_mem_p95_wait_min": round(p95_highmem_min, 2),
        "batch_p95_wait_min": round(p95_batch_min, 2),
        "overall_p95_wait_min": round(p95_all_min, 2),
        "weighted_p95_wait_min": round(weighted_p95_wait_min, 2),
        "avg_cost_per_job_usd": round(avg_cost_per_job, 4),
        "total_cost_usd": round(total_compute_cost_usd + total_storage_cost_usd, 2),
        "fragmentation_waste_pct": round(frag_waste_pct, 2),
        "bin_packing_efficiency_pct": round(max(0.0, 100.0 - frag_waste_pct), 2),
        "oom_failures": oom_failures,
        "oom_rate": round(oom_rate, 4),
        "oom_rate_pct": round(oom_rate * 100.0, 2),
        "timeout_failures": timeout_failures,
        "spot_failures": spot_failures,
        "spot_kill_rate": round(spot_kill_rate, 4),
        "spot_kill_rate_pct": round(spot_kill_rate * 100.0, 2),
        "failure_rate": round(failure_rate, 4),
        "failure_rate_pct": round(failure_rate * 100.0, 2),
        "squatter_placements": squatter_placements,
        "tier_job_counts": tier_job_counts,
        "storage_job_counts": storage_job_counts,
    }


def evaluate_program(
    candidate_code: str | None = None,
    split: str = "val",
    scenario_overrides: dict[str, Any] | None = None,
    *,
    require_markers: bool = True,
) -> dict[str, Any]:
    """3-stage AlphaEvolve evaluator for Pillar 1 Cluster Scheduling."""
    canonical_seed = SEED_FILE_PATH.read_text(encoding="utf-8")
    if candidate_code is None:
        cand_dir = os.environ.get("_CANDIDATE_DIR", "")
        cand_path = Path(cand_dir) / "main.py" if cand_dir else SEED_FILE_PATH
        candidate_code = cand_path.read_text(encoding="utf-8")

    # Stage 1: AST & EVOLVE-BLOCK sandbox gate
    try:
        spliced_source, _ = validate_candidate_source(
            candidate_code,
            canonical_seed_source=canonical_seed,
            required_functions=REQUIRED_FUNCTIONS,
            allowed_imports=ALLOWED_IMPORTS,
            require_markers=require_markers,
        )
        mod = execute_candidate_module(spliced_source, "pillar1_candidate")
        dispatch_fn = mod.__dict__["plan_job_dispatch"]
    except SandboxViolationError as exc:
        return make_evaluation_result(
            primary_metric="overall_score",
            overall_score=HARD_GATE_PENALTY,
            valid=False,
            gate_passed=False,
            stage_reached="stage_1_ast_safety",
            split=split,
            metrics={"error": str(exc)},
            insights=[("stage_1_ast_violation", str(exc))],
        )
    except Exception as exc:
        return make_evaluation_result(
            primary_metric="overall_score",
            overall_score=HARD_GATE_PENALTY,
            valid=False,
            gate_passed=False,
            stage_reached="stage_1_ast_safety",
            split=split,
            metrics={"error": str(exc)},
            insights=[("module_load_error", f"Failed to load candidate module: {exc}")],
        )

    overrides = scenario_overrides or {}
    trace = generate_hpc_trace(
        num_jobs=int(overrides.get("num_jobs", 10_000)),
        seed=int(overrides.get("seed", 42)),
        split=split,
        burst_multiplier=float(overrides.get("burst_multiplier", 1.0)),
        gpu_share_override=overrides.get("gpu_share_override"),
        spot_preemption_rate=float(overrides.get("spot_preemption_rate", 0.08)),
        dws_availability=float(overrides.get("dws_availability", 0.85)),
    )

    try:
        sim_metrics = run_with_timeout(
            lambda: run_policy_simulation(
                dispatch_fn,
                trace,
                dws_availability_override=overrides.get("dws_availability"),
                spot_preemption_override=overrides.get("spot_preemption_rate"),
            ),
            timeout_sec=20.0,
        )
    except CandidateTimeoutError as exc:
        return make_evaluation_result(
            primary_metric="overall_score",
            overall_score=HARD_GATE_PENALTY,
            valid=False,
            gate_passed=False,
            stage_reached="stage_2_hard_gates",
            split=split,
            metrics={"error": str(exc)},
            insights=[("execution_timeout", str(exc))],
        )
    except Exception as exc:
        return make_evaluation_result(
            primary_metric="overall_score",
            overall_score=HARD_GATE_PENALTY,
            valid=False,
            gate_passed=False,
            stage_reached="stage_2_hard_gates",
            split=split,
            metrics={"error": str(exc)},
            insights=[("runtime_exception", f"Candidate raised exception during simulation: {exc}")],
        )

    # Stage 2: Anti-reward-hacking hard SLA & feasibility floors
    violations: list[tuple[str, str]] = []
    if sim_metrics["starved_or_dropped_jobs"] > MAX_STARVED_OR_DROPPED_JOBS:
        violations.append(
            (
                "hard_gate_starvation",
                f"Violated zero-starvation floor: {sim_metrics['starved_or_dropped_jobs']} jobs were dropped, "
                "routed to incompatible shapes, or waited >24h. Ensure every job class has a valid pool and tier.",
            )
        )
    if sim_metrics["interactive_p95_wait_sec"] >= MAX_INTERACTIVE_P95_WAIT_SEC:
        violations.append(
            (
                "hard_gate_interactive_sla",
                f"Violated interactive startup SLA (<120s): interactive_p95_wait_sec={sim_metrics['interactive_p95_wait_sec']:.1f}s. "
                "Prioritize `interactive_ide` jobs onto warm `cpu_general` / `gen4_on_demand` slots without DWS delay.",
            )
        )
    if sim_metrics["oom_rate"] > MAX_OOM_RATE_FLOOR:
        violations.append(
            (
                "hard_gate_oom_floor",
                f"Violated hard OOM rate floor (<=22%): oom_rate={sim_metrics['oom_rate_pct']:.2f}% "
                f"({sim_metrics['oom_failures']} OOM kills). Right-size `alloc_mem_gb` using `hist_p95_mem_ratio`.",
            )
        )
    if sim_metrics["spot_kill_rate"] > MAX_SPOT_KILL_RATE_FLOOR:
        violations.append(
            (
                "hard_gate_spot_preemption_floor",
                f"Violated Spot preemption kill floor (<=4%): spot_kill_rate={sim_metrics['spot_kill_rate_pct']:.2f}% "
                f"({sim_metrics['spot_failures']} unrecoverable Spot kills). Only place `checkpointable=True` jobs on `spot`.",
            )
        )

    if violations:
        return make_evaluation_result(
            primary_metric="overall_score",
            overall_score=HARD_GATE_PENALTY,
            valid=False,
            gate_passed=False,
            stage_reached="stage_2_hard_gates",
            split=split,
            metrics=sim_metrics,
            insights=violations,
        )

    # Stage 3: Actionable bottleneck insights for evolutionary parent mutation
    insights: list[tuple[str, str]] = [
        (
            "overall_summary",
            f"overall_score={sim_metrics['raw_overall_score']:.3f} | "
            f"GPU P95 wait={sim_metrics['gpu_p95_wait_min']:.1f}m | "
            f"Interactive P95={sim_metrics['interactive_p95_wait_sec']:.1f}s | "
            f"Cost/job=${sim_metrics['avg_cost_per_job_usd']:.2f} | "
            f"Fragmentation={sim_metrics['fragmentation_waste_pct']:.1f}% | "
            f"Failure rate={sim_metrics['failure_rate_pct']:.2f}%",
        )
    ]
    if sim_metrics["squatter_placements"] > 0:
        insights.append(
            (
                "bin_packing_fragmentation",
                f"{sim_metrics['squatter_placements']} small <=8 vCPU non-GPU jobs landed on 64-core GPU/Ultramem nodes, "
                "stranding multi-GPU shapes. Exclude `gpu_pool` and `ultramem_pool` for small CPU workloads.",
            )
        )
    if sim_metrics["oom_failures"] + sim_metrics["timeout_failures"] > 20:
        insights.append(
            (
                "right_sizing_opportunity",
                f"Observed {sim_metrics['oom_failures']} OOM kills and {sim_metrics['timeout_failures']} walltime timeouts. "
                "Scale `alloc_mem_gb` using `hist_p95_mem_ratio * 1.06` (clamped to pool max RAM) and pad `alloc_walltime_sec` using `hist_p95_runtime_sec`.",
            )
        )
    if sim_metrics["tier_job_counts"]["dws_flex_start"] == 0:
        insights.append(
            (
                "dws_spillover_unused",
                "Zero jobs utilized `dws_flex_start`. Route `gpu_training` and `high_mem_analytics` burst spillover to "
                "`dws_flex_start` (32% cheaper than On-Demand, zero preemption) before `gen4_on_demand`.",
            )
        )
    if sim_metrics["storage_job_counts"]["shared_nfs_home"] > int(0.45 * sim_metrics["total_jobs"]):
        insights.append(
            (
                "nfs_metadata_contention",
                "Heavy reliance on `shared_nfs_home` causes metadata contention slowdowns. Route `high_iops_scratch` / "
                "`metadata_heavy` compute jobs to `nvme_parallel_scratch` and `streaming_throughput` batch jobs to `object_storage_tier`.",
            )
        )

    return make_evaluation_result(
        primary_metric="overall_score",
        overall_score=sim_metrics["raw_overall_score"],
        valid=True,
        gate_passed=True,
        stage_reached="stage_3_benchmark",
        split=split,
        metrics=sim_metrics,
        insights=insights,
        secondary_scores={
            "neg_gpu_p95_wait_min": -sim_metrics["gpu_p95_wait_min"],
            "neg_avg_cost_per_job_usd": -sim_metrics["avg_cost_per_job_usd"],
            "bin_packing_efficiency_pct": sim_metrics["bin_packing_efficiency_pct"],
            "neg_failure_rate_pct": -sim_metrics["failure_rate_pct"],
        },
    )
