"""Shared MAP-Elites + Multi-Island Evolutionary Engine for all 3 AlphaEvolve HPC Pillars.

Supports two execution modes:
  1. `live`: Real multi-generation evolutionary search powered by Vertex AI Gemini
     (`gemini-2.5-flash` in `us-central1`), using MAP-Elites behavioral niches,
     multi-island parent selection, and structured evaluator diagnostic `insights`.
  2. `replay`: Deterministic offline replay mode executing verified progressive
     mutations (including rejected reward-hacking candidates) through the real
     3-stage `evaluator.py` harnesses in seconds with 100% reproducibility for CI.

Strict Data-Hygiene & Split Isolation:
  - Evolutionary search evaluates candidates strictly on `val` (and `train` verification).
  - The `test` split is strictly held out and evaluated ONLY ONCE at completion on
    the unmodified seed program and the single winning candidate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import time
from typing import Any, Callable
import yaml

from pillars.cluster_scheduling import baselines as p1_baselines
from pillars.cluster_scheduling import evaluator as p1_eval
from pillars.cluster_scheduling.trace_generator import generate_hpc_trace
from pillars.common.models import EVOLVE_END_MARKER, EVOLVE_START_MARKER, HARD_GATE_PENALTY
from pillars.common.sandbox import compute_unified_diff, extract_evolve_block, replace_evolve_block
from pillars.common.vertex_client import VertexGeminiMutator
from pillars.demand_forecasting import evaluator as p3_eval
from pillars.hpc_code_optimization import evaluator as p2_eval

REPO_ROOT = Path(__file__).resolve().parents[2]
PILLARS_DIR = REPO_ROOT / "pillars"
RESULTS_DIR = PILLARS_DIR / "results"

PILLAR_NAMES: tuple[str, ...] = (
    "cluster_scheduling",
    "hpc_code_optimization",
    "demand_forecasting",
)

ISLAND_NAMES: tuple[str, ...] = (
    "island_0_efficiency",
    "island_1_sla_accuracy",
    "island_2_cost_tiering",
)

MUTATION_DIRECTIVES: dict[str, tuple[str, ...]] = {
    "cluster_scheduling": (
        "Enforce strict shape guardrails (`req_gpus == 0` must NEVER use `gpu_pool`; small `<=8` vCPU jobs must avoid `ultramem_pool`), keep `interactive_ide` at highest priority on warm CPU/On-Demand pools (<120s SLA), and route `gpu_training` burst to `dws_flex_start` before `gen4_on_demand`.",
        "Add historical P95 memory and runtime right-sizing (`alloc_mem_gb = min(pool_max_mem, req_mem_gb * max(1.0, hist_p95_mem_ratio * 1.06))`, `alloc_walltime_sec = max(req_walltime_sec, hist_p95_runtime_sec * 1.22)`) to eliminate OOM kills and walltime timeouts.",
        "Exploit `spot` strictly when `job['checkpointable']` is True on `batch_elastic`/`cpu_general`, route `high_iops_scratch` & `metadata_heavy` jobs to `nvme_parallel_scratch`, and route `streaming_throughput` batch arrays to `object_storage_tier`.",
        "Refine shortest-expected-GPU-work + wait-aging priority scoring (`wait_so_far_sec` boost + smaller `req_gpus * hist_p95_runtime_sec` fast-lane) so 1-GPU and 4-GPU jobs avoid head-of-line blocking behind 8-GPU jobs while keeping 0 starved jobs.",
        "Synthesize the strongest elements from parent and inspiration elites: balance `committed_reservation` -> `spot` (checkpointable batch) -> `dws_flex_start` (GPU/Ultramem bursts) -> `gen4_on_demand` while minimizing fragmentation, cost/job, OOMs, and P95 wait.",
    ),
    "hpc_code_optimization": (
        "Replace the scalar Python nested loops with fully vectorized float64 NumPy pair computations using `erfc_vec(alpha * s)` and exact minimum-image periodic wrap `disp -= box_length * np.round(disp * inv_box)` while preserving rel L2 error < 1e-6.",
        "Exploit Newton's 3rd Law skew-symmetry: evaluate only upper-triangular pairs `i < j` (`np.triu_indices(n_particles, k=1)`), filter immediately to active pairs with `r2 < rcut2`, compute forces only on interacting pairs, and scatter `+f_vec` to `i` and `-f_vec` to `j` using `np.bincount` per coordinate.",
        "Eliminate `np.triu_indices` memory allocation overhead and optimize vectorized pair pruning: broadcast 1D coordinate differences `(x[:, None] - x[None, :])` one axis at a time with early cutoff masking `(abs(dx) < cutoff)` before computing `dy`, `dz`, or use 3D spatial cell-list binning when `box_length >= 3.0 * cutoff`.",
        "Optimize NumPy cache locality and minimize temporary array allocations: compute `dx` on upper triangle or spatial neighbor pairs, prune non-interacting pairs before `sqrt`/`exp`/`erfc_vec`, use `np.bincount(idx_i, weights=fx, minlength=n_particles)` for ultra-fast C-level force reduction.",
        "Combine spatial cell-list / axis-pruned candidate pair generation with Newton's 3rd Law symmetry and `np.bincount` force scatter to maximize `particles_per_sec` across N=320..640 while maintaining double-precision L2 error < 1e-12.",
    ),
    "demand_forecasting": (
        "Decompose causal `history_window` by `day_of_week` (0..6 weekday vs weekend multipliers), recent secular growth trend slope (comparing last 21 days vs prior 21 days), and `quarter_day` (0..90, capturing the surge around quarter_day 70..86) for accurate 7d tactical and 90d quarterly P50/P99 forecasts.",
        "Optimize commercial compute tier stacking under the 5x asymmetric stockout penalty: anchor `cud_1y_3y_units` near ~0.85 * sustained P50 baseline (avoiding >1.05x 90d P50 lock-in excess), allocate `spot_units` up to `0.95 * max_safe_spot_share * peak_p99`, `dws_flex_units` up to `0.95 * dws_capacity_cap_ratio * peak_p99`, and right-size `on_demand_units` to cover ~1.03x tactical peak P99.",
        "Optimize `tiered_storage_tb` lifecycle allocation: forecast peak 7-day `hot_working_set_tb` with growth & weekday surge, set `nvme_parallel_fs_tb` to `1.05 * peak_hot_working_set_tb` (satisfying >=0.85 coverage floor without overpaying $8/TB-day on cold data!), and allocate remaining total TB (`1.02 * peak_st_p99 - nvme_tb`) to $0.67/TB-day `object_storage_tb`.",
        "Calibrate separate pool-specific trend rates (`gpu_accelerators` grows much faster over 90 days than `cpu_general`!), empirically estimate `p99 / p50` burst ratios per pool & quarter-surge window from `history_window`, and tune Newsvendor P99 safety factor as a smooth function of `asymmetric_penalty_ratio`.",
        "Fuse the best causal forecaster (weekday seasonality + pool-specific linear trend + Gaussian `quarter_day` 78 surge kernel) with precision CUD + Spot + DWS Flex + On-Demand + NVMe/Object Storage tier sizing to simultaneously minimize WMAPE and total risk-adjusted TCO.",
    ),
}


# Deterministic offline replay candidates (used ONLY when `mode="replay"` for fast CI & unit tests)
OFFLINE_REPLAY_BLOCKS: dict[str, tuple[str, ...]] = {
    "cluster_scheduling": (
        # Replay 1: Overly aggressive Spot saboteur (demonstrates Stage 2 anti-reward-hacking rejection!)
        """def plan_job_dispatch(
    job: dict[str, Any],
    pool_utilization: dict[str, float],
    queue_backlog: dict[str, int],
) -> dict[str, Any]:
    pools = ["gpu_pool"] if job["req_gpus"] > 0 else ["cpu_general", "batch_elastic"]
    return {
        "priority_score": 100.0 if job["workload_class"] == "interactive_ide" else 1.0,
        "preferred_pools": pools,
        "tier_preference": ["spot", "committed_reservation"],
        "storage_tier": "shared_nfs_home",
        "alloc_mem_gb": float(job["req_mem_gb"]) * 0.75,
        "alloc_walltime_sec": float(job["req_walltime_sec"]),
    }""",
        # Replay 2: Shape guardrails + interactive fast-lane
        """def plan_job_dispatch(
    job: dict[str, Any],
    pool_utilization: dict[str, float],
    queue_backlog: dict[str, int],
) -> dict[str, Any]:
    wclass = job["workload_class"]
    wait_min = float(job.get("wait_so_far_sec", 0.0)) / 60.0
    if wclass == "interactive_ide":
        return {
            "priority_score": 1000.0 + wait_min * 5.0,
            "preferred_pools": ["cpu_general", "batch_elastic"],
            "tier_preference": ["committed_reservation", "gen4_on_demand"],
            "storage_tier": "shared_nfs_home",
            "alloc_mem_gb": min(64.0, float(job["req_mem_gb"]) * max(1.0, float(job["hist_p95_mem_ratio"]) * 1.05)),
            "alloc_walltime_sec": max(float(job["req_walltime_sec"]), float(job["hist_p95_runtime_sec"]) * 1.20),
        }
    if job["req_gpus"] > 0:
        return {
            "priority_score": 120.0 + wait_min * 1.5 - float(job["req_gpus"]) * 4.0,
            "preferred_pools": ["gpu_pool"],
            "tier_preference": ["committed_reservation", "dws_flex_start", "gen4_on_demand"],
            "storage_tier": "nvme_parallel_scratch",
            "alloc_mem_gb": min(512.0, float(job["req_mem_gb"]) * max(1.0, float(job["hist_p95_mem_ratio"]) * 1.06)),
            "alloc_walltime_sec": max(float(job["req_walltime_sec"]), float(job["hist_p95_runtime_sec"]) * 1.22),
        }
    if wclass == "high_mem_analytics" or float(job["req_mem_gb"]) > 120.0:
        return {
            "priority_score": 90.0 + wait_min * 1.2,
            "preferred_pools": ["ultramem_pool"],
            "tier_preference": ["committed_reservation", "dws_flex_start", "gen4_on_demand"],
            "storage_tier": "nvme_parallel_scratch",
            "alloc_mem_gb": min(1024.0, float(job["req_mem_gb"]) * max(1.0, float(job["hist_p95_mem_ratio"]) * 1.06)),
            "alloc_walltime_sec": max(float(job["req_walltime_sec"]), float(job["hist_p95_runtime_sec"]) * 1.22),
        }
    tiers = ["committed_reservation", "spot", "dws_flex_start", "gen4_on_demand"] if job["checkpointable"] else ["committed_reservation", "dws_flex_start", "gen4_on_demand"]
    storage = "object_storage_tier" if job["io_profile"] == "streaming_throughput" else "nvme_parallel_scratch"
    return {
        "priority_score": 40.0 + wait_min * 0.8,
        "preferred_pools": ["batch_elastic", "cpu_general"],
        "tier_preference": tiers,
        "storage_tier": storage,
        "alloc_mem_gb": min(64.0, float(job["req_mem_gb"]) * max(1.0, float(job["hist_p95_mem_ratio"]) * 1.05)),
        "alloc_walltime_sec": max(float(job["req_walltime_sec"]), float(job["hist_p95_runtime_sec"]) * 1.22),
    }""",
    ),
    "hpc_code_optimization": (
        # Replay 1: Broken cutoff approximation (demonstrates Stage 2 physics accuracy gate returning -1e12!)
        """def solve_ewald_forces(
    positions: np.ndarray,
    charges: np.ndarray,
    box_length: float,
    cutoff: float,
    alpha: float,
    softening: float = 0.08,
) -> tuple[np.ndarray, float]:
    return np.zeros_like(positions, dtype=np.float64), 0.0""",
        # Replay 2: Upper-triangle Newton's 3rd Law + sparse cutoff mask + bincount scatter
        """def solve_ewald_forces(
    positions: np.ndarray,
    charges: np.ndarray,
    box_length: float,
    cutoff: float,
    alpha: float,
    softening: float = 0.08,
) -> tuple[np.ndarray, float]:
    pos = np.asarray(positions, dtype=np.float64)
    q = np.asarray(charges, dtype=np.float64)
    n = pos.shape[0]
    inv_box = 1.0 / box_length
    rcut2 = cutoff * cutoff
    soft2 = softening * softening
    s_cut = math.sqrt(rcut2 + soft2)
    shift_val = math.erfc(alpha * s_cut) / s_cut
    two_alpha_sqrt_pi = 2.0 * alpha / math.sqrt(math.pi)

    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = pos[idx_j, 0] - pos[idx_i, 0]
    dx -= box_length * np.round(dx * inv_box)
    m_x = np.abs(dx) < cutoff
    idx_i = idx_i[m_x]
    idx_j = idx_j[m_x]
    dx = dx[m_x]

    dy = pos[idx_j, 1] - pos[idx_i, 1]
    dy -= box_length * np.round(dy * inv_box)
    dz = pos[idx_j, 2] - pos[idx_i, 2]
    dz -= box_length * np.round(dz * inv_box)

    r2 = dx * dx + dy * dy + dz * dz
    mask = r2 < rcut2
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]
    dx = dx[mask]
    dy = dy[mask]
    dz = dz[mask]
    r2 = r2[mask]

    s = np.sqrt(r2 + soft2)
    as_arg = alpha * s
    erfc_s = erfc_vec(as_arg)
    qij = q[idx_i] * q[idx_j]

    total_energy = float(np.sum(qij * (erfc_s / s - shift_val)))
    pref = -qij * (erfc_s + two_alpha_sqrt_pi * s * np.exp(-as_arg * as_arg)) / (s * s * s)
    fx = pref * dx
    fy = pref * dy
    fz = pref * dz

    forces = np.empty((n, 3), dtype=np.float64)
    forces[:, 0] = np.bincount(idx_i, weights=fx, minlength=n) - np.bincount(idx_j, weights=fx, minlength=n)
    forces[:, 1] = np.bincount(idx_i, weights=fy, minlength=n) - np.bincount(idx_j, weights=fy, minlength=n)
    forces[:, 2] = np.bincount(idx_i, weights=fz, minlength=n) - np.bincount(idx_j, weights=fz, minlength=n)
    return forces, total_energy""",
    ),
    "demand_forecasting": (
        # Replay 1: Under-provisioned hot NVMe storage saboteur (demonstrates Stage 2 storage gate -1e12!)
        """def forecast_and_allocate(
    history_window: list[dict[str, Any]],
    pool_specs: dict[str, dict[str, Any]],
    asymmetric_penalty_ratio: float = 5.0,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for pool in COMPUTE_POOLS:
        out[pool] = {
            "tactical_7d_p50": [10.0] * 7,
            "tactical_7d_p99": [12.0] * 7,
            "quarterly_90d_p50": 10.0,
            "quarterly_90d_p99": 12.0,
            "allocation": {"cud_1y_3y_units": 5.0, "dws_flex_units": 0.0, "on_demand_units": 5.0, "spot_units": 0.0},
        }
    out["tiered_storage_tb"] = {
        "tactical_7d_p50": [500.0] * 7,
        "tactical_7d_p99": [520.0] * 7,
        "quarterly_90d_p50": 500.0,
        "quarterly_90d_p99": 520.0,
        "allocation": {"nvme_parallel_fs_tb": 10.0, "object_storage_tb": 510.0},
    }
    return out""",
        # Replay 2: Causal seasonal + trend + quarter-surge forecaster + CUD/DWS/Spot/NVMe optimizer
        """def forecast_and_allocate(
    history_window: list[dict[str, Any]],
    pool_specs: dict[str, dict[str, Any]],
    asymmetric_penalty_ratio: float = 5.0,
) -> dict[str, dict[str, Any]]:
    n_hist = len(history_window)
    last_rec = history_window[-1]
    last_day = int(last_rec["day_index"])
    out: dict[str, dict[str, Any]] = {}

    safety_pad = 1.0 + 0.012 * math.log(max(1.0, asymmetric_penalty_ratio))

    for pool in COMPUTE_POOLS:
        p50_arr = np.array([float(r[pool]["p50"]) for r in history_window], dtype=np.float64)
        p99_arr = np.array([float(r[pool]["p99"]) for r in history_window], dtype=np.float64)
        dows = np.array([int(r["day_of_week"]) for r in history_window], dtype=int)
        qdays = np.array([int(r["quarter_day"]) for r in history_window], dtype=np.float64)

        q_mult_hist = 1.0 + 0.26 * np.exp(-0.5 * ((qdays - 78.0) / 7.5) ** 2)
        deseas = p50_arr / q_mult_hist

        dow_factors = np.ones(7, dtype=np.float64)
        recent_slice = min(n_hist, 56)
        base_med = float(np.mean(deseas[-recent_slice:]))
        for d in range(7):
            mask = dows[-recent_slice:] == d
            if np.any(mask):
                dow_factors[d] = float(np.mean(deseas[-recent_slice:][mask])) / max(base_med, 1e-6)
        dow_factors /= np.mean(dow_factors)

        norm_series = deseas / dow_factors[dows]
        w_len = min(n_hist, 84)
        xs = np.arange(w_len, dtype=np.float64)
        slope, intercept = np.polyfit(xs, norm_series[-w_len:], 1)
        slope = max(0.0, float(slope))
        cur_level = float(intercept + slope * (w_len - 1))

        burst_ratios = p99_arr[-recent_slice:] / np.maximum(p50_arr[-recent_slice:], 1e-6)
        p99_ratio_90 = float(np.percentile(burst_ratios, 82))

        tac_p50: list[float] = []
        tac_p99: list[float] = []
        for k in range(1, 8):
            fut_dow = (last_day + k) % 7
            fut_q = (last_day + k) % 91
            q_boost = 1.0 + 0.26 * math.exp(-0.5 * ((fut_q - 78.0) / 7.5) ** 2)
            v50 = (cur_level + slope * k) * float(dow_factors[fut_dow]) * q_boost
            v99 = v50 * p99_ratio_90 * (1.0 + 0.04 * (q_boost - 1.0) / 0.26)
            tac_p50.append(max(1.0, v50))
            tac_p99.append(max(v50, v99))

        q90_p50_vals: list[float] = []
        q90_p99_vals: list[float] = []
        for k in range(1, 91):
            fut_dow = (last_day + k) % 7
            fut_q = (last_day + k) % 91
            q_boost = 1.0 + 0.26 * math.exp(-0.5 * ((fut_q - 78.0) / 7.5) ** 2)
            v50 = (cur_level + slope * k) * float(dow_factors[fut_dow]) * q_boost
            v99 = v50 * p99_ratio_90
            q90_p50_vals.append(v50)
            q90_p99_vals.append(v99)

        q90_p50 = float(np.mean(q90_p50_vals))
        q90_p99 = float(np.percentile(q90_p99_vals, 95))

        pspec = pool_specs[pool]
        peak_7d_p99 = max(tac_p99) * safety_pad
        min_7d_p99 = min(tac_p99)

        cud_u = min(q90_p50 * 0.96, float(np.mean(tac_p50)) * 0.88)
        spot_u = float(pspec["max_safe_spot_share"]) * min_7d_p99 * 0.95
        dws_u = float(pspec["dws_capacity_cap_ratio"]) * min_7d_p99 * 0.92
        od_u = max(0.0, peak_7d_p99 - (cud_u + spot_u + dws_u))

        out[pool] = {
            "tactical_7d_p50": tac_p50,
            "tactical_7d_p99": tac_p99,
            "quarterly_90d_p50": q90_p50,
            "quarterly_90d_p99": max(q90_p50, q90_p99),
            "allocation": {
                "cud_1y_3y_units": cud_u,
                "dws_flex_units": dws_u,
                "on_demand_units": od_u,
                "spot_units": spot_u,
            },
        }

    st_p50 = np.array([float(r["tiered_storage_tb"]["p50"]) for r in history_window], dtype=np.float64)
    st_hot = np.array([float(r["tiered_storage_tb"]["hot_working_set_tb"]) for r in history_window], dtype=np.float64)
    w_st = min(n_hist, 56)
    xs_st = np.arange(w_st, dtype=np.float64)
    st_slope, st_int = np.polyfit(xs_st, st_p50[-w_st:], 1)
    st_cur = float(st_int + st_slope * (w_st - 1))
    hot_ratio_p90 = float(np.percentile(st_hot[-w_st:] / np.maximum(st_p50[-w_st:], 1.0), 92))

    st_tac_50 = [max(1.0, st_cur + float(st_slope) * k) for k in range(1, 8)]
    st_tac_99 = [v * 1.032 for v in st_tac_50]
    st_q90_50 = float(np.mean([st_cur + float(st_slope) * k for k in range(1, 91)]))
    st_q90_99 = (st_cur + float(st_slope) * 86.0) * 1.035

    peak_st_99 = max(st_tac_99) * 1.01
    nvme_tb = max(st_tac_50) * hot_ratio_p90 * 1.06
    obj_tb = max(0.0, peak_st_99 - nvme_tb)

    out["tiered_storage_tb"] = {
        "tactical_7d_p50": st_tac_50,
        "tactical_7d_p99": st_tac_99,
        "quarterly_90d_p50": st_q90_50,
        "quarterly_90d_p99": max(st_q90_50, st_q90_99),
        "allocation": {
            "nvme_parallel_fs_tb": nvme_tb,
            "object_storage_tb": obj_tb,
        },
    }
    return out""",
    ),
}


@dataclass
class ArchiveEntry:
    """Represents a valid evaluated program stored in an island + MAP-Elites cell."""

    generation: int
    island: str
    niche: str
    block_code: str
    full_source: str
    val_score: float
    metrics: dict[str, Any]
    insights: list[dict[str, str]]


class MapElitesIslandArchive:
    """Multi-island + MAP-Elites behavioral diversity archive for AlphaEvolve search."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.islands: dict[str, list[ArchiveEntry]] = {isl: [] for isl in ISLAND_NAMES}
        self.niche_grid: dict[str, ArchiveEntry] = {}
        self.best_entry: ArchiveEntry | None = None

    def register(self, entry: ArchiveEntry) -> bool:
        self.islands[entry.island].append(entry)
        self.islands[entry.island].sort(key=lambda e: e.val_score, reverse=True)
        self.islands[entry.island] = self.islands[entry.island][:8]

        improved_global = False
        if self.best_entry is None or entry.val_score > self.best_entry.val_score:
            self.best_entry = entry
            improved_global = True

        cur_niche = self.niche_grid.get(entry.niche)
        if cur_niche is None or entry.val_score > cur_niche.val_score:
            self.niche_grid[entry.niche] = entry

        return improved_global

    def select_parent_and_inspirations(
        self, target_island: str
    ) -> tuple[ArchiveEntry, list[tuple[float, str, str]]]:
        assert self.best_entry is not None
        pool = self.islands[target_island] or [self.best_entry]
        # Tournament selection among top island candidates
        k = min(len(pool), 3)
        contenders = self.rng.sample(pool, k=k)
        parent = max(contenders, key=lambda c: c.val_score)

        inspirations: list[tuple[float, str, str]] = []
        for niche_key, elite in sorted(
            self.niche_grid.items(), key=lambda kv: kv[1].val_score, reverse=True
        ):
            if elite.block_code != parent.block_code:
                inspirations.append((elite.val_score, niche_key, elite.block_code))
            if len(inspirations) >= 2:
                break
        return parent, inspirations


def _compute_niche(pillar: str, metrics: dict[str, Any]) -> str:
    if pillar == "cluster_scheduling":
        gpu_w = float(metrics.get("gpu_p95_wait_min", 500.0))
        cost = float(metrics.get("avg_cost_per_job_usd", 10.0))
        w_bin = "fast_gpu" if gpu_w < 80 else ("med_gpu" if gpu_w < 220 else "slow_gpu")
        c_bin = "low_cost" if cost < 3.0 else ("mid_cost" if cost < 4.2 else "high_cost")
        return f"{w_bin}_{c_bin}"
    if pillar == "hpc_code_optimization":
        sp = float(metrics.get("speedup_vs_baseline", 1.0))
        if sp >= 80.0:
            return "ultra_vectorized_80x_plus"
        if sp >= 25.0:
            return "sparse_tiled_25x_80x"
        if sp >= 5.0:
            return "dense_vectorized_5x_25x"
        return "scalar_or_hybrid_lt5x"
    # demand_forecasting
    wmape = float(metrics.get("blended_wmape_pct", 20.0))
    st_rate = float(metrics.get("stockout_breach_rate_pct", 50.0))
    w_bin = "high_acc" if wmape < 6.5 else ("med_acc" if wmape < 11.0 else "coarse_acc")
    s_bin = "high_sla" if st_rate < 10.0 else ("med_sla" if st_rate < 30.0 else "low_sla")
    return f"{w_bin}_{s_bin}"


def get_pillar_evaluator(pillar: str) -> Callable[..., dict[str, Any]]:
    if pillar == "cluster_scheduling":
        return p1_eval.evaluate_program
    if pillar == "hpc_code_optimization":
        return p2_eval.evaluate_program
    if pillar == "demand_forecasting":
        return p3_eval.evaluate_program
    raise ValueError(f"Unknown pillar '{pillar}'. Expected one of {PILLAR_NAMES}.")


def load_pillar_config(pillar: str) -> tuple[dict[str, Any], str]:
    pdir = PILLARS_DIR / pillar
    spec = yaml.safe_load((pdir / "experiment.yaml").read_text(encoding="utf-8"))
    seed_src = (pdir / "main.py").read_text(encoding="utf-8")
    return spec, seed_src


def compute_improvement_pct(baseline_score: float, evolved_score: float, pillar: str) -> float:
    """Compute interpretable percentage improvement on primary metric."""
    if pillar == "hpc_code_optimization":
        # Positive throughput metric (`particles_per_sec`): percentage throughput gain
        return round(((evolved_score - baseline_score) / max(abs(baseline_score), 1e-9)) * 100.0, 2)
    # Negated operational penalty / risk-adjusted cost (`overall_score < 0`): percentage penalty reduction
    return round(((evolved_score - baseline_score) / max(abs(baseline_score), 1e-9)) * 100.0, 2)


def _build_kpi_summary(
    pillar: str,
    base_m: dict[str, Any],
    evol_m: dict[str, Any],
) -> list[dict[str, Any]]:
    if pillar == "cluster_scheduling":
        return [
            {
                "kpi": "GPU Training P95 Queue Wait",
                "unit": "minutes",
                "baseline": base_m["gpu_p95_wait_min"],
                "evolved": evol_m["gpu_p95_wait_min"],
                "delta_pct": round(
                    ((evol_m["gpu_p95_wait_min"] - base_m["gpu_p95_wait_min"]) / max(base_m["gpu_p95_wait_min"], 1e-6)) * 100.0,
                    1,
                ),
            },
            {
                "kpi": "Interactive IDE P95 Startup",
                "unit": "seconds",
                "baseline": base_m["interactive_p95_wait_sec"],
                "evolved": evol_m["interactive_p95_wait_sec"],
                "sla_ceiling": 120.0,
            },
            {
                "kpi": "Average Compute + Storage Cost / Job",
                "unit": "USD",
                "baseline": base_m["avg_cost_per_job_usd"],
                "evolved": evol_m["avg_cost_per_job_usd"],
                "delta_pct": round(
                    ((evol_m["avg_cost_per_job_usd"] - base_m["avg_cost_per_job_usd"]) / max(base_m["avg_cost_per_job_usd"], 1e-6)) * 100.0,
                    1,
                ),
            },
            {
                "kpi": "OOM + Timeout + Preemption Failure Rate",
                "unit": "%",
                "baseline": base_m["failure_rate_pct"],
                "evolved": evol_m["failure_rate_pct"],
                "delta_pct": round(
                    ((evol_m["failure_rate_pct"] - base_m["failure_rate_pct"]) / max(base_m["failure_rate_pct"], 1e-6)) * 100.0,
                    1,
                ),
            },
            {
                "kpi": "Bin-Packing Efficiency",
                "unit": "%",
                "baseline": base_m["bin_packing_efficiency_pct"],
                "evolved": evol_m["bin_packing_efficiency_pct"],
                "delta_pts": round(
                    evol_m["bin_packing_efficiency_pct"] - base_m["bin_packing_efficiency_pct"],
                    2,
                ),
            },
        ]

    if pillar == "hpc_code_optimization":
        return [
            {
                "kpi": "Molecular Force-Field Throughput",
                "unit": "particles/sec",
                "baseline": base_m["particles_per_sec"],
                "evolved": evol_m["particles_per_sec"],
                "speedup_factor": evol_m["speedup_vs_baseline"],
            },
            {
                "kpi": "Median Conformation Batch Latency",
                "unit": "ms",
                "baseline": base_m["median_batch_latency_ms"],
                "evolved": evol_m["median_batch_latency_ms"],
                "delta_pct": round(
                    ((evol_m["median_batch_latency_ms"] - base_m["median_batch_latency_ms"]) / max(base_m["median_batch_latency_ms"], 1e-6)) * 100.0,
                    1,
                ),
            },
            {
                "kpi": "Relative L2 Force Numerical Error",
                "unit": "dimensionless",
                "baseline": base_m["rel_l2_force_error"],
                "evolved": evol_m["rel_l2_force_error"],
                "hard_gate_ceiling": 1.0e-6,
            },
            {
                "kpi": "Symplectic Velocity-Verlet Energy Drift",
                "unit": "dimensionless",
                "baseline": base_m["symplectic_energy_drift"],
                "evolved": evol_m["symplectic_energy_drift"],
                "hard_gate_ceiling": 1.0e-4,
            },
        ]

    return [
        {
            "kpi": "Total Risk-Adjusted 12-Week TCO",
            "unit": "USD ($K)",
            "baseline": base_m["total_risk_adjusted_tco_k_usd"],
            "evolved": evol_m["total_risk_adjusted_tco_k_usd"],
            "delta_pct": round(
                ((evol_m["total_risk_adjusted_tco_k_usd"] - base_m["total_risk_adjusted_tco_k_usd"]) / max(base_m["total_risk_adjusted_tco_k_usd"], 1e-6)) * 100.0,
                1,
            ),
        },
        {
            "kpi": "Asymmetric 5x Capacity Stockout Penalty",
            "unit": "USD ($K)",
            "baseline": base_m["asymmetric_stockout_penalty_k_usd"],
            "evolved": evol_m["asymmetric_stockout_penalty_k_usd"],
            "delta_pct": round(
                ((evol_m["asymmetric_stockout_penalty_k_usd"] - base_m["asymmetric_stockout_penalty_k_usd"]) / max(base_m["asymmetric_stockout_penalty_k_usd"], 1e-6)) * 100.0,
                1,
            ),
        },
        {
            "kpi": "Blended Multi-Horizon WMAPE (7d + 90d)",
            "unit": "%",
            "baseline": base_m["blended_wmape_pct"],
            "evolved": evol_m["blended_wmape_pct"],
            "delta_pct": round(
                ((evol_m["blended_wmape_pct"] - base_m["blended_wmape_pct"]) / max(base_m["blended_wmape_pct"], 1e-6)) * 100.0,
                1,
            ),
        },
        {
            "kpi": "Daily P99 Stockout Breach Rate",
            "unit": "%",
            "baseline": base_m["stockout_breach_rate_pct"],
            "evolved": evol_m["stockout_breach_rate_pct"],
            "delta_pts": round(
                evol_m["stockout_breach_rate_pct"] - base_m["stockout_breach_rate_pct"],
                2,
            ),
        },
    ]


def propose_and_evaluate_step(
    pillar: str,
    parent_code: str | None = None,
    *,
    mode: str = "live",
    seed: int = 42,
    split: str = "val",
) -> dict[str, Any]:
    """Run a single fast (<3s) candidate mutation + sandboxed evaluation step for live UI demos."""
    spec, seed_src = load_pillar_config(pillar)
    eval_fn = get_pillar_evaluator(pillar)

    base_source = parent_code if parent_code else seed_src
    parent_block = extract_evolve_block(base_source, require_markers=True)
    parent_eval = eval_fn(candidate_code=base_source, split=split)

    t0 = time.perf_counter()
    if mode == "live":
        mutator = VertexGeminiMutator(timeout_sec=25.0)
        gen_resp = mutator.generate_mutated_block(
            problem_description=str(spec["problem_description"]),
            surrounding_context=seed_src,
            parent_block=parent_block,
            parent_score=float(parent_eval["overall_score"]),
            parent_insights=parent_eval["insights"]["insights"],
            mutation_directive=MUTATION_DIRECTIVES[pillar][0],
            temperature=0.45,
            thinking_budget=0,  # Ultra-low-latency (<2s) interactive mode
        )
        mutated_block = gen_resp["mutated_block"]
        model_used = gen_resp["model"]
    else:
        mutated_block = OFFLINE_REPLAY_BLOCKS[pillar][-1]
        model_used = "deterministic-offline-replay"

    try:
        cand_full = replace_evolve_block(seed_src, mutated_block)
    except Exception:
        cand_full = f"{EVOLVE_START_MARKER}\n{mutated_block}\n{EVOLVE_END_MARKER}\n"

    cand_eval = eval_fn(candidate_code=cand_full, split=split)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "pillar": pillar,
        "mode": mode,
        "model": model_used,
        "elapsed_ms": round(elapsed_ms, 1),
        "parent_score": parent_eval["overall_score"],
        "candidate_score": cand_eval["overall_score"],
        "valid": cand_eval["valid"],
        "stage_reached": cand_eval["stage_reached"],
        "mutated_evolve_block": mutated_block,
        "code_diff": compute_unified_diff(parent_block, mutated_block),
        "metrics": cand_eval["metrics"],
        "insights": cand_eval["insights"]["insights"],
    }


def run_pillar_evolution(
    pillar: str,
    *,
    num_candidates: int = 14,
    mode: str = "live",
    seed: int = 42,
) -> dict[str, Any]:
    """Execute full multi-generation evolutionary search for a pillar and evaluate on held-out test."""
    spec, seed_src = load_pillar_config(pillar)
    eval_fn = get_pillar_evaluator(pillar)
    seed_block = extract_evolve_block(seed_src, require_markers=True)

    # Evaluate seed baseline on train and val (NEVER test during evolution!)
    seed_train_eval = eval_fn(candidate_code=seed_src, split="train")
    seed_val_eval = eval_fn(candidate_code=seed_src, split="val")
    if not seed_val_eval["valid"]:
        raise RuntimeError(f"Seed program for '{pillar}' failed validation gate: {seed_val_eval}")

    archive = MapElitesIslandArchive(seed=seed)
    seed_entry = ArchiveEntry(
        generation=0,
        island=ISLAND_NAMES[0],
        niche=_compute_niche(pillar, seed_val_eval["metrics"]),
        block_code=seed_block,
        full_source=seed_src,
        val_score=float(seed_val_eval["overall_score"]),
        metrics=seed_val_eval["metrics"],
        insights=seed_val_eval["insights"]["insights"],
    )
    archive.register(seed_entry)

    generation_history: list[dict[str, Any]] = [
        {
            "generation": 0,
            "island": "seed_baseline",
            "niche": seed_entry.niche,
            "candidate_val_score": seed_entry.val_score,
            "best_val_score_so_far": seed_entry.val_score,
            "valid": True,
            "stage_reached": seed_val_eval["stage_reached"],
            "latency_ms": 0.0,
            "model": "seed_program",
            "insight_summary": seed_entry.insights[0]["text"] if seed_entry.insights else "Seed baseline",
        }
    ]

    mutator = VertexGeminiMutator(timeout_sec=50.0) if mode == "live" else None
    directives = MUTATION_DIRECTIVES[pillar]

    if mode == "replay":
        replay_seq = OFFLINE_REPLAY_BLOCKS[pillar]
        total_steps = len(replay_seq)
    else:
        total_steps = num_candidates

    valid_count = 0
    rejected_count = 0

    for step_idx in range(1, total_steps + 1):
        island = ISLAND_NAMES[(step_idx - 1) % len(ISLAND_NAMES)]
        parent, inspirations = archive.select_parent_and_inspirations(island)
        directive = directives[(step_idx - 1) % len(directives)]
        temp = 0.35 + 0.15 * ((step_idx - 1) % 4)

        t_step = time.perf_counter()
        model_label = "deterministic-offline-replay"
        if mode == "live":
            assert mutator is not None
            try:
                gen_out = mutator.generate_mutated_block(
                    problem_description=str(spec["problem_description"]),
                    surrounding_context=seed_src,
                    parent_block=parent.block_code,
                    parent_score=parent.val_score,
                    parent_insights=parent.insights,
                    inspiration_elites=inspirations,
                    mutation_directive=directive,
                    temperature=temp,
                    thinking_budget=1024,
                )
                cand_block = gen_out["mutated_block"]
                model_label = gen_out["model"]
            except Exception as exc:
                rejected_count += 1
                generation_history.append(
                    {
                        "generation": step_idx,
                        "island": island,
                        "niche": "generation_error",
                        "candidate_val_score": HARD_GATE_PENALTY,
                        "best_val_score_so_far": archive.best_entry.val_score if archive.best_entry else seed_entry.val_score,
                        "valid": False,
                        "stage_reached": "llm_generation_error",
                        "latency_ms": round((time.perf_counter() - t_step) * 1000.0, 1),
                        "model": "gemini-2.5-flash",
                        "insight_summary": str(exc)[:300],
                    }
                )
                continue
        else:
            cand_block = OFFLINE_REPLAY_BLOCKS[pillar][step_idx - 1]

        try:
            cand_full = replace_evolve_block(seed_src, cand_block)
        except Exception:
            cand_full = f"{EVOLVE_START_MARKER}\n{cand_block}\n{EVOLVE_END_MARKER}\n"

        val_res = eval_fn(candidate_code=cand_full, split="val")
        step_ms = (time.perf_counter() - t_step) * 1000.0

        if val_res["valid"] and float(val_res["overall_score"]) > HARD_GATE_PENALTY * 0.1:
            # Verify candidate also passes hard constraint floors on `train` split to prevent val overfitting
            train_check = eval_fn(candidate_code=cand_full, split="train")
            if train_check["valid"]:
                valid_count += 1
                niche = _compute_niche(pillar, val_res["metrics"])
                entry = ArchiveEntry(
                    generation=step_idx,
                    island=island,
                    niche=niche,
                    block_code=extract_evolve_block(cand_full, require_markers=True),
                    full_source=cand_full,
                    val_score=float(val_res["overall_score"]),
                    metrics=val_res["metrics"],
                    insights=val_res["insights"]["insights"],
                )
                archive.register(entry)
                cand_score = entry.val_score
                is_valid = True
                stage_str = val_res["stage_reached"]
                ins_text = entry.insights[0]["text"] if entry.insights else "Valid candidate"
            else:
                rejected_count += 1
                niche = "rejected_train_gate"
                cand_score = HARD_GATE_PENALTY
                is_valid = False
                stage_str = train_check["stage_reached"]
                ins_lst = train_check["insights"]["insights"]
                ins_text = ins_lst[0]["text"] if ins_lst else "Failed train split gate"
        else:
            rejected_count += 1
            niche = "rejected_gate"
            cand_score = HARD_GATE_PENALTY
            is_valid = False
            stage_str = val_res["stage_reached"]
            ins_lst = val_res["insights"]["insights"]
            ins_text = ins_lst[0]["text"] if ins_lst else "Failed evaluator gate"

        assert archive.best_entry is not None
        generation_history.append(
            {
                "generation": step_idx,
                "island": island,
                "niche": niche,
                "candidate_val_score": cand_score,
                "best_val_score_so_far": archive.best_entry.val_score,
                "valid": is_valid,
                "stage_reached": stage_str,
                "latency_ms": round(step_ms, 1),
                "model": model_label,
                "insight_summary": ins_text[:400],
            }
        )

    assert archive.best_entry is not None
    winner = archive.best_entry

    # Strictly single-pass Held-Out Test Split evaluation at the very end!
    baseline_test_eval = eval_fn(candidate_code=seed_src, split="test")
    evolved_test_eval = eval_fn(candidate_code=winner.full_source, split="test")
    evolved_train_eval = eval_fn(candidate_code=winner.full_source, split="train")

    baseline_test_score = float(baseline_test_eval["overall_score"])
    evolved_test_score = float(evolved_test_eval["overall_score"])
    improvement_pct = compute_improvement_pct(baseline_test_score, evolved_test_score, pillar)

    extra_artifacts: dict[str, Any] = {}
    if pillar == "cluster_scheduling":
        test_trace = generate_hpc_trace(split="test")
        extra_artifacts["tuned_fairshare_test_metrics"] = p1_eval.run_policy_simulation(
            p1_baselines.tuned_fairshare_dispatch, test_trace
        )
    elif pillar == "hpc_code_optimization":
        extra_artifacts["particle_scaling_benchmarks"] = p2_eval.benchmark_particle_scaling(
            winner.full_source, sizes=(512, 1024, 2048, 4096, 8192)
        )
        extra_artifacts["cuda_simd_companion"] = {
            "filename": "pillars/hpc_code_optimization/kernels/ewald_cuda_simd.cu",
            "target_architectures": "NVIDIA Hopper/Ada CUDA + x86_64 AVX-512",
        }

    return {
        "pillar_id": pillar,
        "title": str(spec["title"]),
        "execution_mode": "vertex_ai_live" if mode == "live" else "deterministic_offline_replay",
        "vertex_ai_model": "gemini-2.5-flash" if mode == "live" else "offline-replay",
        "vertex_ai_region": "us-central1",
        "primary_metric": str(spec["primary_metric"]),
        "higher_is_better": True,
        "num_candidates_evaluated": total_steps,
        "valid_candidates_count": valid_count,
        "rejected_by_gates_count": rejected_count,
        "map_elites_niches_populated": len(archive.niche_grid),
        "winning_generation": winner.generation,
        "winning_island": winner.island,
        "winning_niche": winner.niche,
        "train_split": {
            "baseline_score": float(seed_train_eval["overall_score"]),
            "evolved_score": float(evolved_train_eval["overall_score"]),
        },
        "validation_split": {
            "baseline_score": float(seed_val_eval["overall_score"]),
            "evolved_score": float(winner.val_score),
        },
        "baseline_score": baseline_test_score,
        "evolved_score": evolved_test_score,
        "held_out_improvement_pct": improvement_pct,
        "baseline_metrics": baseline_test_eval["metrics"],
        "evolved_metrics": evolved_test_eval["metrics"],
        "kpi_summary": _build_kpi_summary(
            pillar, baseline_test_eval["metrics"], evolved_test_eval["metrics"]
        ),
        "extra_benchmarks": extra_artifacts,
        "generation_history": generation_history,
        "seed_code_block": seed_block,
        "winning_evolved_code_block": winner.block_code,
        "winning_full_program": winner.full_source,
        "code_diff": compute_unified_diff(
            seed_block,
            winner.block_code,
            fromfile=f"pillars/{pillar}/main.py (seed)",
            tofile=f"pillars/{pillar}/main.py (evolved)",
        ),
    }


def run_all_pillars_and_save(
    *,
    num_candidates: int = 14,
    mode: str = "live",
    seed: int = 42,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run evolution across all 3 pillars and write schema-consistent JSON results."""
    out_dir = output_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    pillar_summaries: dict[str, Any] = {}
    for p_name in PILLAR_NAMES:
        res = run_pillar_evolution(
            p_name,
            num_candidates=num_candidates,
            mode=mode,
            seed=seed,
        )
        out_path = out_dir / f"{p_name}_results.json"
        out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        pillar_summaries[p_name] = {
            "title": res["title"],
            "primary_metric": res["primary_metric"],
            "execution_mode": res["execution_mode"],
            "vertex_ai_model": res["vertex_ai_model"],
            "num_candidates_evaluated": res["num_candidates_evaluated"],
            "valid_candidates_count": res["valid_candidates_count"],
            "rejected_by_gates_count": res["rejected_by_gates_count"],
            "winning_generation": res["winning_generation"],
            "baseline_score": res["baseline_score"],
            "evolved_score": res["evolved_score"],
            "held_out_improvement_pct": res["held_out_improvement_pct"],
            "kpi_summary": res["kpi_summary"],
            "artifact_file": f"pillars/results/{p_name}_results.json",
        }

    benchmark_summary = {
        "suite_title": "Enterprise Scientific & AI HPC AlphaEvolve Optimization Suite",
        "execution_mode": "vertex_ai_live" if mode == "live" else "deterministic_offline_replay",
        "vertex_ai_model": "gemini-2.5-flash" if mode == "live" else "offline-replay",
        "vertex_ai_region": "us-central1",
        "total_candidates_evaluated": sum(
            s["num_candidates_evaluated"] for s in pillar_summaries.values()
        ),
        "pillars": pillar_summaries,
    }
    (out_dir / "benchmark_summary.json").write_text(
        json.dumps(benchmark_summary, indent=2) + "\n", encoding="utf-8"
    )
    return benchmark_summary


def load_pillar_summary(pillar: str) -> dict[str, Any]:
    path = RESULTS_DIR / f"{pillar}_results.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_benchmark_summary() -> dict[str, Any]:
    path = RESULTS_DIR / "benchmark_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AlphaEvolve 3-Pillar HPC Optimization Suite.")
    parser.add_argument("--mode", choices=("live", "replay"), default="live")
    parser.add_argument("--candidates", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pillar", choices=(*PILLAR_NAMES, "all"), default="all")
    args = parser.parse_args()

    if args.pillar == "all":
        summary = run_all_pillars_and_save(
            num_candidates=args.candidates,
            mode=args.mode,
            seed=args.seed,
        )
        print(json.dumps(summary, indent=2))
    else:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        res = run_pillar_evolution(
            args.pillar,
            num_candidates=args.candidates,
            mode=args.mode,
            seed=args.seed,
        )
        out_path = RESULTS_DIR / f"{args.pillar}_results.json"
        out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        print(
            f"[{args.pillar}] baseline={res['baseline_score']:.2f} -> "
            f"evolved={res['evolved_score']:.2f} ({res['held_out_improvement_pct']:+.2f}%)"
        )


if __name__ == "__main__":
    main()
