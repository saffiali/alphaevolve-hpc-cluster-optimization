"""Deterministic multi-pool HPC cluster capacity telemetry generator (52-week rolling horizon).

Generates daily P50 average load, P99 burst load, and hot-vs-cold storage working-set
ratios across 4 enterprise resource pools:
  1. `cpu_general` (Gen-4 x86 compute, k-vCPU-hours/day)
  2. `ultramem_analytics` (High-memory analytics, node-hours/day)
  3. `gpu_accelerators` (Multi-GPU AI & simulation, GPU-hours/day)
  4. `tiered_storage_tb` (Active High-IOPS NVMe Parallel FS vs Object Storage, TB)
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np

POOL_COMMERCIAL_SPECS: dict[str, dict[str, Any]] = {
    "cpu_general": {
        "unit": "k_vcpu_hrs_per_day",
        "on_demand_usd_per_unit": 38.0,
        "cud_1y_3y_multiplier": 0.58,   # 42% CUD discount
        "dws_flex_multiplier": 0.76,    # 24% DWS discount
        "spot_multiplier": 0.32,        # 68% Spot discount
        "max_safe_spot_share": 0.36,
        "dws_capacity_cap_ratio": 0.25,
    },
    "ultramem_analytics": {
        "unit": "node_hrs_per_day",
        "on_demand_usd_per_unit": 235.0,
        "cud_1y_3y_multiplier": 0.55,   # 45% CUD discount
        "dws_flex_multiplier": 0.72,    # 28% DWS discount
        "spot_multiplier": 0.35,        # 65% Spot discount
        "max_safe_spot_share": 0.08,
        "dws_capacity_cap_ratio": 0.30,
    },
    "gpu_accelerators": {
        "unit": "gpu_hrs_per_day",
        "on_demand_usd_per_unit": 94.0,
        "cud_1y_3y_multiplier": 0.55,   # 45% 1Y/3Y GPU CUD + Reservation discount
        "dws_flex_multiplier": 0.70,    # 30% DWS Flex Start GPU discount
        "spot_multiplier": 0.35,        # 65% Spot GPU discount
        "max_safe_spot_share": 0.12,
        "dws_capacity_cap_ratio": 0.38,
    },
    "tiered_storage_tb": {
        "unit": "tb_stored",
        "nvme_parallel_fs_usd_per_tb_day": 8.00,  # High-IOPS NVMe Parallel FS (~$240/TB-mo)
        "object_storage_usd_per_tb_day": 0.67,    # Cloud Object Storage (~$20/TB-mo)
    },
}


def generate_cluster_telemetry(
    decision_weeks: int = 52,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate 52 weeks of decision origins + 90-day forward realization horizon (454 days total).

    Strict Causal Guarantee:
      At any rolling decision origin `t`, `forecast_and_allocate` receives ONLY
      `days[0 : t + 1]` (`day <= t`). Future days `t + 1 .. t + 90` exist strictly inside
      the evaluator harness to score 7-day tactical and 90-day quarterly realizations.
    """
    total_days = decision_weeks * 7 + 90
    rng = np.random.default_rng(seed)
    days = np.arange(total_days, dtype=np.float64)
    dow = (days.astype(int)) % 7  # 0=Mon .. 4=Fri, 5=Sat, 6=Sun

    # Weekday vs weekend R&D activity factor
    weekday_factor = np.where(dow < 5, 1.16, 0.66)
    # Slight mid-week peak (Tue-Thu)
    weekday_factor = np.where((dow >= 1) & (dow <= 3), 1.22, weekday_factor)

    # Quarterly R&D release / submission surges near end of each 91-day quarter
    q_phase = np.mod(days, 91.0)
    quarterly_surge = 1.0 + 0.28 * np.exp(-0.5 * ((q_phase - 78.0) / 7.5) ** 2)

    # 1. CPU General (k-vCPU-hrs/day)
    cpu_trend = 185.0 * (1.0 + 0.24 * (days / 364.0))
    cpu_p50 = cpu_trend * weekday_factor * (0.94 + 0.12 * quarterly_surge) * rng.lognormal(0.0, 0.055, total_days)
    cpu_burst_mult = 1.22 + 0.10 * (quarterly_surge - 1.0) / 0.28 + rng.uniform(0.0, 0.11, total_days)
    cpu_p99 = cpu_p50 * cpu_burst_mult

    # 2. Ultramem Analytics (node-hrs/day)
    umem_trend = 42.0 * (1.0 + 0.30 * (days / 364.0))
    umem_wave = 1.0 + 0.10 * np.sin(2.0 * math.pi * days / 28.0)
    umem_p50 = umem_trend * np.where(dow < 5, 1.12, 0.78) * umem_wave * quarterly_surge * rng.lognormal(0.0, 0.065, total_days)
    umem_p99 = umem_p50 * (1.26 + rng.uniform(0.0, 0.13, total_days))

    # 3. GPU Accelerators (gpu-hrs/day — strong secular AI growth + bursty multi-GPU training waves)
    gpu_trend = 140.0 * (1.0 + 0.68 * (days / 364.0))
    gpu_weekday = np.where(dow < 5, 1.14, 0.82)  # Long training runs spill into weekends
    gpu_p50 = gpu_trend * gpu_weekday * quarterly_surge * rng.lognormal(0.0, 0.075, total_days)
    gpu_p99 = gpu_p50 * (1.30 + 0.14 * (quarterly_surge - 1.0) / 0.28 + rng.uniform(0.0, 0.14, total_days))

    # 4. Tiered Storage (total TB + active hot high-IOPS working-set fraction)
    storage_total_tb = 680.0 * (1.0 + 0.54 * (days / 364.0)) + 28.0 * (quarterly_surge - 1.0) / 0.28
    storage_total_tb *= rng.normal(1.0, 0.012, total_days)
    # Active hot working set spikes during weekday & quarterly training surges (~21%..31% of total footprint)
    hot_ratio = np.clip(
        0.21 + 0.04 * (dow < 5).astype(float) + 0.06 * (quarterly_surge - 1.0) / 0.28 + rng.normal(0.0, 0.015, total_days),
        0.16,
        0.38,
    )
    storage_hot_p99_tb = storage_total_tb * (hot_ratio + 0.025)

    records: list[dict[str, Any]] = []
    for d in range(total_days):
        records.append(
            {
                "day_index": int(d),
                "day_of_week": int(dow[d]),
                "quarter_day": int(q_phase[d]),
                "cpu_general": {
                    "p50": round(float(cpu_p50[d]), 3),
                    "p99": round(float(cpu_p99[d]), 3),
                },
                "ultramem_analytics": {
                    "p50": round(float(umem_p50[d]), 3),
                    "p99": round(float(umem_p99[d]), 3),
                },
                "gpu_accelerators": {
                    "p50": round(float(gpu_p50[d]), 3),
                    "p99": round(float(gpu_p99[d]), 3),
                },
                "tiered_storage_tb": {
                    "p50": round(float(storage_total_tb[d]), 3),
                    "p99": round(float(storage_total_tb[d] * 1.03), 3),
                    "hot_working_set_tb": round(float(storage_hot_p99_tb[d]), 3),
                },
            }
        )

    # Rolling weekly origin indices across the 52-week year (starting at week 12 = day 84 so >=12 weeks history exists):
    #   - train: weeks 12..27 (16 weekly rolling origins)
    #   - val:   weeks 28..39 (12 weekly rolling origins)
    #   - test:  weeks 40..51 (12 strictly held-out weekly rolling origins)
    origins = {
        "train": [w * 7 for w in range(12, 28)],
        "val": [w * 7 for w in range(28, 40)],
        "test": [w * 7 for w in range(40, 52)],
    }

    return {
        "records": records,
        "origins": origins,
        "pool_specs": POOL_COMMERCIAL_SPECS,
    }
