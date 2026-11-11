"""Seed program for Pillar 3: Multi-Horizon HPC Capacity Demand Forecasting & Allocation.

Generates causal 7-day tactical (daily) and 90-day quarterly P50/P99 capacity
forecasts plus commercial resource allocations (1Y/3Y CUDs & Reservations vs.
DWS Flex Start, On-Demand, Spot, and High-IOPS NVMe Parallel FS vs. Object Storage)
under asymmetric capacity stockout penalties (stockouts cost 5x idle headroom).
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np

COMPUTE_POOLS = ("cpu_general", "ultramem_analytics", "gpu_accelerators")


# EVOLVE-BLOCK-START
def forecast_and_allocate(
    history_window: list[dict[str, Any]],
    pool_specs: dict[str, dict[str, Any]],
    asymmetric_penalty_ratio: float = 5.0,
) -> dict[str, dict[str, Any]]:
    """Forecast 7-day tactical & 90-day quarterly demand and allocate commercial tiers.

    Args:
        history_window: Strictly causal daily telemetry `records[0 .. origin_day]`.
            Each entry has `day_index`, `day_of_week` (0=Mon..6=Sun), `quarter_day` (0..90),
            and per-pool dicts:
              - compute pools (`cpu_general`, `ultramem_analytics`, `gpu_accelerators`):
                `{"p50": float, "p99": float}`
              - `tiered_storage_tb`:
                `{"p50": float, "p99": float, "hot_working_set_tb": float}`
        pool_specs: Unit commercial pricing, discount multipliers, `max_safe_spot_share`,
            and `dws_capacity_cap_ratio` per pool.
        asymmetric_penalty_ratio: Ratio of stockout penalty vs On-Demand unit price (default 5.0).

    Returns:
        Dictionary keyed by pool name:
          - For compute pools (`cpu_general`, `ultramem_analytics`, `gpu_accelerators`):
            {
              "tactical_7d_p50": list[float] (len 7),
              "tactical_7d_p99": list[float] (len 7),
              "quarterly_90d_p50": float,
              "quarterly_90d_p99": float,
              "allocation": {
                "cud_1y_3y_units": float,    # 24/7 committed baseline over next window
                "dws_flex_units": float,     # DWS Flex Start burst slice
                "on_demand_units": float,    # Elastic On-Demand ceiling slice
                "spot_units": float,         # Fault-tolerant Spot slice
              }
            }
          - For `tiered_storage_tb`:
            {
              "tactical_7d_p50": list[float] (len 7),
              "tactical_7d_p99": list[float] (len 7),
              "quarterly_90d_p50": float,
              "quarterly_90d_p99": float,
              "allocation": {
                "nvme_parallel_fs_tb": float,
                "object_storage_tb": float,
              }
            }
    """
    # Naive Seed Heuristic:
    # 1. Simple 14-day flat moving average (ignores weekday/weekend cycle, secular AI/GPU
    #    growth trend, and 91-day quarterly submission surges).
    # 2. Static +20% P99 buffer and fixed 50% CUD / 50% On-Demand split (ignores DWS Flex Start,
    #    safe Spot share, and asymmetric 5x stockout cost).
    # 3. Fixed 60% NVMe Parallel FS / 40% Object Storage static storage split.
    recent = history_window[-14:] if len(history_window) >= 14 else history_window
    out: dict[str, dict[str, Any]] = {}

    for pool in COMPUTE_POOLS:
        p50_ma = float(np.mean([d[pool]["p50"] for d in recent]))
        p99_est = p50_ma * 1.20

        cud_units = 0.50 * p99_est
        od_units = 0.50 * p99_est
        out[pool] = {
            "tactical_7d_p50": [p50_ma] * 7,
            "tactical_7d_p99": [p99_est] * 7,
            "quarterly_90d_p50": p50_ma,
            "quarterly_90d_p99": p99_est,
            "allocation": {
                "cud_1y_3y_units": cud_units,
                "dws_flex_units": 0.0,
                "on_demand_units": od_units,
                "spot_units": 0.0,
            },
        }

    st_p50 = float(np.mean([d["tiered_storage_tb"]["p50"] for d in recent]))
    st_p99 = st_p50 * 1.15
    out["tiered_storage_tb"] = {
        "tactical_7d_p50": [st_p50] * 7,
        "tactical_7d_p99": [st_p99] * 7,
        "quarterly_90d_p50": st_p50,
        "quarterly_90d_p99": st_p99,
        "allocation": {
            "nvme_parallel_fs_tb": 0.60 * st_p99,
            "object_storage_tb": 0.40 * st_p99,
        },
    }
    return out
# EVOLVE-BLOCK-END
