"""Tests for Pillar 3 Multi-Horizon Demand Forecasting & Commercial Resource Allocation."""

from __future__ import annotations

from pillars.common.models import HARD_GATE_PENALTY
from pillars.demand_forecasting.evaluator import evaluate_program, run_rolling_backtest
from pillars.demand_forecasting.telemetry_generator import generate_cluster_telemetry


def test_telemetry_splits_strictly_chronological_and_causal() -> None:
    ds = generate_cluster_telemetry(decision_weeks=52, seed=42)
    train_o = ds["origins"]["train"]
    val_o = ds["origins"]["val"]
    test_o = ds["origins"]["test"]

    assert max(train_o) < min(val_o)
    assert max(val_o) < min(test_o)

    seen_max_days: list[int] = []

    def _spy_forecaster(history_window, pool_specs, asymmetric_penalty_ratio=5.0):
        seen_max_days.append(history_window[-1]["day_index"])
        out = {}
        for p in ("cpu_general", "ultramem_analytics", "gpu_accelerators"):
            v = history_window[-1][p]["p50"]
            out[p] = {
                "tactical_7d_p50": [v] * 7,
                "tactical_7d_p99": [v * 1.4] * 7,
                "quarterly_90d_p50": v,
                "quarterly_90d_p99": v * 1.4,
                "allocation": {
                    "cud_1y_3y_units": v * 0.7,
                    "dws_flex_units": v * 0.2,
                    "on_demand_units": v * 0.5,
                    "spot_units": v * 0.1,
                },
            }
        st = history_window[-1]["tiered_storage_tb"]["p50"]
        out["tiered_storage_tb"] = {
            "tactical_7d_p50": [st] * 7,
            "tactical_7d_p99": [st * 1.05] * 7,
            "quarterly_90d_p50": st,
            "quarterly_90d_p99": st * 1.1,
            "allocation": {
                "nvme_parallel_fs_tb": st * 0.35,
                "object_storage_tb": st * 0.70,
            },
        }
        return out

    bt = run_rolling_backtest(_spy_forecaster, split="test")
    assert bt["gate_passed"] is True
    assert seen_max_days == test_o


def test_seed_forecaster_passes_baseline_evaluation() -> None:
    res = evaluate_program(split="test")
    assert res["valid"] is True
    assert res["gate_passed"] is True
    assert res["overall_score"] > HARD_GATE_PENALTY
    assert res["metrics"]["blended_wmape_pct"] > 5.0


def test_sabotaged_quantile_inversion_returns_hard_penalty() -> None:
    sabotaged = '''# EVOLVE-BLOCK-START
def forecast_and_allocate(history_window, pool_specs, asymmetric_penalty_ratio=5.0):
    out = {}
    for pool in COMPUTE_POOLS:
        # Invalid: p99 < p50!
        out[pool] = {
            "tactical_7d_p50": [200.0] * 7,
            "tactical_7d_p99": [150.0] * 7,
            "quarterly_90d_p50": 200.0,
            "quarterly_90d_p99": 150.0,
            "allocation": {"cud_1y_3y_units": 100.0, "dws_flex_units": 0.0, "on_demand_units": 50.0, "spot_units": 0.0},
        }
    out["tiered_storage_tb"] = {
        "tactical_7d_p50": [700.0] * 7,
        "tactical_7d_p99": [720.0] * 7,
        "quarterly_90d_p50": 700.0,
        "quarterly_90d_p99": 720.0,
        "allocation": {"nvme_parallel_fs_tb": 300.0, "object_storage_tb": 420.0},
    }
    return out
# EVOLVE-BLOCK-END
'''
    res = evaluate_program(candidate_code=sabotaged, split="val")
    assert res["valid"] is False
    assert res["overall_score"] == HARD_GATE_PENALTY


def test_sabotaged_zero_allocation_returns_hard_penalty() -> None:
    sabotaged = '''# EVOLVE-BLOCK-START
def forecast_and_allocate(history_window, pool_specs, asymmetric_penalty_ratio=5.0):
    out = {}
    for pool in COMPUTE_POOLS:
        p50 = history_window[-1][pool]["p50"]
        out[pool] = {
            "tactical_7d_p50": [p50] * 7,
            "tactical_7d_p99": [p50 * 1.25] * 7,
            "quarterly_90d_p50": p50,
            "quarterly_90d_p99": p50 * 1.25,
            "allocation": {"cud_1y_3y_units": 0.0, "dws_flex_units": 0.0, "on_demand_units": 0.0, "spot_units": 0.0},
        }
    st = history_window[-1]["tiered_storage_tb"]["p50"]
    out["tiered_storage_tb"] = {
        "tactical_7d_p50": [st] * 7,
        "tactical_7d_p99": [st * 1.05] * 7,
        "quarterly_90d_p50": st,
        "quarterly_90d_p99": st * 1.05,
        "allocation": {"nvme_parallel_fs_tb": 0.0, "object_storage_tb": st * 1.05},
    }
    return out
# EVOLVE-BLOCK-END
'''
    res = evaluate_program(candidate_code=sabotaged, split="val")
    assert res["valid"] is False
    assert res["overall_score"] == HARD_GATE_PENALTY
