"""Out-of-sample rolling-origin backtester & evaluator for Pillar 3 Demand Forecasting."""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from pillars.common.models import HARD_GATE_PENALTY, make_evaluation_result
from pillars.common.sandbox import (
    CandidateTimeoutError,
    SandboxViolationError,
    execute_candidate_module,
    run_with_timeout,
    validate_candidate_source,
)
from pillars.demand_forecasting.telemetry_generator import (
    POOL_COMMERCIAL_SPECS,
    generate_cluster_telemetry,
)

PILLAR_DIR = Path(__file__).resolve().parent
SEED_FILE_PATH = PILLAR_DIR / "main.py"
ALLOWED_IMPORTS = frozenset({"math", "numpy", "typing", "__future__", "statistics", "collections"})
REQUIRED_FUNCTIONS = ("forecast_and_allocate",)

COMPUTE_POOLS = ("cpu_general", "ultramem_analytics", "gpu_accelerators")
ALL_POOLS = ("cpu_general", "ultramem_analytics", "gpu_accelerators", "tiered_storage_tb")

# Hard anti-reward-hacking constraint floors
MAX_STOCKOUT_BREACH_RATE_FLOOR = 0.65   # Blocks zero/under-allocation starvation cheating
MIN_HOT_STORAGE_COVERAGE_FLOOR = 0.85   # Blocks dumping active hot IOPS working set onto cold storage
MAX_WMAPE_SANITY_FLOOR = 65.0           # Blocks absurd forecast values


def _validate_plan_schema(plan: Any) -> tuple[bool, str]:
    if not isinstance(plan, dict):
        return False, "forecast_and_allocate must return a dict keyed by pool name."

    for pool in ALL_POOLS:
        if pool not in plan or not isinstance(plan[pool], dict):
            return False, f"Missing required pool '{pool}' in forecast output."
        entry = plan[pool]
        t50 = entry.get("tactical_7d_p50")
        t99 = entry.get("tactical_7d_p99")
        if not isinstance(t50, (list, tuple, np.ndarray)) or len(t50) != 7:
            return False, f"Pool '{pool}' tactical_7d_p50 must have length 7."
        if not isinstance(t99, (list, tuple, np.ndarray)) or len(t99) != 7:
            return False, f"Pool '{pool}' tactical_7d_p99 must have length 7."

        for k in range(7):
            v50 = float(t50[k])
            v99 = float(t99[k])
            if not (math.isfinite(v50) and math.isfinite(v99)) or v50 < 0.0 or v99 < 0.0:
                return False, f"Pool '{pool}' tactical day {k} forecast is negative or non-finite."
            if v99 + 1e-6 < v50:
                return False, f"Quantile monotonicity violated on '{pool}' day {k}: p99 ({v99:.2f}) < p50 ({v50:.2f})."

        q50 = float(entry.get("quarterly_90d_p50", -1.0))
        q99 = float(entry.get("quarterly_90d_p99", -1.0))
        if not (math.isfinite(q50) and math.isfinite(q99)) or q50 < 0.0 or q99 < 0.0:
            return False, f"Pool '{pool}' quarterly_90d forecast is negative or non-finite."
        if q99 + 1e-6 < q50:
            return False, f"Quarterly quantile monotonicity violated on '{pool}': p99 ({q99:.2f}) < p50 ({q50:.2f})."

        alloc = entry.get("allocation")
        if not isinstance(alloc, dict):
            return False, f"Pool '{pool}' missing 'allocation' dictionary."

        req_keys = (
            ("nvme_parallel_fs_tb", "object_storage_tb")
            if pool == "tiered_storage_tb"
            else ("cud_1y_3y_units", "dws_flex_units", "on_demand_units", "spot_units")
        )
        for ak in req_keys:
            aval = float(alloc.get(ak, -1.0))
            if not math.isfinite(aval) or aval < 0.0:
                return False, f"Pool '{pool}' allocation '{ak}' must be finite and >= 0."

    return True, ""


def run_rolling_backtest(
    model_fn: Any,
    split: str = "val",
    *,
    asymmetric_penalty_ratio: float = 5.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute strict causal rolling-origin backtest across 52-week multi-pool telemetry."""
    dataset = generate_cluster_telemetry(decision_weeks=52, seed=seed)
    records = dataset["records"]
    pool_specs = dataset["pool_specs"]
    if split not in dataset["origins"]:
        raise ValueError(f"Unknown split '{split}'. Expected train, val, or test.")

    origins = dataset["origins"][split]

    tactical_abs_err = 0.0
    tactical_actual_sum = 0.0
    tactical_p99_abs_err = 0.0
    tactical_p99_actual_sum = 0.0
    quarterly_abs_err = 0.0
    quarterly_actual_sum = 0.0

    total_compute_spend_usd = 0.0
    total_storage_spend_usd = 0.0
    total_idle_headroom_cost_usd = 0.0
    total_stockout_penalty_usd = 0.0

    total_pool_days = 0
    stockout_pool_days = 0
    hot_storage_coverage_ratios: list[float] = []

    tier_spend_breakdown = {
        "cud_1y_3y_usd": 0.0,
        "dws_flex_usd": 0.0,
        "on_demand_usd": 0.0,
        "spot_usd": 0.0,
        "nvme_parallel_fs_usd": 0.0,
        "object_storage_usd": 0.0,
    }
    sample_fan_chart: list[dict[str, Any]] = []

    for orig_idx, origin_day in enumerate(origins):
        # Strictly causal slice: candidate NEVER sees records after `origin_day`
        causal_history = [copy.deepcopy(r) for r in records[: origin_day + 1]]
        plan = model_fn(
            causal_history,
            copy.deepcopy(pool_specs),
            float(asymmetric_penalty_ratio),
        )
        ok, err_msg = _validate_plan_schema(plan)
        if not ok:
            return {
                "gate_passed": False,
                "stage_reached": "stage_2_hard_gates",
                "violation_label": "schema_or_quantile_violation",
                "violation_text": err_msg,
            }

        tactical_realized = records[origin_day + 1 : origin_day + 8]
        quarterly_realized = records[origin_day + 1 : origin_day + 91]

        # Capture last origin's GPU trajectory for interactive planner fan-chart visualization
        if orig_idx == len(origins) - 1:
            gpu_plan = plan["gpu_accelerators"]
            for k, day_rec in enumerate(tactical_realized):
                sample_fan_chart.append(
                    {
                        "day_offset": k + 1,
                        "actual_p50": day_rec["gpu_accelerators"]["p50"],
                        "actual_p99": day_rec["gpu_accelerators"]["p99"],
                        "forecast_p50": round(float(gpu_plan["tactical_7d_p50"][k]), 2),
                        "forecast_p99": round(float(gpu_plan["tactical_7d_p99"][k]), 2),
                        "provisioned_total": round(
                            sum(float(v) for v in gpu_plan["allocation"].values()), 2
                        ),
                    }
                )

        # Evaluate compute pools across 7-day tactical realization + 90-day quarterly horizon
        for pool in COMPUTE_POOLS:
            pspec = pool_specs[pool]
            od_rate = float(pspec["on_demand_usd_per_unit"])
            p_entry = plan[pool]
            alloc = p_entry["allocation"]

            cud_u = float(alloc["cud_1y_3y_units"])
            dws_u = float(alloc["dws_flex_units"])
            od_u = float(alloc["on_demand_units"])
            spot_u = float(alloc["spot_units"])
            total_prov_cap = cud_u + dws_u + od_u + spot_u

            # Quarterly 90-day forecast accuracy vs realized 90-day P50 mean and P99 envelope
            q_act_p50 = float(np.mean([r[pool]["p50"] for r in quarterly_realized]))
            q_act_p99 = float(np.percentile([r[pool]["p99"] for r in quarterly_realized], 95))
            quarterly_abs_err += 0.5 * abs(float(p_entry["quarterly_90d_p50"]) - q_act_p50) + 0.5 * abs(
                float(p_entry["quarterly_90d_p99"]) - q_act_p99
            )
            quarterly_actual_sum += 0.5 * (q_act_p50 + q_act_p99)

            # Long-term CUD lock-in check against sustained 90-day P50 baseline:
            # Over-committing 1Y/3Y CUD above 90-day sustained P50 incurs off-peak lock-in waste
            cud_sustained_excess = max(0.0, cud_u - q_act_p50 * 1.05)

            for k, day_rec in enumerate(tactical_realized):
                act_p50 = float(day_rec[pool]["p50"])
                act_p99 = float(day_rec[pool]["p99"])
                pred_p50 = float(p_entry["tactical_7d_p50"][k])
                pred_p99 = float(p_entry["tactical_7d_p99"][k])

                tactical_abs_err += abs(pred_p50 - act_p50)
                tactical_actual_sum += act_p50
                tactical_p99_abs_err += abs(pred_p99 - act_p99)
                tactical_p99_actual_sum += act_p99

                total_pool_days += 1

                # Daily active consumption envelope (weighted P50 sustained + burst spikes)
                active_demand = 0.68 * act_p50 + 0.32 * act_p99

                # 1. CUD 1Y/3Y commitment is paid 24/7 regardless of daily dips
                cud_cost = (cud_u + 0.35 * cud_sustained_excess) * od_rate * float(pspec["cud_1y_3y_multiplier"])
                rem = max(0.0, active_demand - cud_u)

                # 2. Safe Spot slice
                safe_spot_cap = float(pspec["max_safe_spot_share"]) * act_p99
                usable_spot = min(spot_u, safe_spot_cap)
                unsafe_spot_excess = max(0.0, spot_u - safe_spot_cap)
                spot_used = min(rem, usable_spot)
                rem -= spot_used
                # Over-allocating Spot above safe fault-tolerant share triggers preemption waste
                spot_cost = (
                    spot_used * od_rate * float(pspec["spot_multiplier"])
                    + unsafe_spot_excess * od_rate * 0.95
                )

                # 3. DWS Flex Start burst slice
                dws_market_cap = float(pspec["dws_capacity_cap_ratio"]) * act_p99
                usable_dws = min(dws_u, dws_market_cap)
                unusable_dws = max(0.0, dws_u - dws_market_cap)
                dws_used = min(rem, usable_dws)
                rem -= dws_used
                dws_cost = (
                    dws_used * od_rate * float(pspec["dws_flex_multiplier"])
                    + unusable_dws * od_rate * 0.12
                )

                # 4. On-Demand elastic burst up to allocated ceiling
                od_used = min(rem, od_u)
                od_cost = od_used * od_rate

                effective_cap = cud_u + usable_spot + usable_dws + od_u

                # Idle warm-headroom reservation fee vs asymmetric 5x stockout penalty!
                idle_headroom_unit_rate = 0.22 * od_rate
                idle_units = max(0.0, effective_cap - act_p99)
                idle_cost = idle_units * idle_headroom_unit_rate

                shortfall_units = max(0.0, act_p99 - effective_cap)
                if shortfall_units > 1e-4:
                    stockout_pool_days += 1
                stockout_pen = shortfall_units * idle_headroom_unit_rate * float(asymmetric_penalty_ratio)

                # Sanity check against absurd over-provisioning
                if total_prov_cap > 8.0 * act_p99:
                    idle_cost += (total_prov_cap - 8.0 * act_p99) * od_rate

                total_compute_spend_usd += cud_cost + spot_cost + dws_cost + od_cost
                total_idle_headroom_cost_usd += idle_cost
                total_stockout_penalty_usd += stockout_pen

                tier_spend_breakdown["cud_1y_3y_usd"] += cud_cost
                tier_spend_breakdown["spot_usd"] += spot_cost
                tier_spend_breakdown["dws_flex_usd"] += dws_cost
                tier_spend_breakdown["on_demand_usd"] += od_cost

        # Evaluate Tiered Storage pool (High-IOPS NVMe Parallel FS vs Object Storage)
        st_spec = pool_specs["tiered_storage_tb"]
        st_entry = plan["tiered_storage_tb"]
        nvme_tb = float(st_entry["allocation"]["nvme_parallel_fs_tb"])
        obj_tb = float(st_entry["allocation"]["object_storage_tb"])
        tot_st_cap = nvme_tb + obj_tb

        q_st_p50 = float(np.mean([r["tiered_storage_tb"]["p50"] for r in quarterly_realized]))
        quarterly_abs_err += abs(float(st_entry["quarterly_90d_p50"]) - q_st_p50)
        quarterly_actual_sum += q_st_p50

        for k, day_rec in enumerate(tactical_realized):
            act_tot_tb = float(day_rec["tiered_storage_tb"]["p99"])
            act_hot_tb = float(day_rec["tiered_storage_tb"]["hot_working_set_tb"])
            pred_st_p50 = float(st_entry["tactical_7d_p50"][k])

            tactical_abs_err += abs(pred_st_p50 - float(day_rec["tiered_storage_tb"]["p50"]))
            tactical_actual_sum += float(day_rec["tiered_storage_tb"]["p50"])

            hot_cov = nvme_tb / max(act_hot_tb, 1e-6)
            hot_storage_coverage_ratios.append(min(1.5, hot_cov))

            nvme_cost = nvme_tb * float(st_spec["nvme_parallel_fs_usd_per_tb_day"])
            obj_cost = obj_tb * float(st_spec["object_storage_usd_per_tb_day"])
            total_storage_spend_usd += nvme_cost + obj_cost
            tier_spend_breakdown["nvme_parallel_fs_usd"] += nvme_cost
            tier_spend_breakdown["object_storage_usd"] += obj_cost

            total_pool_days += 1
            st_shortfall = max(0.0, act_tot_tb - tot_st_cap)
            hot_shortfall = max(0.0, act_hot_tb - nvme_tb)
            if st_shortfall > 1e-3 or hot_shortfall > 0.15 * act_hot_tb:
                stockout_pool_days += 1
            # Storage stockout or hot-IOPS throttling penalty at asymmetric multiplier
            total_stockout_penalty_usd += (
                st_shortfall * float(st_spec["nvme_parallel_fs_usd_per_tb_day"]) * float(asymmetric_penalty_ratio)
                + hot_shortfall * float(st_spec["nvme_parallel_fs_usd_per_tb_day"]) * 0.65 * float(asymmetric_penalty_ratio)
            )

    tactical_wmape_pct = (tactical_abs_err / max(tactical_actual_sum, 1e-6)) * 100.0
    tactical_p99_wmape_pct = (tactical_p99_abs_err / max(tactical_p99_actual_sum, 1e-6)) * 100.0
    quarterly_wmape_pct = (quarterly_abs_err / max(quarterly_actual_sum, 1e-6)) * 100.0
    blended_wmape_pct = 0.45 * tactical_wmape_pct + 0.30 * tactical_p99_wmape_pct + 0.25 * quarterly_wmape_pct

    stockout_rate = stockout_pool_days / max(1, total_pool_days)
    mean_hot_cov = float(np.mean(hot_storage_coverage_ratios)) if hot_storage_coverage_ratios else 0.0

    effective_infra_tco_k = (total_compute_spend_usd + total_storage_spend_usd + total_idle_headroom_cost_usd) / 1000.0
    stockout_pen_k = total_stockout_penalty_usd / 1000.0
    total_risk_adj_tco_k = effective_infra_tco_k + stockout_pen_k

    # Negated total risk-adjusted TCO ($K) + forecast WMAPE accuracy term
    raw_score = -(total_risk_adj_tco_k + 8.5 * blended_wmape_pct)

    return {
        "gate_passed": True,
        "raw_overall_score": round(raw_score, 4),
        "total_risk_adjusted_tco_k_usd": round(total_risk_adj_tco_k, 3),
        "effective_infra_tco_k_usd": round(effective_infra_tco_k, 3),
        "compute_spend_k_usd": round(total_compute_spend_usd / 1000.0, 3),
        "storage_spend_k_usd": round(total_storage_spend_usd / 1000.0, 3),
        "idle_headroom_cost_k_usd": round(total_idle_headroom_cost_usd / 1000.0, 3),
        "asymmetric_stockout_penalty_k_usd": round(stockout_pen_k, 3),
        "tactical_7d_wmape_pct": round(tactical_wmape_pct, 2),
        "tactical_7d_p99_wmape_pct": round(tactical_p99_wmape_pct, 2),
        "quarterly_90d_wmape_pct": round(quarterly_wmape_pct, 2),
        "blended_wmape_pct": round(blended_wmape_pct, 2),
        "stockout_breach_rate": round(stockout_rate, 4),
        "stockout_breach_rate_pct": round(stockout_rate * 100.0, 2),
        "hot_storage_coverage_ratio": round(mean_hot_cov, 3),
        "asymmetric_penalty_ratio": float(asymmetric_penalty_ratio),
        "origins_evaluated": len(origins),
        "tier_spend_breakdown_k_usd": {
            k: round(v / 1000.0, 3) for k, v in tier_spend_breakdown.items()
        },
        "sample_gpu_fan_chart": sample_fan_chart,
    }


def evaluate_program(
    candidate_code: str | None = None,
    split: str = "val",
    scenario_overrides: dict[str, Any] | None = None,
    *,
    require_markers: bool = True,
) -> dict[str, Any]:
    """3-stage AlphaEvolve evaluator for Pillar 3 Demand Forecasting & Allocation."""
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
        mod = execute_candidate_module(spliced_source, "pillar3_candidate")
        model_fn = mod.__dict__["forecast_and_allocate"]
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
            insights=[("module_load_error", f"Failed to load candidate forecaster: {exc}")],
        )

    overrides = scenario_overrides or {}
    asym_ratio = float(overrides.get("asymmetric_penalty_ratio", 5.0))
    seed = int(overrides.get("seed", 42))

    try:
        bt = run_with_timeout(
            lambda: run_rolling_backtest(
                model_fn,
                split=split,
                asymmetric_penalty_ratio=asym_ratio,
                seed=seed,
            ),
            timeout_sec=15.0,
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
            insights=[("runtime_exception", f"Backtest raised runtime exception: {exc}")],
        )

    if not bt.get("gate_passed", False):
        return make_evaluation_result(
            primary_metric="overall_score",
            overall_score=HARD_GATE_PENALTY,
            valid=False,
            gate_passed=False,
            stage_reached="stage_2_hard_gates",
            split=split,
            metrics=bt,
            insights=[(str(bt["violation_label"]), str(bt["violation_text"]))],
        )

    # Stage 2: Hard SLA & Storage Coverage anti-reward-hacking gates
    violations: list[tuple[str, str]] = []
    if bt["stockout_breach_rate"] > MAX_STOCKOUT_BREACH_RATE_FLOOR:
        violations.append(
            (
                "hard_gate_stockout_sla",
                f"Violated SLA stockout ceiling (<=65%): stockout_breach_rate={bt['stockout_breach_rate_pct']:.2f}%. "
                "Provision sufficient CUD + DWS Flex + Spot + On-Demand capacity to cover P99 bursts.",
            )
        )
    if bt["hot_storage_coverage_ratio"] < MIN_HOT_STORAGE_COVERAGE_FLOOR:
        violations.append(
            (
                "hard_gate_hot_storage_coverage",
                f"Violated High-IOPS NVMe hot working-set coverage floor (>=0.85): "
                f"coverage={bt['hot_storage_coverage_ratio']:.3f}. Allocate `nvme_parallel_fs_tb` >= 1.05 * hot_working_set_tb.",
            )
        )
    if bt["blended_wmape_pct"] > MAX_WMAPE_SANITY_FLOOR:
        violations.append(
            (
                "hard_gate_wmape_sanity",
                f"Violated forecast WMAPE sanity floor (<=65%): blended_wmape_pct={bt['blended_wmape_pct']:.2f}%.",
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
            metrics=bt,
            insights=violations,
        )

    # Stage 3: Diagnostic feedback for Gemini parent mutation
    insights: list[tuple[str, str]] = [
        (
            "backtest_summary",
            f"overall_score={bt['raw_overall_score']:.2f} | Risk-Adj TCO=${bt['total_risk_adjusted_tco_k_usd']:.1f}K "
            f"(Infra=${bt['effective_infra_tco_k_usd']:.1f}K + StockoutPen=${bt['asymmetric_stockout_penalty_k_usd']:.1f}K) | "
            f"Blended WMAPE={bt['blended_wmape_pct']:.2f}% (7d={bt['tactical_7d_wmape_pct']:.2f}%, 90d={bt['quarterly_90d_wmape_pct']:.2f}%) | "
            f"Stockout rate={bt['stockout_breach_rate_pct']:.2f}%",
        )
    ]
    if bt["tactical_7d_wmape_pct"] > 9.0:
        insights.append(
            (
                "day_of_week_and_trend_decomposition",
                "Tactical 7-day WMAPE is elevated because a flat moving average misses day-of-week seasonality "
                "(Mon-Fri vs Sat-Sun), secular growth slope, and 91-day quarterly surges (`quarter_day` near 70..86). "
                "Estimate weekday multipliers (`day_of_week` 0..6), linear/exponential trend slope, and `quarter_day` surge profile!",
            )
        )
    if bt["tier_spend_breakdown_k_usd"]["dws_flex_usd"] < 1.0:
        insights.append(
            (
                "commercial_tier_mix_opportunity",
                "Zero DWS Flex Start or Spot utilization detected. Size `cud_1y_3y_units` near the sustained P50/off-peak "
                "floor (~0.78-0.88 * tactical_p50_mean), allocate `spot_units` up to `max_safe_spot_share * peak_p99`, "
                "`dws_flex_units` up to `dws_capacity_cap_ratio * peak_p99`, and cover remaining asymmetric P99 tail with `on_demand_units`!",
            )
        )
    if bt["storage_spend_k_usd"] > 35.0:
        insights.append(
            (
                "storage_tiering_overprovisioning",
                f"Storage spend is ${bt['storage_spend_k_usd']:.1f}K because too much cold data sits on $8/TB-day NVMe Parallel FS. "
                "Right-size `nvme_parallel_fs_tb` to ~1.06 * predicted peak `hot_working_set_tb` and place remaining TB on $0.67/TB-day `object_storage_tb`!",
            )
        )

    return make_evaluation_result(
        primary_metric="overall_score",
        overall_score=float(bt["raw_overall_score"]),
        valid=True,
        gate_passed=True,
        stage_reached="stage_3_benchmark",
        split=split,
        metrics=bt,
        insights=insights,
        secondary_scores={
            "neg_risk_adjusted_tco_k_usd": -float(bt["total_risk_adjusted_tco_k_usd"]),
            "neg_blended_wmape_pct": -float(bt["blended_wmape_pct"]),
            "neg_stockout_breach_rate_pct": -float(bt["stockout_breach_rate_pct"]),
        },
    )
