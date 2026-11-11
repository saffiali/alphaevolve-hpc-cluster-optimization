"""Tests for shared evolutionary runner (deterministic replay + live Vertex AI result artifacts)."""

from __future__ import annotations

from pathlib import Path
import pytest

from pillars.common.evolution_runner import (
    PILLAR_NAMES,
    RESULTS_DIR,
    get_pillar_evaluator,
    load_benchmark_summary,
    load_pillar_summary,
    propose_and_evaluate_step,
    run_pillar_evolution,
)


def test_deterministic_offline_replay_reproducibility() -> None:
    r1 = run_pillar_evolution("demand_forecasting", mode="replay", seed=42)
    r2 = run_pillar_evolution("demand_forecasting", mode="replay", seed=42)
    assert r1["baseline_score"] == r2["baseline_score"]
    assert r1["evolved_score"] == r2["evolved_score"]
    assert r1["evolved_score"] > r1["baseline_score"]
    assert r1["rejected_by_gates_count"] >= 1


def test_single_step_replay_helper() -> None:
    step = propose_and_evaluate_step("cluster_scheduling", mode="replay")
    assert step["valid"] is True
    assert step["candidate_score"] > step["parent_score"]
    assert len(step["code_diff"]) > 20


@pytest.mark.parametrize("pillar", PILLAR_NAMES)
def test_exported_live_vertex_ai_result_artifacts_schema_and_truthfulness(pillar: str) -> None:
    res_file = RESULTS_DIR / f"{pillar}_results.json"
    assert res_file.exists(), f"Missing artifact {res_file}"

    data = load_pillar_summary(pillar)
    assert data["pillar_id"] == pillar
    assert data["execution_mode"] == "vertex_ai_live"
    assert data["vertex_ai_model"] == "gemini-2.5-flash"
    assert data["num_candidates_evaluated"] >= 12
    assert data["evolved_score"] > data["baseline_score"]
    assert data["held_out_improvement_pct"] > 10.0
    assert len(data["generation_history"]) >= 13
    assert len(data["seed_code_block"].strip()) > 30
    assert len(data["winning_evolved_code_block"].strip()) > 30
    assert data["winning_evolved_code_block"] != data["seed_code_block"]
    assert "@@" in data["code_diff"]

    # Re-evaluate the saved winning program live on the held-out test split to prove authenticity
    eval_fn = get_pillar_evaluator(pillar)
    recheck = eval_fn(candidate_code=data["winning_full_program"], split="test")
    assert recheck["valid"] is True
    if pillar != "hpc_code_optimization":
        # Deterministic discrete-event & backtest evaluators match saved score to floating precision
        assert abs(recheck["overall_score"] - data["evolved_score"]) < 1e-3
    else:
        # Hardware wall-clock throughput has minor jitter; verify strict physics + >10x speedup!
        assert recheck["metrics"]["rel_l2_force_error"] < 1e-6
        assert recheck["metrics"]["speedup_vs_baseline"] > 10.0


def test_benchmark_summary_consistency() -> None:
    summary_file = RESULTS_DIR / "benchmark_summary.json"
    assert summary_file.exists()
    summary = load_benchmark_summary()
    assert summary["execution_mode"] == "vertex_ai_live"
    assert set(summary["pillars"].keys()) == set(PILLAR_NAMES)
