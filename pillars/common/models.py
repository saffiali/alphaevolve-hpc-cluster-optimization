"""AlphaEvolve-compatible Pydantic evaluation models and structured result helpers."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field

HARD_GATE_PENALTY: float = -1e12
EVOLVE_START_MARKER: str = "# EVOLVE-BLOCK-START"
EVOLVE_END_MARKER: str = "# EVOLVE-BLOCK-END"

try:
    from alpha_evolve.models import (  # type: ignore[import-not-found]
        AlphaEvolveEvaluationInsight,
        AlphaEvolveEvaluationInsights,
        AlphaEvolveEvaluationScore,
        AlphaEvolveEvaluationScores,
        AlphaEvolveProgramEvaluation,
    )
except ImportError:

    class AlphaEvolveEvaluationScore(BaseModel):
        """Single scalar metric entry produced by an AlphaEvolve evaluator."""

        metric: str
        score: float | None = None

    class AlphaEvolveEvaluationScores(BaseModel):
        """Collection of metric scores for a candidate program."""

        scores: list[AlphaEvolveEvaluationScore] = Field(default_factory=list)

    class AlphaEvolveEvaluationInsight(BaseModel):
        """Actionable diagnostic feedback item returned to the LLM mutator."""

        label: str
        text: str

    class AlphaEvolveEvaluationInsights(BaseModel):
        """Collection of diagnostic insights for parent-selection and mutation."""

        insights: list[AlphaEvolveEvaluationInsight] = Field(default_factory=list)

    class AlphaEvolveProgramEvaluation(BaseModel):
        """Top-level schema matching Google Cloud HPC AlphaEvolve worker output."""

        scores: AlphaEvolveEvaluationScores
        insights: AlphaEvolveEvaluationInsights


def make_evaluation_result(
    *,
    primary_metric: str,
    overall_score: float,
    valid: bool,
    gate_passed: bool,
    stage_reached: str,
    split: str,
    metrics: dict[str, Any],
    insights: list[tuple[str, str]],
    secondary_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a unified evaluation dictionary compatible with AlphaEvolve and Python callers.

    Whenever `valid` or `gate_passed` is False (or `overall_score` is NaN/Inf),
    `overall_score` is forced to `HARD_GATE_PENALTY` (-1e12) to prevent reward hacking.
    """
    if not valid or not gate_passed or not math.isfinite(overall_score):
        effective_score = HARD_GATE_PENALTY
    else:
        effective_score = float(overall_score)

    score_entries = [
        AlphaEvolveEvaluationScore(metric=primary_metric, score=effective_score),
    ]
    if primary_metric != "overall_score":
        score_entries.append(
            AlphaEvolveEvaluationScore(metric="overall_score", score=effective_score)
        )
    if secondary_scores:
        for name, val in secondary_scores.items():
            if math.isfinite(val):
                score_entries.append(
                    AlphaEvolveEvaluationScore(metric=name, score=float(val))
                )

    ae_payload = AlphaEvolveProgramEvaluation(
        scores=AlphaEvolveEvaluationScores(scores=score_entries),
        insights=AlphaEvolveEvaluationInsights(
            insights=[
                AlphaEvolveEvaluationInsight(label=lbl, text=txt[:1200])
                for lbl, txt in insights
            ]
        ),
    ).model_dump()

    return {
        "overall_score": effective_score,
        "primary_metric": primary_metric,
        "valid": bool(valid and gate_passed),
        "gate_passed": bool(gate_passed),
        "stage_reached": stage_reached,
        "split": split,
        "metrics": metrics,
        "scores": ae_payload["scores"],
        "insights": ae_payload["insights"],
    }
