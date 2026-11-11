"""Shared AlphaEvolve evaluation models, AST sandbox, Vertex AI client, and evolutionary runner."""

from pillars.common.models import (
    EVOLVE_END_MARKER,
    EVOLVE_START_MARKER,
    HARD_GATE_PENALTY,
    AlphaEvolveEvaluationInsight,
    AlphaEvolveEvaluationInsights,
    AlphaEvolveEvaluationScore,
    AlphaEvolveEvaluationScores,
    AlphaEvolveProgramEvaluation,
    make_evaluation_result,
)
from pillars.common.sandbox import (
    SandboxViolationError,
    compute_unified_diff,
    execute_candidate_module,
    extract_evolve_block,
    replace_evolve_block,
    validate_candidate_source,
)

__all__ = [
    "EVOLVE_END_MARKER",
    "EVOLVE_START_MARKER",
    "HARD_GATE_PENALTY",
    "AlphaEvolveEvaluationInsight",
    "AlphaEvolveEvaluationInsights",
    "AlphaEvolveEvaluationScore",
    "AlphaEvolveEvaluationScores",
    "AlphaEvolveProgramEvaluation",
    "SandboxViolationError",
    "compute_unified_diff",
    "execute_candidate_module",
    "extract_evolve_block",
    "make_evaluation_result",
    "replace_evolve_block",
    "validate_candidate_source",
]
