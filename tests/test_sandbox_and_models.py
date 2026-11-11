"""Unit tests for AlphaEvolve EVOLVE-BLOCK splicing, AST sandbox, and evaluation models."""

from __future__ import annotations

import pytest

from pillars.common.models import HARD_GATE_PENALTY, make_evaluation_result
from pillars.common.sandbox import (
    CandidateTimeoutError,
    SandboxViolationError,
    compute_unified_diff,
    extract_evolve_block,
    replace_evolve_block,
    run_with_timeout,
    validate_candidate_source,
)

SAMPLE_SEED = '''import math

CONST_GUARD = 10


# EVOLVE-BLOCK-START
def compute_value(x: float) -> float:
    return x + CONST_GUARD
# EVOLVE-BLOCK-END
'''


def test_extract_and_replace_evolve_block() -> None:
    block = extract_evolve_block(SAMPLE_SEED)
    assert "def compute_value" in block

    new_block = "def compute_value(x: float) -> float:\n    return x * 2.0 + CONST_GUARD"
    spliced = replace_evolve_block(SAMPLE_SEED, new_block)
    assert "# EVOLVE-BLOCK-START\ndef compute_value" in spliced
    assert "x * 2.0" in spliced
    diff = compute_unified_diff(block, new_block)
    assert "-    return x + CONST_GUARD" in diff
    assert "+    return x * 2.0 + CONST_GUARD" in diff


def test_marker_format_violations() -> None:
    with pytest.raises(SandboxViolationError):
        extract_evolve_block("def compute_value(x):\n    return x\n")

    with pytest.raises(SandboxViolationError):
        extract_evolve_block("  # EVOLVE-BLOCK-START\ndef f():\n    return 1\n# EVOLVE-BLOCK-END\n")

    with pytest.raises(SandboxViolationError):
        extract_evolve_block("# EVOLVE-BLOCK-START inline note\ndef f():\n    return 1\n# EVOLVE-BLOCK-END\n")

    with pytest.raises(SandboxViolationError):
        extract_evolve_block("# EVOLVE-BLOCK-START\n# comment only\n# EVOLVE-BLOCK-END\n")


def test_outer_harness_tampering_rejected() -> None:
    tampered = SAMPLE_SEED.replace("CONST_GUARD = 10", "CONST_GUARD = 999999")
    with pytest.raises(SandboxViolationError, match="Outer harness tampering"):
        validate_candidate_source(
            tampered,
            canonical_seed_source=SAMPLE_SEED,
            required_functions=("compute_value",),
            allowed_imports=frozenset({"math"}),
        )


@pytest.mark.parametrize(
    "bad_snippet",
    [
        "# EVOLVE-BLOCK-START\nimport os\ndef compute_value(x):\n    return 1.0\n# EVOLVE-BLOCK-END",
        "# EVOLVE-BLOCK-START\nimport subprocess\ndef compute_value(x):\n    return 1.0\n# EVOLVE-BLOCK-END",
        "# EVOLVE-BLOCK-START\ndef compute_value(x):\n    return eval('1+1')\n# EVOLVE-BLOCK-END",
        "# EVOLVE-BLOCK-START\ndef compute_value(x):\n    return ().__class__.__bases__\n# EVOLVE-BLOCK-END",
    ],
)
def test_ast_sandbox_blocks_unsafe_constructs(bad_snippet: str) -> None:
    with pytest.raises(SandboxViolationError):
        validate_candidate_source(
            bad_snippet,
            canonical_seed_source=SAMPLE_SEED,
            required_functions=("compute_value",),
            allowed_imports=frozenset({"math"}),
        )


def test_timeout_guard_terminates_runaway_loop() -> None:
    def _infinite_loop() -> int:
        acc = 0
        while True:
            acc += 1
        return acc

    with pytest.raises(CandidateTimeoutError):
        run_with_timeout(_infinite_loop, timeout_sec=0.15)


def test_make_evaluation_result_forces_hard_gate_penalty() -> None:
    res = make_evaluation_result(
        primary_metric="overall_score",
        overall_score=99999.0,
        valid=True,
        gate_passed=False,
        stage_reached="stage_2_hard_gates",
        split="val",
        metrics={},
        insights=[("gate", "failed")],
    )
    assert res["overall_score"] == HARD_GATE_PENALTY
    assert res["valid"] is False
    assert res["scores"]["scores"][0]["score"] == HARD_GATE_PENALTY
