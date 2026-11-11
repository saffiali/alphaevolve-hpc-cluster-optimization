"""Tests for Pillar 2 HPC Ewald Solver, IEEE-754 physics gates, and CUDA/SIMD template."""

from __future__ import annotations

import math
import numpy as np

from pillars.common.models import HARD_GATE_PENALTY
from pillars.hpc_code_optimization.evaluator import (
    CUDA_KERNEL_PATH,
    evaluate_program,
    load_cuda_simd_kernel_showcase,
)
from pillars.hpc_code_optimization.physics_reference import erfc_vec


def test_erfc_vec_double_precision_accuracy() -> None:
    grid = np.linspace(-6.0, 10.0, 2000, dtype=np.float64)
    expected = np.array([math.erfc(float(x)) for x in grid], dtype=np.float64)
    actual = erfc_vec(grid)
    rel_err = np.max(np.abs(actual - expected) / np.maximum(expected, 1e-300))
    assert rel_err < 1e-14


def test_seed_solver_passes_all_physics_invariants() -> None:
    res = evaluate_program(split="val")
    assert res["valid"] is True
    assert res["gate_passed"] is True
    assert res["stage_reached"] == "stage_3_throughput_benchmark"
    assert res["metrics"]["rel_l2_force_error"] < 1e-6
    assert res["metrics"]["rel_energy_error"] < 1e-6
    assert res["metrics"]["net_momentum_residual"] < 1e-9
    assert res["metrics"]["symplectic_energy_drift"] < 1e-4
    assert res["overall_score"] > 100.0


def test_cuda_simd_companion_kernel_has_evolve_markers() -> None:
    assert CUDA_KERNEL_PATH.exists()
    showcase = load_cuda_simd_kernel_showcase()
    src = showcase["source"]
    assert "// EVOLVE-BLOCK-START" in src
    assert "// EVOLVE-BLOCK-END" in src
    assert "__shared__" in src


def test_sabotaged_zero_force_kernel_rejected_by_stage2_physics_gate() -> None:
    sabotaged = '''# EVOLVE-BLOCK-START
def solve_ewald_forces(positions, charges, box_length, cutoff, alpha, softening=0.08):
    return np.zeros_like(positions, dtype=np.float64), 0.0
# EVOLVE-BLOCK-END
'''
    res = evaluate_program(candidate_code=sabotaged, split="val")
    assert res["valid"] is False
    assert res["overall_score"] == HARD_GATE_PENALTY
    assert res["stage_reached"] == "stage_2_physics_gate"


def test_sabotaged_approximate_kernel_violating_1e6_tolerance_returns_hard_penalty() -> None:
    sabotaged = '''# EVOLVE-BLOCK-START
def solve_ewald_forces(positions, charges, box_length, cutoff, alpha, softening=0.08):
    # Perturbs softening to 0.085 (creating ~2e-3 force error > 1e-6 gate!)
    pos = np.asarray(positions, dtype=np.float64)
    q = np.asarray(charges, dtype=np.float64)
    n = pos.shape[0]
    disp = pos[None, :, :] - pos[:, None, :]
    disp -= box_length * np.round(disp / box_length)
    r2 = np.sum(disp * disp, axis=-1)
    np.fill_diagonal(r2, cutoff * cutoff + 10.0)
    mask = r2 < cutoff * cutoff
    bad_soft = softening * 1.05
    s = np.sqrt(r2[mask] + bad_soft * bad_soft)
    s_cut = math.sqrt(cutoff * cutoff + bad_soft * bad_soft)
    shift_val = math.erfc(alpha * s_cut) / s_cut
    erfc_s = erfc_vec(alpha * s)
    qij = (q[:, None] * q[None, :])[mask]
    energy = 0.5 * float(np.sum(qij * (erfc_s / s - shift_val)))
    f_mag = np.zeros_like(r2)
    f_mag[mask] = -qij * (erfc_s + (2.0 * alpha / math.sqrt(math.pi)) * s * np.exp(-(alpha * s) ** 2)) / (s ** 3)
    forces = np.sum(f_mag[:, :, None] * disp, axis=1)
    return forces, energy
# EVOLVE-BLOCK-END
'''
    res = evaluate_program(candidate_code=sabotaged, split="val")
    assert res["valid"] is False
    assert res["overall_score"] == HARD_GATE_PENALTY
    assert res["stage_reached"] == "stage_2_physics_gate"


def test_sabotaged_oracle_import_blocked_by_stage1_sandbox() -> None:
    sabotaged = '''# EVOLVE-BLOCK-START
from pillars.hpc_code_optimization.physics_reference import reference_ewald_forces

def solve_ewald_forces(positions, charges, box_length, cutoff, alpha, softening=0.08):
    return reference_ewald_forces(positions, charges, box_length, cutoff, alpha, softening)
# EVOLVE-BLOCK-END
'''
    res = evaluate_program(candidate_code=sabotaged, split="val")
    assert res["valid"] is False
    assert res["overall_score"] == HARD_GATE_PENALTY
    assert res["stage_reached"] == "stage_1_ast_safety"
