"""3-Stage Cascaded Evaluator for Pillar 2: 3D Ewald Force-Field Kernel Optimization.

Cascaded Pipeline:
  - Stage 1 (`stage_1_ast_safety`): Syntax, EVOLVE-BLOCK integrity, outer-harness immutability,
    and strict AST blocklist preventing oracle imports or unsafe syscalls.
  - Stage 2 (`stage_2_physics_gate`): IEEE-754 double-precision physics verification against
    ground-truth Ewald reference:
      1. Relative L2 Force Error < 1e-6
      2. Relative Potential Energy Error < 1e-6
      3. Net Momentum Residual < 1e-9 (Newton's 3rd Law invariance)
      4. Symplectic Velocity-Verlet Energy Conservation Drift < 1e-4
    Any violation immediately returns `HARD_GATE_PENALTY` (-1e12) to prevent reward hacking.
  - Stage 3 (`stage_3_throughput_benchmark`): High-precision median `particles_per_sec`
    and speedup benchmark across split-isolated molecular conformations.
"""

from __future__ import annotations

import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from pillars.common.models import HARD_GATE_PENALTY, make_evaluation_result
from pillars.common.sandbox import (
    CandidateTimeoutError,
    SandboxViolationError,
    execute_candidate_module,
    run_with_timeout,
    validate_ast_safety,
    validate_candidate_source,
)
from pillars.hpc_code_optimization.physics_reference import (
    generate_molecular_conformation,
    reference_ewald_forces,
    verify_symplectic_energy_conservation,
)

PILLAR_DIR = Path(__file__).resolve().parent
SEED_FILE_PATH = PILLAR_DIR / "main.py"
CUDA_KERNEL_PATH = PILLAR_DIR / "kernels" / "ewald_cuda_simd.cu"

OUTER_ALLOWED_IMPORTS = frozenset({"math", "numpy", "typing", "__future__", "pillars"})
BLOCK_ALLOWED_IMPORTS = frozenset({"math", "numpy"})
REQUIRED_FUNCTIONS = ("solve_ewald_forces",)

MAX_REL_L2_FORCE_ERROR = 1e-6
MAX_REL_ENERGY_ERROR = 1e-6
MAX_NET_MOMENTUM_RESIDUAL = 1e-9
MAX_SYMPLECTIC_ENERGY_DRIFT = 1e-4

SPLIT_SEEDS: dict[str, tuple[tuple[int, str, int], ...]] = {
    "train": (
        (256, "solvated_biomolecule", 101),
        (384, "ionic_lattice", 102),
    ),
    "val": (
        (320, "solvated_biomolecule", 201),
        (448, "clustered_droplet", 202),
    ),
    "test": (
        (384, "solvated_biomolecule", 301),
        (512, "ionic_lattice", 302),
        (640, "clustered_droplet", 303),
    ),
}

_BASELINE_THROUGHPUT_CACHE: dict[str, float] = {}


def _run_stage2_and_stage3(
    solver_fn: Any,
    split: str,
    *,
    benchmark_repeats: int = 3,
) -> dict[str, Any]:
    if split not in SPLIT_SEEDS:
        raise ValueError(f"Unknown split '{split}'. Expected train, val, or test.")

    specs = SPLIT_SEEDS[split]
    max_force_err = 0.0
    max_energy_err = 0.0
    max_momentum_res = 0.0
    max_symplectic_drift = 0.0

    # Stage 2: Strict physics gate across split conformations
    conformations: list[dict[str, Any]] = []
    for n_part, regime, seed in specs:
        conf = generate_molecular_conformation(n_part, seed, regime=regime)
        conformations.append(conf)

        ref_forces, ref_energy = reference_ewald_forces(
            conf["positions"],
            conf["charges"],
            conf["box_length"],
            conf["cutoff"],
            conf["alpha"],
            conf["softening"],
        )
        cand_out = solver_fn(
            conf["positions"].copy(),
            conf["charges"].copy(),
            float(conf["box_length"]),
            float(conf["cutoff"]),
            float(conf["alpha"]),
            float(conf["softening"]),
        )
        if not isinstance(cand_out, tuple) or len(cand_out) != 2:
            return {
                "gate_passed": False,
                "stage_reached": "stage_2_physics_gate",
                "violation_label": "invalid_return_signature",
                "violation_text": "solve_ewald_forces must return a 2-tuple `(forces_ndarray, potential_energy_float)`.",
            }

        cand_forces, cand_energy = cand_out
        cand_forces = np.asarray(cand_forces, dtype=np.float64)
        cand_energy = float(cand_energy)

        if cand_forces.shape != ref_forces.shape:
            return {
                "gate_passed": False,
                "stage_reached": "stage_2_physics_gate",
                "violation_label": "invalid_force_shape",
                "violation_text": f"Expected forces shape {ref_forces.shape}, got {cand_forces.shape}.",
            }
        if not np.all(np.isfinite(cand_forces)) or not math.isfinite(cand_energy):
            return {
                "gate_passed": False,
                "stage_reached": "stage_2_physics_gate",
                "violation_label": "non_finite_numerics",
                "violation_text": "Candidate produced NaN or Inf in forces or potential energy.",
            }

        ref_norm = max(float(np.linalg.norm(ref_forces)), 1e-12)
        force_err = float(np.linalg.norm(cand_forces - ref_forces)) / ref_norm
        energy_err = abs(cand_energy - ref_energy) / max(abs(ref_energy), 1e-12)
        net_force_res = float(np.linalg.norm(np.sum(cand_forces, axis=0))) / ref_norm

        max_force_err = max(max_force_err, force_err)
        max_energy_err = max(max_energy_err, energy_err)
        max_momentum_res = max(max_momentum_res, net_force_res)

        if force_err >= MAX_REL_L2_FORCE_ERROR:
            return {
                "gate_passed": False,
                "stage_reached": "stage_2_physics_gate",
                "rel_l2_force_error": force_err,
                "rel_energy_error": energy_err,
                "violation_label": "physics_force_accuracy_violation",
                "violation_text": (
                    f"Stage 2 Physics Gate failed on {regime} (N={n_part}): "
                    f"rel_l2_force_error={force_err:.3e} >= 1.0e-6. "
                    "Ensure exact minimum-image convention, softened distance s=sqrt(r^2+softening^2), "
                    "and float64 `erfc_vec(alpha * s)`."
                ),
            }
        if energy_err >= MAX_REL_ENERGY_ERROR:
            return {
                "gate_passed": False,
                "stage_reached": "stage_2_physics_gate",
                "rel_l2_force_error": force_err,
                "rel_energy_error": energy_err,
                "violation_label": "physics_energy_accuracy_violation",
                "violation_text": (
                    f"Stage 2 Physics Gate failed on {regime} (N={n_part}): "
                    f"rel_energy_error={energy_err:.3e} >= 1.0e-6. "
                    "Include the potential shift `erfc(alpha * s_cut) / s_cut` and 0.5 pair double-count factor."
                ),
            }
        if net_force_res >= MAX_NET_MOMENTUM_RESIDUAL:
            return {
                "gate_passed": False,
                "stage_reached": "stage_2_physics_gate",
                "rel_l2_force_error": force_err,
                "net_momentum_residual": net_force_res,
                "violation_label": "newton_third_law_violation",
                "violation_text": (
                    f"Newton's 3rd Law momentum conservation failed: net force residual={net_force_res:.3e} >= 1.0e-9."
                ),
            }

    # Symplectic 2-step Velocity-Verlet energy conservation invariant check on first conformation
    symplectic_drift = verify_symplectic_energy_conservation(solver_fn, conformations[0])
    max_symplectic_drift = max(max_symplectic_drift, symplectic_drift)
    if symplectic_drift >= MAX_SYMPLECTIC_ENERGY_DRIFT:
        return {
            "gate_passed": False,
            "stage_reached": "stage_2_physics_gate",
            "symplectic_energy_drift": symplectic_drift,
            "violation_label": "energy_conservation_violation",
            "violation_text": (
                f"Symplectic Velocity-Verlet Hamiltonian drift={symplectic_drift:.3e} >= 1.0e-4."
            ),
        }

    # Stage 3: High-precision throughput benchmark across all split conformations
    total_particles = sum(int(c["positions"].shape[0]) for c in conformations)
    sample_throughputs: list[float] = []
    sample_latencies_ms: list[float] = []

    for _ in range(benchmark_repeats):
        t0 = time.perf_counter()
        for c in conformations:
            solver_fn(
                c["positions"],
                c["charges"],
                float(c["box_length"]),
                float(c["cutoff"]),
                float(c["alpha"]),
                float(c["softening"]),
            )
        elapsed = max(time.perf_counter() - t0, 1e-9)
        sample_throughputs.append(total_particles / elapsed)
        sample_latencies_ms.append(elapsed * 1000.0)

    median_pps = float(statistics.median(sample_throughputs))
    median_ms = float(statistics.median(sample_latencies_ms))

    return {
        "gate_passed": True,
        "stage_reached": "stage_3_throughput_benchmark",
        "particles_per_sec": round(median_pps, 2),
        "median_batch_latency_ms": round(median_ms, 3),
        "total_particles_evaluated": total_particles,
        "rel_l2_force_error": float(f"{max_force_err:.3e}"),
        "rel_energy_error": float(f"{max_energy_err:.3e}"),
        "net_momentum_residual": float(f"{max_momentum_res:.3e}"),
        "symplectic_energy_drift": float(f"{max_symplectic_drift:.3e}"),
    }


def evaluate_program(
    candidate_code: str | None = None,
    split: str = "val",
    scenario_overrides: dict[str, Any] | None = None,
    *,
    require_markers: bool = True,
) -> dict[str, Any]:
    """3-stage AlphaEvolve evaluator for Pillar 2 HPC Ewald Kernel Optimization."""
    canonical_seed = SEED_FILE_PATH.read_text(encoding="utf-8")
    is_seed_eval = False
    if candidate_code is None:
        cand_dir = os.environ.get("_CANDIDATE_DIR", "")
        cand_path = Path(cand_dir) / "main.py" if cand_dir else SEED_FILE_PATH
        candidate_code = cand_path.read_text(encoding="utf-8")
        if not cand_dir:
            is_seed_eval = True

    # Stage 1: AST & EVOLVE-BLOCK safety check (plus strict inner-block import check)
    try:
        spliced_source, block_code = validate_candidate_source(
            candidate_code,
            canonical_seed_source=canonical_seed,
            required_functions=REQUIRED_FUNCTIONS,
            allowed_imports=OUTER_ALLOWED_IMPORTS,
            require_markers=require_markers,
        )
        # Prevent candidate EVOLVE-BLOCK from importing `pillars` reference solver
        validate_ast_safety(block_code, allowed_imports=BLOCK_ALLOWED_IMPORTS)
        mod = execute_candidate_module(spliced_source, "pillar2_candidate")
        solver_fn = mod.__dict__["solve_ewald_forces"]
    except SandboxViolationError as exc:
        return make_evaluation_result(
            primary_metric="particles_per_sec",
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
            primary_metric="particles_per_sec",
            overall_score=HARD_GATE_PENALTY,
            valid=False,
            gate_passed=False,
            stage_reached="stage_1_ast_safety",
            split=split,
            metrics={"error": str(exc)},
            insights=[("module_load_error", f"Failed to load candidate kernel: {exc}")],
        )

    repeats = int((scenario_overrides or {}).get("benchmark_repeats", 3))
    try:
        bench = run_with_timeout(
            lambda: _run_stage2_and_stage3(solver_fn, split, benchmark_repeats=repeats),
            timeout_sec=25.0,
        )
    except CandidateTimeoutError as exc:
        return make_evaluation_result(
            primary_metric="particles_per_sec",
            overall_score=HARD_GATE_PENALTY,
            valid=False,
            gate_passed=False,
            stage_reached="stage_2_physics_gate",
            split=split,
            metrics={"error": str(exc)},
            insights=[("execution_timeout", str(exc))],
        )
    except Exception as exc:
        return make_evaluation_result(
            primary_metric="particles_per_sec",
            overall_score=HARD_GATE_PENALTY,
            valid=False,
            gate_passed=False,
            stage_reached="stage_2_physics_gate",
            split=split,
            metrics={"error": str(exc)},
            insights=[("runtime_exception", f"Kernel raised runtime exception: {exc}")],
        )

    if not bench.get("gate_passed", False):
        return make_evaluation_result(
            primary_metric="particles_per_sec",
            overall_score=HARD_GATE_PENALTY,
            valid=False,
            gate_passed=False,
            stage_reached=str(bench.get("stage_reached", "stage_2_physics_gate")),
            split=split,
            metrics=bench,
            insights=[(str(bench["violation_label"]), str(bench["violation_text"]))],
        )

    # Compute speedup relative to naive O(N^2) seed program on the same split
    if is_seed_eval:
        _BASELINE_THROUGHPUT_CACHE[split] = float(bench["particles_per_sec"])
    elif split not in _BASELINE_THROUGHPUT_CACHE:
        seed_mod = execute_candidate_module(canonical_seed, "pillar2_seed_ref")
        seed_bench = _run_stage2_and_stage3(
            seed_mod.__dict__["solve_ewald_forces"], split, benchmark_repeats=1
        )
        _BASELINE_THROUGHPUT_CACHE[split] = max(float(seed_bench["particles_per_sec"]), 1.0)

    base_pps = _BASELINE_THROUGHPUT_CACHE.get(split, float(bench["particles_per_sec"]))
    speedup = float(bench["particles_per_sec"]) / max(base_pps, 1e-6)
    bench["baseline_particles_per_sec"] = round(base_pps, 2)
    bench["speedup_vs_baseline"] = round(speedup, 2)

    insights: list[tuple[str, str]] = [
        (
            "throughput_summary",
            f"Stage 1-3 PASSED | throughput={bench['particles_per_sec']:.1f} particles/s "
            f"({speedup:.2f}x speedup vs seed) | latency={bench['median_batch_latency_ms']:.2f}ms | "
            f"rel_L2_force_err={bench['rel_l2_force_error']:.2e} | energy_drift={bench['symplectic_energy_drift']:.2e}",
        )
    ]
    if speedup < 10.0:
        insights.append(
            (
                "vectorization_opportunity",
                "Kernel is still bottlenecked by Python loop overhead. Vectorize pair interactions in NumPy "
                "using upper-triangular Newton's 3rd Law symmetry (i < j), spatial 3D cell-list / Verlet neighbor "
                "pruning (`cutoff` << `box_length`), or cache-friendly row tiling with `erfc_vec(alpha * s)`.",
            )
        )
    elif speedup < 80.0:
        insights.append(
            (
                "neighbor_list_tiling_opportunity",
                "Dense O(N^2) NumPy broadcasting computes distances for all N^2 pairs even though cutoff < 0.46*L. "
                "Exploit upper-triangle pair compression (`np.triu_indices` or 3D cell-linked lists) + sparse "
                "`np.add.at` / `np.bincount` force accumulation to cut memory bandwidth and FLOPs by >2x.",
            )
        )

    return make_evaluation_result(
        primary_metric="particles_per_sec",
        overall_score=float(bench["particles_per_sec"]),
        valid=True,
        gate_passed=True,
        stage_reached="stage_3_throughput_benchmark",
        split=split,
        metrics=bench,
        insights=insights,
        secondary_scores={
            "speedup_vs_baseline": round(speedup, 3),
            "neg_latency_ms": -float(bench["median_batch_latency_ms"]),
        },
    )


def benchmark_particle_scaling(
    evolved_code: str,
    sizes: tuple[int, ...] = (512, 1024, 2048, 4096),
) -> list[dict[str, Any]]:
    """Benchmark particle-count scaling (N=512..8192) for the Interactive Lab UI."""
    canonical_seed = SEED_FILE_PATH.read_text(encoding="utf-8")
    spliced_evolved, _ = validate_candidate_source(
        evolved_code,
        canonical_seed_source=canonical_seed,
        required_functions=REQUIRED_FUNCTIONS,
        allowed_imports=OUTER_ALLOWED_IMPORTS,
        require_markers=True,
    )
    evolved_fn = execute_candidate_module(spliced_evolved, "pillar2_scale_ev").__dict__["solve_ewald_forces"]
    seed_fn = execute_candidate_module(canonical_seed, "pillar2_scale_seed").__dict__["solve_ewald_forces"]

    # Calibrate seed O(N^2) throughput at N=384 so N=4096..8192 doesn't block live UI requests
    cal_conf = generate_molecular_conformation(384, 701)
    t0 = time.perf_counter()
    seed_fn(
        cal_conf["positions"],
        cal_conf["charges"],
        cal_conf["box_length"],
        cal_conf["cutoff"],
        cal_conf["alpha"],
        cal_conf["softening"],
    )
    seed_t384 = max(time.perf_counter() - t0, 1e-6)

    rows: list[dict[str, Any]] = []
    for n in sizes:
        conf = generate_molecular_conformation(n, 800 + n)
        t_ev0 = time.perf_counter()
        evolved_fn(
            conf["positions"],
            conf["charges"],
            conf["box_length"],
            conf["cutoff"],
            conf["alpha"],
            conf["softening"],
        )
        ev_sec = max(time.perf_counter() - t_ev0, 1e-6)
        ev_pps = n / ev_sec

        if n <= 512:
            t_s0 = time.perf_counter()
            seed_fn(
                conf["positions"],
                conf["charges"],
                conf["box_length"],
                conf["cutoff"],
                conf["alpha"],
                conf["softening"],
            )
            seed_sec = max(time.perf_counter() - t_s0, 1e-6)
        else:
            # Exact O(N^2) scaling extrapolation from measured N=384 baseline
            seed_sec = seed_t384 * ((n / 384.0) ** 2)
        seed_pps = n / seed_sec

        rows.append(
            {
                "n_particles": n,
                "baseline_latency_ms": round(seed_sec * 1000.0, 2),
                "evolved_latency_ms": round(ev_sec * 1000.0, 2),
                "baseline_particles_per_sec": round(seed_pps, 1),
                "evolved_particles_per_sec": round(ev_pps, 1),
                "speedup_factor": round(ev_pps / max(seed_pps, 1e-6), 2),
            }
        )
    return rows


def load_cuda_simd_kernel_showcase() -> dict[str, str]:
    """Return companion CUDA / AVX-512 kernel template for Cloud Batch GPU/HPC workers."""
    source = CUDA_KERNEL_PATH.read_text(encoding="utf-8") if CUDA_KERNEL_PATH.exists() else ""
    return {
        "filename": "kernels/ewald_cuda_simd.cu",
        "target_architectures": "NVIDIA Hopper/Ada CUDA (SM89/SM90) + x86_64 AVX-512 (Cloud Batch H3/C4D/A3)",
        "source": source,
    }
