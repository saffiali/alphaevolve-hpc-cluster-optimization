"""Seed program for Pillar 2: 3D Short-Range Shifted Ewald Molecular Force-Field Solver.

Optimizes pairwise electrostatic + softened force and potential energy computation
under periodic boundary conditions (Minimum Image Convention).

The vectorized IEEE-754 double-precision helper `erfc_vec(x)` is pre-imported and
available in module scope for candidate kernels inside `# EVOLVE-BLOCK`.
"""

from __future__ import annotations

import math
import numpy as np

from pillars.hpc_code_optimization.physics_reference import erfc_vec  # noqa: F401


# EVOLVE-BLOCK-START
def solve_ewald_forces(
    positions: np.ndarray,
    charges: np.ndarray,
    box_length: float,
    cutoff: float,
    alpha: float,
    softening: float = 0.08,
) -> tuple[np.ndarray, float]:
    """Compute 3D periodic shifted-Ewald atomic forces `(N, 3)` and total potential energy.

    Exact physics equations required by Stage 2 accuracy gate (rel L2 error < 1e-6):
      For all pairs i != j with minimum-image displacement `d = pos[j] - pos[i] - L * round((pos[j] - pos[i]) / L)`
      satisfying `r^2 = dot(d, d) < cutoff^2`:
        s = sqrt(r^2 + softening^2)
        s_cut = sqrt(cutoff^2 + softening^2)
        shift_val = erfc(alpha * s_cut) / s_cut
        V_ij = q_i * q_j * (erfc(alpha * s) / s - shift_val)
        total_energy = 0.5 * sum_{i != j} V_ij
        F_{i <- j} = -q_i * q_j * (erfc(alpha * s) + (2 * alpha / sqrt(pi)) * s * exp(-(alpha * s)^2)) / s^3 * d
    """
    n_particles = int(positions.shape[0])
    forces = np.zeros((n_particles, 3), dtype=np.float64)
    total_energy = 0.0

    rcut2 = cutoff * cutoff
    soft2 = softening * softening
    s_cut = math.sqrt(rcut2 + soft2)
    shift_val = math.erfc(alpha * s_cut) / s_cut
    two_alpha_over_sqrt_pi = 2.0 * alpha / math.sqrt(math.pi)
    inv_box = 1.0 / box_length

    # Naive O(N^2) un-tiled scalar nested loop (baseline seed program)
    for i in range(n_particles):
        xi = float(positions[i, 0])
        yi = float(positions[i, 1])
        zi = float(positions[i, 2])
        qi = float(charges[i])
        fxi = 0.0
        fyi = 0.0
        fzi = 0.0

        for j in range(n_particles):
            if i == j:
                continue
            dx = float(positions[j, 0]) - xi
            dy = float(positions[j, 1]) - yi
            dz = float(positions[j, 2]) - zi

            dx -= box_length * round(dx * inv_box)
            dy -= box_length * round(dy * inv_box)
            dz -= box_length * round(dz * inv_box)

            r2 = dx * dx + dy * dy + dz * dz
            if r2 < rcut2:
                s = math.sqrt(r2 + soft2)
                erfc_s = math.erfc(alpha * s)
                qij = qi * float(charges[j])
                total_energy += 0.5 * qij * (erfc_s / s - shift_val)

                pref = -qij * (
                    erfc_s + two_alpha_over_sqrt_pi * s * math.exp(-(alpha * s) * (alpha * s))
                ) / (s * s * s)
                fxi += pref * dx
                fyi += pref * dy
                fzi += pref * dz

        forces[i, 0] = fxi
        forces[i, 1] = fyi
        forces[i, 2] = fzi

    return forces, total_energy
# EVOLVE-BLOCK-END
