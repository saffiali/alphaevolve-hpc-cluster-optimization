"""Ground-truth IEEE-754 reference physics & molecular conformation suite for Pillar 2.

Provides:
  - `erfc_vec`: Double-precision minimax rational `erfc(x)` implementation in pure NumPy
    (Cody 1969 algorithm, accurate to < 1e-15 relative error across all x >= 0).
  - `generate_molecular_conformation`: Deterministic 3D periodic molecular systems across
    train, validation, and held-out test splits.
  - `reference_ewald_forces`: Exact float64 reference short-range Ewald force-field solver.
  - `verify_symplectic_energy_conservation`: Velocity-Verlet Hamiltonian conservation gate.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np


def erfc_vec(x: np.ndarray) -> np.ndarray:
    """IEEE-754 float64 vectorized complementary error function `erfc(x)` in pure NumPy.

    Uses W. J. Cody (1969), Mathematics of Computation 23(107), 631-637 piecewise
    minimax rational approximations across |x| <= 0.46875, 0.46875 < |x| <= 4.0,
    and |x| > 4.0, achieving < 5e-16 relative error vs `math.erfc` without SciPy.
    """
    arr = np.asarray(x, dtype=np.float64)
    ax = np.abs(arr)
    out = np.empty_like(ax, dtype=np.float64)

    # Region 1: |x| <= 0.46875 -> erfc(x) = 1 - erf(x)
    m1 = ax <= 0.46875
    if np.any(m1):
        z = ax[m1]
        z2 = z * z
        a = (
            3.16112374387056560e00,
            1.13864154151050156e02,
            3.77485237685302021e02,
            3.20937758913846947e03,
            1.85777706184603153e-01,
        )
        b = (
            2.36012909523441209e01,
            2.44024637934444173e02,
            1.28261652607737228e03,
            2.84423683343917062e03,
        )
        num = (((a[4] * z2 + a[0]) * z2 + a[1]) * z2 + a[2]) * z2 + a[3]
        den = (((z2 + b[0]) * z2 + b[1]) * z2 + b[2]) * z2 + b[3]
        erf_val = z * (num / den)
        out[m1] = 1.0 - erf_val

    # Region 2: 0.46875 < |x| <= 4.0
    m2 = (~m1) & (ax <= 4.0)
    if np.any(m2):
        z = ax[m2]
        c = (
            5.64188496988670089e-01,
            8.88314979438837594e00,
            6.61191906371416295e01,
            2.98635138197400131e02,
            8.81952221241769090e02,
            1.71204761263407058e03,
            2.05107837782607147e03,
            1.23033935479799725e03,
            2.15311535474403846e-08,
        )
        d = (
            1.57449261107098347e01,
            1.17693950891312499e02,
            5.37181101862009858e02,
            1.62138957456669019e03,
            3.29079923573345963e03,
            4.36261909014324716e03,
            3.43936767414372164e03,
            1.23033935480374942e03,
        )
        num = c[8]
        for coeff in c[:8]:
            num = num * z + coeff
        den = 1.0
        for coeff in d:
            den = den * z + coeff
        z_int = np.floor(z * 16.0) / 16.0
        del_z = z - z_int
        out[m2] = np.exp(-z_int * z_int) * np.exp(-(z + z_int) * del_z) * (num / den)

    # Region 3: |x| > 4.0 (rare tail beyond screening cutoff; exact IEEE-754 math.erfc)
    m3 = ax > 4.0
    if np.any(m3):
        out[m3] = np.fromiter((math.erfc(float(v)) for v in ax[m3]), dtype=np.float64, count=int(np.sum(m3)))

    return np.where(arr < 0.0, 2.0 - out, out)


def generate_molecular_conformation(
    n_particles: int,
    seed: int,
    *,
    density: float = 0.28,
    regime: str = "solvated_biomolecule",
) -> dict[str, Any]:
    """Generate a deterministic 3D periodic molecular conformation with neutral net charge."""
    rng = np.random.default_rng(seed)
    box_length = float((n_particles / density) ** (1.0 / 3.0))

    if regime == "ionic_lattice":
        # Perturbed cubic lattice + thermal displacement
        side = max(2, int(math.ceil(n_particles ** (1.0 / 3.0))))
        grid = np.linspace(0.5, box_length - 0.5, side, endpoint=True)
        gx, gy, gz = np.meshgrid(grid, grid, grid, indexing="ij")
        base_pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])[:n_particles]
        positions = np.mod(base_pts + rng.normal(0.0, 0.14, size=(n_particles, 3)), box_length)
    elif regime == "clustered_droplet":
        # Two dense macromolecular clusters in solvent
        c1 = np.array([0.32 * box_length, 0.35 * box_length, 0.35 * box_length])
        c2 = np.array([0.68 * box_length, 0.65 * box_length, 0.65 * box_length])
        n_half = n_particles // 2
        p1 = c1 + rng.normal(0.0, 0.16 * box_length, size=(n_half, 3))
        p2 = c2 + rng.normal(0.0, 0.18 * box_length, size=(n_particles - n_half, 3))
        positions = np.mod(np.vstack([p1, p2]), box_length)
    else:
        # Uniform liquid / solvated biomolecular box with soft steric separation
        positions = rng.uniform(0.0, box_length, size=(n_particles, 3))

    # Partial atomic charges in [-0.85, +0.85] e, strictly neutralized
    raw_charges = rng.choice([-0.82, -0.41, 0.25, 0.41, 0.82], size=n_particles)
    raw_charges = raw_charges.astype(np.float64) - float(np.mean(raw_charges))

    cutoff = min(4.2, 0.46 * box_length)
    alpha = 0.55
    softening = 0.08

    return {
        "positions": np.ascontiguousarray(positions, dtype=np.float64),
        "charges": np.ascontiguousarray(raw_charges, dtype=np.float64),
        "box_length": box_length,
        "cutoff": cutoff,
        "alpha": alpha,
        "softening": softening,
        "regime": regime,
    }


def reference_ewald_forces(
    positions: np.ndarray,
    charges: np.ndarray,
    box_length: float,
    cutoff: float,
    alpha: float,
    softening: float = 0.08,
) -> tuple[np.ndarray, float]:
    """Exact IEEE-754 float64 ground-truth short-range shifted Ewald solver.

    Uses potential-shifted real-space Ewald interaction so V(r_cut) = 0 continuously:
        V_pair(r) = q_i * q_j * (erfc(alpha * s) / s - erfc(alpha * s_cut) / s_cut)
        s = sqrt(r^2 + softening^2), for 0 < r < cutoff.
    """
    pos = np.asarray(positions, dtype=np.float64)
    q = np.asarray(charges, dtype=np.float64)
    n = pos.shape[0]
    two_alpha_over_sqrt_pi = 2.0 * alpha / math.sqrt(math.pi)
    s_cut = math.sqrt(cutoff * cutoff + softening * softening)
    shift_val = math.erfc(alpha * s_cut) / s_cut
    rcut2 = cutoff * cutoff
    soft2 = softening * softening

    # Tile in chunks of 256 so ground-truth reference stays fast and O(1) memory even for large N
    forces = np.zeros((n, 3), dtype=np.float64)
    total_energy = 0.0
    chunk = 256

    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        # disp[a, j, d] = pos[j, d] - pos[i0+a, d]
        disp = pos[None, :, :] - pos[i0:i1, None, :]
        disp -= box_length * np.round(disp / box_length)
        r2 = np.sum(disp * disp, axis=-1)

        # Exclude self-interactions
        self_rows = np.arange(i1 - i0)
        r2[self_rows, i0 + self_rows] = rcut2 + 1.0

        mask = r2 < rcut2
        if not np.any(mask):
            continue

        s = np.sqrt(r2[mask] + soft2)
        erfc_s = erfc_vec(alpha * s)
        qi_qj = (q[i0:i1, None] * q[None, :])[mask]

        pair_pot = qi_qj * (erfc_s / s - shift_val)
        total_energy += 0.5 * float(np.sum(pair_pot))

        f_mag = np.zeros_like(r2, dtype=np.float64)
        f_mag[mask] = -qi_qj * (
            erfc_s + two_alpha_over_sqrt_pi * s * np.exp(-(alpha * alpha) * (s * s))
        ) / (s * s * s)

        forces[i0:i1] += np.sum(f_mag[:, :, None] * disp, axis=1)

    return forces, total_energy


def verify_symplectic_energy_conservation(
    solver_fn: Any,
    conformation: dict[str, Any],
    dt: float = 5e-4,
    steps: int = 2,
) -> float:
    """Advance molecular coordinates with Velocity-Verlet and return relative Hamiltonian drift."""
    pos = conformation["positions"].copy()
    q = conformation["charges"]
    box_l = conformation["box_length"]
    rcut = conformation["cutoff"]
    alpha = conformation["alpha"]
    soft = conformation["softening"]

    rng = np.random.default_rng(12345)
    vel = rng.normal(0.0, 0.25, size=pos.shape)
    vel -= np.mean(vel, axis=0, keepdims=True)

    forces, pot0 = solver_fn(pos, q, box_l, rcut, alpha, soft)
    kin0 = 0.5 * float(np.sum(vel * vel))
    h0 = kin0 + pot0

    for _ in range(steps):
        vel_half = vel + 0.5 * dt * forces
        pos = np.mod(pos + dt * vel_half, box_l)
        forces, pot_now = solver_fn(pos, q, box_l, rcut, alpha, soft)
        vel = vel_half + 0.5 * dt * forces

    kin_end = 0.5 * float(np.sum(vel * vel))
    h_end = kin_end + pot_now
    return abs(h_end - h0) / max(abs(h0), 1e-8)
