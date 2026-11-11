// Copyright 2026 Google LLC
// Licensed under the Apache License, Version 2.0
//
// Companion CUDA + C++17 AVX-512 Kernel Template for AlphaEvolve Cloud Batch Workers
// Targets GPU Evaluator VMs (`a3-highgpu-1g`, `g2-standard-8`) and Compute-Optimized
// HPC Evaluator VMs (`h3-standard-88`, `c4d-standard-32`).

#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

struct alignas(32) ParticleSoA {
  const double* __restrict__ px;
  const double* __restrict__ py;
  const double* __restrict__ pz;
  const double* __restrict__ q;
  double* __restrict__ fx;
  double* __restrict__ fy;
  double* __restrict__ fz;
};

#ifdef __CUDACC__
// EVOLVE-BLOCK-START
constexpr int kTileDim = 128;
constexpr int kUnrollFactor = 4;

__global__ void __launch_bounds__(kTileDim, 4) ewald_short_range_tiled_kernel(
    const double4* __restrict__ pos_charge,
    double3* __restrict__ out_forces,
    const int n_particles,
    const double box_length,
    const double inv_box,
    const double rcut2,
    const double soft2,
    const double alpha,
    const double two_alpha_over_sqrt_pi) {
  __shared__ double4 sh_tile[kTileDim];

  const int gid = blockIdx.x * kTileDim + threadIdx.x;
  const bool active = (gid < n_particles);

  double4 my_pq = active ? pos_charge[gid] : make_double4(0.0, 0.0, 0.0, 0.0);
  double acc_fx = 0.0;
  double acc_fy = 0.0;
  double acc_fz = 0.0;

  const int num_tiles = (n_particles + kTileDim - 1) / kTileDim;
  for (int tile = 0; tile < num_tiles; ++tile) {
    const int src_idx = tile * kTileDim + threadIdx.x;
    sh_tile[threadIdx.x] =
        (src_idx < n_particles) ? pos_charge[src_idx] : make_double4(0.0, 0.0, 0.0, 0.0);
    __syncthreads();

    if (active) {
      const int tile_limit = min(kTileDim, n_particles - tile * kTileDim);
      #pragma unroll kUnrollFactor
      for (int j = 0; j < tile_limit; ++j) {
        const int global_j = tile * kTileDim + j;
        if (global_j == gid) continue;

        double dx = sh_tile[j].x - my_pq.x;
        double dy = sh_tile[j].y - my_pq.y;
        double dz = sh_tile[j].z - my_pq.z;

        dx -= box_length * nearbyint(dx * inv_box);
        dy -= box_length * nearbyint(dy * inv_box);
        dz -= box_length * nearbyint(dz * inv_box);

        const double r2 = fma(dx, dx, fma(dy, dy, dz * dz));
        if (r2 < rcut2) {
          const double s2 = r2 + soft2;
          const double inv_s = rsqrt(s2);
          const double s = s2 * inv_s;
          const double as = alpha * s;
          const double erfc_val = erfc(as);
          const double gauss_val = two_alpha_over_sqrt_pi * s * exp(-as * as);
          const double pref = -(my_pq.w * sh_tile[j].w) * (erfc_val + gauss_val) *
                              (inv_s * inv_s * inv_s);

          acc_fx = fma(pref, dx, acc_fx);
          acc_fy = fma(pref, dy, acc_fy);
          acc_fz = fma(pref, dz, acc_fz);
        }
      }
    }
    __syncthreads();
  }

  if (active) {
    out_forces[gid] = make_double3(acc_fx, acc_fy, acc_fz);
  }
}
// EVOLVE-BLOCK-END
#endif

// Host AVX-512 SIMD tile reference for CPU-optimized HPC Batch nodes (C4D / H3)
void ewald_avx512_tile_host(
    const ParticleSoA& state,
    int start_idx,
    int end_idx,
    int n_particles,
    double box_length,
    double cutoff,
    double alpha,
    double softening) {
  const double inv_box = 1.0 / box_length;
  const double rcut2 = cutoff * cutoff;
  const double soft2 = softening * softening;
  const double kTwoAlphaSqrtPi = 2.0 * alpha / 1.77245385090551602729;

  for (int i = start_idx; i < end_idx; ++i) {
    double fxi = 0.0, fyi = 0.0, fzi = 0.0;
    const double xi = state.px[i];
    const double yi = state.py[i];
    const double zi = state.pz[i];
    const double qi = state.q[i];

    for (int j = 0; j < n_particles; ++j) {
      if (i == j) continue;
      double dx = state.px[j] - xi;
      double dy = state.py[j] - yi;
      double dz = state.pz[j] - zi;
      dx -= box_length * std::nearbyint(dx * inv_box);
      dy -= box_length * std::nearbyint(dy * inv_box);
      dz -= box_length * std::nearbyint(dz * inv_box);

      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 < rcut2) {
        const double s = std::sqrt(r2 + soft2);
        const double inv_s3 = 1.0 / (s * s * s);
        const double pref = -qi * state.q[j] *
                            (std::erfc(alpha * s) + kTwoAlphaSqrtPi * s * std::exp(-alpha * alpha * s * s)) *
                            inv_s3;
        fxi += pref * dx;
        fyi += pref * dy;
        fzi += pref * dz;
      }
    }
    state.fx[i] = fxi;
    state.fy[i] = fyi;
    state.fz[i] = fzi;
  }
}
