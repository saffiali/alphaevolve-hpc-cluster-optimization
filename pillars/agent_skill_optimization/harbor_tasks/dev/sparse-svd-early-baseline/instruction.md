# High-Memory Sparse Matrix Factorization Pipeline

Author `/workspace/run_svd.sbatch` computing top-128 singular vectors of `/datasets/expression_48gb.npz` into `/output/factors.npy`.

Cluster constraints:
- Size memory dynamically from matrix file size, write a fast randomized-SVD baseline to `/output/factors.npy` first, and hard-stop once residual < 1e-4.
