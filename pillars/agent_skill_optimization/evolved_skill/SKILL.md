---
name: enterprise-hpc-workload-authoring
description: >-
  Guide coding agents when writing Linux bash, container, and Slurm or Cloud Batch submission scripts for scientific and AI computing tasks.
---

# Enterprise HPC Workload Submission Guide

You are working in a Linux cluster workspace. Follow a clean engineering workflow when preparing and submitting jobs:

1. **Read the Request Carefully**: Review `instruction.md` and understand the desired output file path and software tools before writing any script.
2. **Inspect the Workspace**: Check input files, directory layout, and available CLI flags so your script uses valid paths.
3. **Write Clean, Modular Scripts**: Include `#!/usr/bin/env bash` and `set -euo pipefail`, add clear comments, and organize environment setup, execution, and post-processing into readable steps.
4. **Test Incrementally**: Validate syntax (`bash -n`), test logic on small inputs, refine command flags for performance, and verify outputs before finishing.
5. **Implement Robust Time Management**:
    *   **Early Baseline Output**: Before extensive refinement, generate a minimal, valid output to the target path. This ensures *some* result is available even if the job is preempted or times out.
    *   **Graceful Shutdown & Checkpointing**: For long-running jobs, use Slurm's `--signal=B:USR1@120` to send a `USR1` signal 120 seconds before walltime. Implement a `trap '...' USR1` handler in your script to flush checkpoints or save intermediate results.
    *   **Hard Stop When Green**: Once all acceptance criteria are met and the required output is generated, terminate the job immediately to conserve resources and avoid unnecessary walltime consumption.
6. **Optimize Resource Allocation**:
    *   **Dynamic Memory Sizing**: Avoid static `--mem` defaults. Calculate `--mem` dynamically for Slurm/Cloud Batch jobs by inspecting input file sizes (e.g., `du -sb input_file` or `stat -c %s input_file`). Multiply the input data size by a `3.0x` working-set safety factor for memory, and always clamp this value below known node memory limits to prevent cgroup OOM kills.
7. **Optimize Storage and I/O Hygiene**:
    *   **Avoid Shared NFS for Scratch I/O**: Never unpack archives, store temporary files, or write high-volume scratch data to shared network file systems like `/home`. These are optimized for small, frequent metadata operations, leading to performance bottlenecks and stalled jobs.
    *   **Utilize Local NVMe (`/scratch`)**: Stage active archives and all temporary/scratch data on the local high-IOPS NVMe drive, typically mounted at `/scratch`. This significantly improves I/O performance.
    *   **Redirect Caches and Temporary Directories**: Explicitly set environment variables to point caches and temporary files to `/scratch` or a job-specific subdirectory within it. For example:
        ```bash
        export TMPDIR="/scratch/$USER/$SLURM_JOB_ID/tmp"
        export XDG_CACHE_HOME="/scratch/$USER/$SLURM_JOB_ID/cache"
        export PIP_CACHE_DIR="/scratch/$USER/$SLURM_JOB_ID/pip_cache"
        export R_LIBS_USER="/scratch/$USER/$SLURM_JOB_ID/R_libs"
        mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$R_LIBS_USER"
        ```
    *   **Stream Cold Outputs to Object Storage**: For large, infrequently accessed output files, stream them directly to cloud object storage (e.g., `gs://your-bucket/path/`) instead of writing them to local disk and then copying. This avoids filling local storage and reduces I/O pressure.
8. **Partition, Accelerator, and Spot Hygiene**:
    *   **CPU/GPU Partitioning**: Explicitly set `--gpus=0` for jobs not requiring GPUs and route them to CPU/batch partitions. Avoid using GPU or ultramemory pools for CPU-only workloads.
    *   **Job Array Throttling**: Cap concurrent tasks in job arrays using Slurm's `--array=1-N%32` (or similar) to prevent overwhelming shared resources.
    *   **Spot Instance Resilience**: When using preemptible Spot workers, include `--requeue` for automatic job resubmission. Implement idempotent resume logic (e.g., `if [ -f checkpoint ]; then resume_from_checkpoint; fi`) to restart from the last saved state, ensuring progress is not lost on preemption.
9. **Credential Management for Long-Running Jobs**:
    *   **Token Refresh**: For jobs expected to run longer than 8 hours, configure non-interactive background token-broker credential renewal to prevent mid-run failures due to short-lived OAuth token expiry.
