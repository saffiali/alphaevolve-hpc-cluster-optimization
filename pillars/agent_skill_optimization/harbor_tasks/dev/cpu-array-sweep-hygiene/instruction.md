# 200-Shard Parametric CPU Simulation Sweep

Create `/workspace/sweep_array.sbatch` executing 200 independent 4-vCPU parameter shards writing to `/output/shard_${SLURM_ARRAY_TASK_ID}.csv`.

Cluster constraints:
- Never place small 4-vCPU CPU-only tasks onto scarce GPU or ultramem partitions.
- Enforce bounded job-array throttling (`%32` max concurrent tasks) and explicit `--gpus=0`.
