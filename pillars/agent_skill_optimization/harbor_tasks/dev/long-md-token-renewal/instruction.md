# 18-Hour Solvated Trajectory Simulation With Cloud Sync

Create `/workspace/run_trajectory_18h.sbatch` streaming trajectory frames every 30 minutes to `gs://hpc-trajectories/run01/` and writing `/output/final_state.chk`.

Cluster constraints:
- Short-lived federated OAuth credentials expire after 8 hours unless non-interactive token-broker renewal is configured.
