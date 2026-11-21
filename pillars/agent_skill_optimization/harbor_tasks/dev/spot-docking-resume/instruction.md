# Preemptible Spot Ligand-Docking Batch Array

Write `/workspace/spot_docking_worker.sh` for fault-tolerant Spot workers evaluating 10,000 ligands into `/output/docked_poses.sdf`.

Cluster constraints:
- Spot instances can be reclaimed with 30-120s notice; script must resume idempotently from existing `/scratch` / bucket checkpoints.
