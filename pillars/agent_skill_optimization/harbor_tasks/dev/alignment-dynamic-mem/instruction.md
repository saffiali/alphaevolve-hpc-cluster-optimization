# High-Throughput Sequence Index & Alignment Job

Generate `/workspace/run_align.sbatch` processing `/datasets/reads_34gb.bam` into `/output/aligned_sorted.bam`.

Cluster constraints:
- Default partition memory (`8G`) triggers immediate cgroup OOM kills on multi-GB binary inputs.
- Inspect input byte footprint dynamically and size `--mem` with appropriate working-set headroom on a CPU partition (`0` GPUs).
