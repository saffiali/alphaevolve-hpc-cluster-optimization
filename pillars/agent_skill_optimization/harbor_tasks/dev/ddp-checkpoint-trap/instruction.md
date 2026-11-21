# Distributed 4-GPU PyTorch Training Under Tight Walltime

Author `/workspace/submit_ddp.sbatch` to launch a 4-GPU `torchrun` training run on `/datasets/conformers_18gb.tar` writing validated weights to `/output/model_final.pt` and `/output/metrics.json`.

Cluster constraints:
- Strict 4-hour scheduler walltime and tight 180s authoring timeout.
- Pre-timeout preemption/walltime signals must flush state cleanly without corrupting `/output/model_final.pt`, and a valid baseline checkpoint must land at `/output/model_final.pt` early before extended epoch tuning.
