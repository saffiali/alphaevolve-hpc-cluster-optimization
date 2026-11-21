# 50,000-File Conformer Archive Feature Extraction

Write `/workspace/extract_descriptors.sh` to unpack `/datasets/conformers_50k.tar.gz`, compute 3D descriptors, and export consolidated parquet shards to `gs://hpc-lakehouse/descriptors/` plus `/output/summary.json`.

Cluster constraints:
- Shared `/home` mounts enforce strict metadata IOPS limits and stall when unpacking thousands of small files.
- Route extraction, temporary directories, and caches onto high-IOPS local NVMe `/scratch`.
