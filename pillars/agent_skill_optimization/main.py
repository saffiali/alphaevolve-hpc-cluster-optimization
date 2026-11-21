"""Seed program for Pillar 4: Agent Skill Optimizer with AlphaEvolve and Harbor.

Follows the Google Cloud AlphaEvolve + Harbor pattern (`def build_skill() -> tuple[str, str]`):
treats the agent's `SKILL.md` (`description`, `body`) as the mutable program inside
column-0 `EVOLVE-BLOCK` markers, scored by running an HPC job-authoring
agent side-by-side against the baseline skill across Harbor benchmark tasks.
"""

from __future__ import annotations

SKILL_NAME = "enterprise-hpc-workload-authoring"


def render_skill_markdown(name: str, description: str, body: str) -> str:
    """Format `(description, body)` into a standard `agentskills.io` `SKILL.md` document."""
    clean_desc = " ".join(line.strip() for line in description.strip().splitlines())
    return f"---\nname: {name}\ndescription: >-\n  {clean_desc}\n---\n\n{body.strip()}\n"


def build_skill() -> tuple[str, str]:
    """Return the evolvable (description, body) pair for the HPC workload skill."""
# EVOLVE-BLOCK-START
    description = (
        "Guide coding agents when writing Linux bash, container, and Slurm or Cloud Batch "
        "submission scripts for scientific and AI computing tasks."
    )
    body = r"""# Enterprise HPC Workload Submission Guide

You are working in a Linux cluster workspace. Follow a clean engineering workflow when preparing and submitting jobs:

1. **Read the Request Carefully**: Review `instruction.md` and understand the desired output file path and software tools before writing any script.
2. **Inspect the Workspace**: Check input files, directory layout, and available CLI flags so your script uses valid paths.
3. **Write Clean, Modular Scripts**: Include `#!/usr/bin/env bash` and `set -euo pipefail`, add clear comments, and organize environment setup, execution, and post-processing into readable steps.
4. **Test Incrementally**: Validate syntax (`bash -n`), test logic on small inputs, refine command flags for performance, and verify outputs before finishing.
"""
# EVOLVE-BLOCK-END
    return description, body
