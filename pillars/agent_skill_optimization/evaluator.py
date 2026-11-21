"""Harbor Paired-Baseline Evaluator & Anti-Reward-Hacking Gatekeeper for Pillar 4.

Implements the Google Cloud AlphaEvolve + Harbor Agent Skill Optimizer pattern:
  - Tier 1 (`stage_1_skill_integrity`): AST sandbox, `agentskills.io` frontmatter & length
    bounds (`<= 8,000` chars), anti-prompt-injection / verifier-tampering checks, and
    strict zero-tolerance check against Harbor task-ID or filename answer-key hardcoding
    (returns `HARD_GATE_PENALTY = -1e12` on any violation).
  - Tier 2 & 3 (`stage_3_harbor_paired_eval`): Evaluates an HPC job-authoring coding agent
    conditioned on the candidate `(description, body)` skill *concurrently side-by-side*
    with the baseline skill under shared model-latency waves across Harbor v1.1 tasks
    (`8` `dev` tasks during evolution; `4` strictly unseen `holdout` tasks for final test).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
import statistics
import tomllib
from typing import Any

import numpy as np

from pillars.common.models import HARD_GATE_PENALTY, make_evaluation_result
from pillars.common.sandbox import (
    CandidateTimeoutError,
    SandboxViolationError,
    execute_candidate_module,
    run_with_timeout,
    validate_candidate_source,
)

PILLAR_DIR = Path(__file__).resolve().parent
SEED_FILE_PATH = PILLAR_DIR / "main.py"
HARBOR_TASKS_DIR = PILLAR_DIR / "harbor_tasks"

ALLOWED_IMPORTS = frozenset({"__future__", "typing"})
REQUIRED_FUNCTIONS = ("build_skill",)

MAX_SKILL_TOTAL_CHARS = 8000
MIN_DESCRIPTION_CHARS = 20
MAX_DESCRIPTION_CHARS = 500
MIN_BODY_CHARS = 120

PROMPT_INJECTION_PATTERNS: tuple[str, ...] = (
    r"ignore\s+(?:all\s+|any\s+)?previous\s+instructions",
    r"disregard\s+(?:all\s+|any\s+)?prior\s+instructions",
    r"override\s+(?:the\s+)?verifier",
    r"patch\s+(?:the\s+)?verifier",
    r"modify\s+(?:the\s+)?verifier",
    r"tamper\s+with\s+(?:the\s+)?verifier",
    r"bypass\s+(?:the\s+)?(?:verifier|validation|check)",
    r"chmod\s+777\s+/verifier",
    r"rm\s+-rf\s+/verifier",
    r"echo\s+['\"]?pass['\"]?\s*>\s*/verifier",
    r"reward_hack",
    r"fake_pass",
    r"system\s*override\s*:\s*grant\s*full\s*score",
)


def _normalize_split(split: str) -> str:
    if split in ("train", "val", "dev"):
        return "dev"
    if split in ("test", "holdout"):
        return "holdout"
    if split == "all":
        return "all"
    raise ValueError(f"Unknown split '{split}'. Expected dev/train/val, holdout/test, or all.")


def load_harbor_tasks(split: str = "dev") -> list[dict[str, Any]]:
    """Load Harbor v1.1 `task.toml` + `instruction.md` specifications for a split."""
    norm = _normalize_split(split)
    subfolders = ("dev", "holdout") if norm == "all" else (norm,)

    tasks: list[dict[str, Any]] = []
    for sub in subfolders:
        split_dir = HARBOR_TASKS_DIR / sub
        if not split_dir.exists():
            continue
        for tdir in sorted(split_dir.iterdir()):
            toml_path = tdir / "task.toml"
            inst_path = tdir / "instruction.md"
            if not toml_path.exists() or not inst_path.exists():
                continue
            cfg = tomllib.loads(toml_path.read_text(encoding="utf-8"))
            inst = inst_path.read_text(encoding="utf-8")
            checks = cfg.get("verifier", {}).get("checks", {})
            tasks.append(
                {
                    "slug": tdir.name,
                    "split": sub,
                    "schema_version": cfg["schema_version"],
                    "task_name": cfg["task"]["name"],
                    "difficulty": cfg["metadata"]["difficulty"],
                    "category": cfg["metadata"]["category"],
                    "agent_timeout_sec": float(cfg["agent"]["timeout_sec"]),
                    "verifier_timeout_sec": float(cfg["verifier"]["timeout_sec"]),
                    "cpus": int(cfg["environment"]["cpus"]),
                    "memory_mb": int(cfg["environment"]["memory_mb"]),
                    "gpus": int(cfg["environment"]["gpus"]),
                    "dataset_gb": float(checks["dataset_gb"]),
                    "walltime_hours": float(checks["walltime_hours"]),
                    "spot_mode": bool(checks["spot_mode"]),
                    "required_protocols": list(checks["required_protocols"]),
                    "instruction": inst,
                }
            )
    return tasks


def _collect_forbidden_answer_key_tokens() -> tuple[str, ...]:
    """Gather all dev & holdout task slugs and benchmark-specific dataset filenames."""
    all_tasks = load_harbor_tasks("all")
    tokens: set[str] = set()
    for t in all_tasks:
        tokens.add(t["slug"].lower())
        # Extract specific `/datasets/...` or `/workspace/...` filenames from instruction.md
        for m in re.findall(r"/(?:datasets|workspace)/([a-zA-Z0-9_.-]+)", t["instruction"]):
            tokens.add(m.lower())
    return tuple(sorted(tokens))


def validate_skill_tier1(description: Any, body: Any) -> tuple[bool, str, str]:
    """Tier 1 structural, budget, anti-injection, and anti-answer-key validation."""
    if not isinstance(description, str) or not isinstance(body, str):
        return (
            False,
            "invalid_return_types",
            "build_skill() must return a 2-tuple of strings `(description: str, body: str)`.",
        )

    desc_s = description.strip()
    body_s = body.strip()
    total_chars = len(desc_s) + len(body_s)

    if "---" in desc_s:
        return (
            False,
            "frontmatter_delimiter_injection",
            "Skill `description` must not contain raw `---` YAML frontmatter delimiters.",
        )

    combined_lower = f"{desc_s}\n{body_s}".lower()

    for pat in PROMPT_INJECTION_PATTERNS:
        if re.search(pat, combined_lower):
            return (
                False,
                "prompt_injection_or_verifier_tampering",
                f"Skill rejected by Tier 1 safety gate: matched adversarial pattern `{pat}`.",
            )

    for forbidden_tok in _collect_forbidden_answer_key_tokens():
        if forbidden_tok in combined_lower:
            return (
                False,
                "answer_key_hardcoding_violation",
                f"Skill rejected by Tier 1 anti-memorization gate: contains benchmark-specific identifier '{forbidden_tok}'. "
                "Evolve generalizable HPC engineering rules, never task-specific filenames or slugs.",
            )

    if len(desc_s) < MIN_DESCRIPTION_CHARS or len(desc_s) > MAX_DESCRIPTION_CHARS:
        return (
            False,
            "description_length_violation",
            f"Skill `description` length ({len(desc_s)} chars) must be between {MIN_DESCRIPTION_CHARS} and {MAX_DESCRIPTION_CHARS} chars.",
        )
    if len(body_s) < MIN_BODY_CHARS:
        return (
            False,
            "body_too_short",
            f"Skill `body` length ({len(body_s)} chars) must be >= {MIN_BODY_CHARS} chars.",
        )
    if total_chars > MAX_SKILL_TOTAL_CHARS:
        return (
            False,
            "skill_token_budget_exceeded",
            f"Total skill size ({total_chars} chars) exceeds hard {MAX_SKILL_TOTAL_CHARS}-char context budget.",
        )
    if not re.search(r"^#{1,3}\s+\S+", body_s, flags=re.MULTILINE):
        return (
            False,
            "missing_markdown_structure",
            "Skill `body` must be structured Markdown containing at least one `#` section heading.",
        )

    words = re.findall(r"[a-zA-Z]{3,}", combined_lower)
    if len(words) > 60:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.22:
            return (
                False,
                "degenerate_keyword_stuffing",
                f"Skill rejected due to unnatural token repetition (unique ratio={unique_ratio:.2f} < 0.22).",
            )

    return True, "", ""


def extract_skill_protocol_strengths(description: str, body: str) -> dict[str, float]:
    """Parse structured operational protocols from a `SKILL.md` into `[0.0 .. 1.0]` strengths."""
    txt = f"{description}\n{body}".lower()

    # 1. Timeout Guard: Early Baseline + Pre-Timeout Signal Trap + Hard Stop When Green
    c_early_base = bool(
        re.search(r"(?:early|first|initial|immediately|before\s+refin)", txt)
        and re.search(r"(?:baseline|working\s+artifact|target\s+path|valid\s+output|minimal\s+working)", txt)
    )
    c_sig_trap = bool(
        re.search(r"(?:--signal|sigusr1|usr1|sigterm|trap\s+)", txt)
        and re.search(r"(?:checkpoint|flush|save|state)", txt)
    )
    c_hard_stop = bool(
        re.search(r"(?:hard\s+stop|stop\s+immediately|exit\s+immediately|do\s+not\s+over-?polish|halt\s+once)", txt)
        and re.search(r"(?:green|pass|verif|accept|succeed)", txt)
    )
    p_timeout = (float(c_early_base) + float(c_sig_trap) + float(c_hard_stop)) / 3.0

    # 2. Input-Size-Driven Dynamic Memory Right-Sizing
    c_stat_du = bool(re.search(r"(?:\bdu\b|\bstat\b|input\s+size|file\s+size|byte\s+size|dataset\s+size|footprint)", txt))
    c_mem_mult = bool(
        re.search(r"(?:--mem|memory)", txt)
        and re.search(r"(?:[234](?:\.\d+)?\s*x|multiplier|headroom|safety\s+factor|scale|proportional)", txt)
    )
    c_oom_ceil = bool(re.search(r"(?:oom|out-of-memory|cgroup|ceiling|clamp|chunk|stream)", txt))
    p_memory = (float(c_stat_du) + float(c_mem_mult) + float(c_oom_ceil)) / 3.0

    # 3. Tiered Storage Hygiene: Local NVMe `/scratch` + Cache Redirects + Object Storage
    c_scratch = bool(
        re.search(r"(?:/scratch|\$scratch|nvme|local\s+scratch)", txt)
        and re.search(r"(?:/home|nfs|metadata|inode)", txt)
    )
    c_caches = bool(
        re.search(r"(?:tmpdir|xdg_cache_home|pip_cache_dir|r_libs_user|cache)", txt)
    )
    c_obj_store = bool(re.search(r"(?:gs://|object\s+storage|bucket|gcloud\s+storage|gsutil)", txt))
    p_storage = (float(c_scratch) + float(c_caches) + float(c_obj_store)) / 3.0

    # 4. Partition, Accelerator & Spot Array Hygiene
    c_cpu_part = bool(
        re.search(r"(?:--gpus=0|gpus\s*=\s*0|non-gpu|cpu-only)", txt)
        and re.search(r"(?:partition|gpu\s+pool|gpu\s+node|ultramem|megamem)", txt)
    )
    c_array_cap = bool(re.search(r"(?:--array|%\d+|concurrency|throttl)", txt))
    c_spot_resume = bool(
        re.search(r"(?:--requeue|idempotent|resume|preempt|spot)", txt)
        and re.search(r"(?:checkpoint|exist|-f\s+)", txt)
    )
    p_partition = (float(c_cpu_part) + float(c_array_cap) + float(c_spot_resume)) / 3.0

    # 5. Long-Horizon (>8h) Non-Interactive Token / Credential Renewal
    c_ttl_aware = bool(re.search(r"(?:8\s*-?\s*h|8\s+hour|multi-hour|long-running|ttl|expir)", txt))
    c_tok_refresh = bool(
        re.search(r"(?:token|credential|oauth|auth)", txt)
        and re.search(r"(?:renew|refresh|broker|daemon|background|periodic)", txt)
    )
    p_token = (float(c_ttl_aware) + float(c_tok_refresh)) / 2.0

    return {
        "timeout_checkpoint_hard_stop": round(p_timeout, 4),
        "input_size_memory_rightsizing": round(p_memory, 4),
        "scratch_nvme_staging": round(p_storage, 4),
        "partition_gpu_hygiene": round(p_partition, 4),
        "token_broker_renewal": round(p_token, 4),
    }


def _simulate_single_agent_trial(
    task: dict[str, Any],
    strengths: dict[str, float],
    *,
    shared_latency_multiplier: float,
    timeout_strictness: float,
    dataset_scale: float,
) -> dict[str, Any]:
    """Simulate Harbor container execution & deterministic script verification for one task."""
    eff_agent_timeout = task["agent_timeout_sec"] / max(0.5, timeout_strictness)
    eff_dataset_gb = task["dataset_gb"] * max(0.25, dataset_scale)

    # Agent authoring duration: without Early-Baseline + Hard-Stop protocol, agent spends ~90%
    # of nominal timeout polishing and hits Harbor timeout whenever model latency spikes (>1.11x)!
    s_time = strengths["timeout_checkpoint_hard_stop"]
    nominal_authoring_sec = task["agent_timeout_sec"] * (0.90 - 0.42 * s_time)
    actual_authoring_sec = nominal_authoring_sec * shared_latency_multiplier

    timed_out = actual_authoring_sec > eff_agent_timeout
    oom_failed = False
    nfs_stalled = False
    partition_failed = False
    auth_expired = False

    req_set = set(task["required_protocols"])
    is_light_task = (eff_dataset_gb <= 10.0) and (task["walltime_hours"] <= 4.0) and not task["spot_mode"]

    if "timeout_checkpoint_hard_stop" in req_set and s_time < 0.66 and not is_light_task:
        # Missing SIGUSR1 checkpoint trap or early baseline causes hard multi-hour/DDP trial timeout
        timed_out = True

    if "input_size_memory_rightsizing" in req_set:
        # Default 16 GB partition memory OOMs when working set (~2.65x dataset_gb + 4 GB) exceeds RAM
        s_mem = strengths["input_size_memory_rightsizing"]
        alloc_mem_gb = 18.0 if s_mem < 0.34 else (eff_dataset_gb * (1.75 + 1.45 * s_mem) + 8.0)
        required_working_mem_gb = eff_dataset_gb * 2.65 + 4.0
        if alloc_mem_gb < required_working_mem_gb:
            oom_failed = True

    if "scratch_nvme_staging" in req_set and strengths["scratch_nvme_staging"] < 0.66 and eff_dataset_gb > 9.5:
        nfs_stalled = True

    if "partition_gpu_hygiene" in req_set and strengths["partition_gpu_hygiene"] < 0.66 and task["spot_mode"]:
        partition_failed = True

    if ("token_broker_renewal" in req_set or task["walltime_hours"] > 8.0) and strengths["token_broker_renewal"] < 0.99:
        auth_expired = True

    passed = not (timed_out or oom_failed or nfs_stalled or partition_failed or auth_expired)

    # Shared environmental model-latency wave affects raw single-run trial scores directly,
    # whereas concurrent paired `(candidate - baseline)` subtraction cancels common-mode drift!
    mean_req_strength = float(np.mean([strengths[p] for p in task["required_protocols"]]))
    common_mode_latency_drift = -28.0 * (shared_latency_multiplier - 1.0)
    trial_points = (95.0 if passed else 0.0) + 5.0 * mean_req_strength + common_mode_latency_drift

    # Attribute primary root cause cleanly across all 5 enterprise HPC failure categories
    reason = "PASS"
    if actual_authoring_sec > eff_agent_timeout:
        reason = "TIMEOUT (exceeded Harbor agent budget during slow model window)"
    elif auth_expired:
        reason = f"AUTH_TOKEN_EXPIRED (8h credential TTL expired during {task['walltime_hours']:.0f}h job)"
    elif partition_failed:
        reason = "PARTITION_OR_SPOT_VIOLATION (missing --gpus=0 CPU isolation, array cap, or Spot resume)"
    elif nfs_stalled and not oom_failed:
        reason = "NFS_METADATA_THRASH (unpacked archives/caches on shared /home instead of NVMe /scratch)"
    elif oom_failed:
        reason = f"OOM_KILL (static memory insufficient for {eff_dataset_gb:.1f} GB input working set)"
    elif timed_out:
        reason = "TIMEOUT (missing SIGUSR1 checkpoint trap or early-baseline guard)"

    return {
        "passed": passed,
        "trial_points": trial_points,
        "authoring_sec": round(actual_authoring_sec, 1),
        "timed_out": timed_out,
        "oom_failed": oom_failed,
        "nfs_stalled": nfs_stalled,
        "partition_failed": partition_failed,
        "auth_expired": auth_expired,
        "reason": reason,
    }


def run_harbor_paired_evaluation(
    cand_desc: str,
    cand_body: str,
    base_desc: str,
    base_body: str,
    *,
    split: str = "dev",
    timeout_strictness: float = 1.0,
    dataset_scale: float = 1.0,
    latency_jitter_std: float = 0.18,
    num_paired_trials: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Run candidate skill side-by-side against baseline skill under identical latency waves."""
    tasks = load_harbor_tasks(split)
    if not tasks:
        raise RuntimeError(f"No Harbor tasks found for split '{split}'.")

    cand_strengths = extract_skill_protocol_strengths(cand_desc, cand_body)
    base_strengths = extract_skill_protocol_strengths(base_desc, base_body)

    rng = np.random.default_rng(seed)

    cand_trial_means: list[float] = []
    base_trial_means: list[float] = []
    paired_trial_deltas: list[float] = []

    cand_passes = 0
    base_passes = 0
    total_task_trials = len(tasks) * num_paired_trials

    cand_failures = {"timeout": 0, "oom": 0, "nfs_thrash": 0, "partition": 0, "auth_expiry": 0}
    base_failures = {"timeout": 0, "oom": 0, "nfs_thrash": 0, "partition": 0, "auth_expiry": 0}
    per_task_summary: list[dict[str, Any]] = []

    # Pre-draw shared environmental model-latency matrix `(num_paired_trials, len(tasks))`
    latency_waves = np.clip(
        rng.lognormal(mean=0.0, sigma=max(0.0, float(latency_jitter_std)), size=(num_paired_trials, len(tasks))),
        0.55,
        2.25,
    )

    task_cand_pass_counts = [0] * len(tasks)
    task_base_pass_counts = [0] * len(tasks)
    task_last_cand_reason = ["PASS"] * len(tasks)

    for tr_idx in range(num_paired_trials):
        c_pts_list: list[float] = []
        b_pts_list: list[float] = []

        for t_idx, task in enumerate(tasks):
            shared_lat = float(latency_waves[tr_idx, t_idx])

            b_out = _simulate_single_agent_trial(
                task,
                base_strengths,
                shared_latency_multiplier=shared_lat,
                timeout_strictness=timeout_strictness,
                dataset_scale=dataset_scale,
            )
            c_out = _simulate_single_agent_trial(
                task,
                cand_strengths,
                shared_latency_multiplier=shared_lat,
                timeout_strictness=timeout_strictness,
                dataset_scale=dataset_scale,
            )

            b_pts_list.append(b_out["trial_points"])
            c_pts_list.append(c_out["trial_points"])

            if b_out["passed"]:
                base_passes += 1
                task_base_pass_counts[t_idx] += 1
            else:
                if b_out["reason"].startswith("TIMEOUT"):
                    base_failures["timeout"] += 1
                elif b_out["reason"].startswith("AUTH_"):
                    base_failures["auth_expiry"] += 1
                elif b_out["reason"].startswith("PARTITION_"):
                    base_failures["partition"] += 1
                elif b_out["reason"].startswith("NFS_"):
                    base_failures["nfs_thrash"] += 1
                else:
                    base_failures["oom"] += 1

            if c_out["passed"]:
                cand_passes += 1
                task_cand_pass_counts[t_idx] += 1
            else:
                task_last_cand_reason[t_idx] = c_out["reason"]
                if c_out["reason"].startswith("TIMEOUT"):
                    cand_failures["timeout"] += 1
                elif c_out["reason"].startswith("AUTH_"):
                    cand_failures["auth_expiry"] += 1
                elif c_out["reason"].startswith("PARTITION_"):
                    cand_failures["partition"] += 1
                elif c_out["reason"].startswith("NFS_"):
                    cand_failures["nfs_thrash"] += 1
                else:
                    cand_failures["oom"] += 1

        c_mean = float(np.mean(c_pts_list))
        b_mean = float(np.mean(b_pts_list))
        cand_trial_means.append(c_mean)
        base_trial_means.append(b_mean)
        paired_trial_deltas.append(c_mean - b_mean)

    for t_idx, task in enumerate(tasks):
        per_task_summary.append(
            {
                "slug": task["slug"],
                "category": task["category"],
                "difficulty": task["difficulty"],
                "baseline_pass_rate_pct": round((task_base_pass_counts[t_idx] / num_paired_trials) * 100.0, 1),
                "candidate_pass_rate_pct": round((task_cand_pass_counts[t_idx] / num_paired_trials) * 100.0, 1),
                "diagnostic": task_last_cand_reason[t_idx],
            }
        )

    cand_pass_rate_pct = (cand_passes / max(1, total_task_trials)) * 100.0
    base_pass_rate_pct = (base_passes / max(1, total_task_trials)) * 100.0
    paired_delta_score = float(np.mean(paired_trial_deltas))

    raw_std = float(statistics.pstdev(cand_trial_means)) if len(cand_trial_means) > 1 else 0.0
    base_raw_std = float(statistics.pstdev(base_trial_means)) if len(base_trial_means) > 1 else 0.0
    unpaired_std = math.hypot(raw_std, base_raw_std)
    paired_std = float(statistics.pstdev(paired_trial_deltas)) if len(paired_trial_deltas) > 1 else 0.0

    # Mild conciseness regularizer above 4,500 chars so concise protocols beat bloated essays
    skill_chars = len(cand_desc.strip()) + len(cand_body.strip())
    conciseness_pen = max(0.0, (skill_chars - 4500) / 1000.0) * 0.75
    effective_delta = paired_delta_score - conciseness_pen

    return {
        "paired_delta_score": round(effective_delta, 4),
        "raw_paired_delta_pts": round(paired_delta_score, 4),
        "candidate_pass_rate_pct": round(cand_pass_rate_pct, 2),
        "baseline_pass_rate_pct": round(base_pass_rate_pct, 2),
        "pass_rate_gain_pct_pts": round(cand_pass_rate_pct - base_pass_rate_pct, 2),
        "candidate_mean_trial_score": round(float(np.mean(cand_trial_means)), 2),
        "baseline_mean_trial_score": round(float(np.mean(base_trial_means)), 2),
        "raw_score_std_across_latency_waves": round(raw_std, 3),
        "unpaired_delta_std_across_latency_waves": round(unpaired_std, 3),
        "paired_delta_std_across_latency_waves": round(paired_std, 3),
        "num_tasks": len(tasks),
        "num_paired_trials_per_task": num_paired_trials,
        "total_paired_trials": total_task_trials,
        "skill_char_length": skill_chars,
        "protocol_coverage": cand_strengths,
        "baseline_protocol_coverage": base_strengths,
        "candidate_failures": cand_failures,
        "baseline_failures": base_failures,
        "per_task_results": per_task_summary,
    }


def evaluate_program(
    candidate_code: str | None = None,
    split: str = "val",
    scenario_overrides: dict[str, Any] | None = None,
    *,
    require_markers: bool = True,
) -> dict[str, Any]:
    """3-Tier AlphaEvolve + Harbor Paired-Baseline Evaluator for Pillar 4."""
    canonical_seed = SEED_FILE_PATH.read_text(encoding="utf-8")
    if candidate_code is None:
        cand_dir = os.environ.get("_CANDIDATE_DIR", "")
        cand_path = Path(cand_dir) / "main.py" if cand_dir else SEED_FILE_PATH
        candidate_code = cand_path.read_text(encoding="utf-8")

    # Tier 1A: AST & EVOLVE-BLOCK integrity check
    try:
        spliced_source, _ = validate_candidate_source(
            candidate_code,
            canonical_seed_source=canonical_seed,
            required_functions=REQUIRED_FUNCTIONS,
            allowed_imports=ALLOWED_IMPORTS,
            require_markers=require_markers,
        )
        cand_mod = execute_candidate_module(spliced_source, "pillar4_candidate")
        build_fn = cand_mod.__dict__["build_skill"]
        cand_out = run_with_timeout(build_fn, timeout_sec=5.0)
    except (SandboxViolationError, CandidateTimeoutError) as exc:
        return make_evaluation_result(
            primary_metric="paired_delta_score",
            overall_score=HARD_GATE_PENALTY,
            valid=False,
            gate_passed=False,
            stage_reached="stage_1_skill_integrity",
            split=split,
            metrics={"error": str(exc)},
            insights=[("stage_1_sandbox_violation", str(exc))],
        )
    except Exception as exc:
        return make_evaluation_result(
            primary_metric="paired_delta_score",
            overall_score=HARD_GATE_PENALTY,
            valid=False,
            gate_passed=False,
            stage_reached="stage_1_skill_integrity",
            split=split,
            metrics={"error": str(exc)},
            insights=[("build_skill_runtime_error", f"build_skill() raised exception: {exc}")],
        )

    if not isinstance(cand_out, tuple) or len(cand_out) != 2:
        return make_evaluation_result(
            primary_metric="paired_delta_score",
            overall_score=HARD_GATE_PENALTY,
            valid=False,
            gate_passed=False,
            stage_reached="stage_1_skill_integrity",
            split=split,
            metrics={"error": "build_skill() must return a 2-tuple (description, body)."},
            insights=[("invalid_signature", "build_skill() must return `(description, body)`.")],
        )

    cand_desc, cand_body = cand_out
    ok, v_label, v_text = validate_skill_tier1(cand_desc, cand_body)
    if not ok:
        return make_evaluation_result(
            primary_metric="paired_delta_score",
            overall_score=HARD_GATE_PENALTY,
            valid=False,
            gate_passed=False,
            stage_reached="stage_1_skill_integrity",
            split=split,
            metrics={"violation": v_label, "detail": v_text},
            insights=[(v_label, v_text)],
        )

    # Load immutable baseline seed skill for concurrent paired comparison
    seed_mod = execute_candidate_module(canonical_seed, "pillar4_seed")
    base_desc, base_body = seed_mod.__dict__["build_skill"]()

    overrides = scenario_overrides or {}
    harbor_split = _normalize_split(split)

    eval_metrics = run_harbor_paired_evaluation(
        cand_desc=cand_desc,
        cand_body=cand_body,
        base_desc=base_desc,
        base_body=base_body,
        split=harbor_split,
        timeout_strictness=float(overrides.get("timeout_strictness", 1.0)),
        dataset_scale=float(overrides.get("dataset_scale", 1.0)),
        latency_jitter_std=float(overrides.get("latency_jitter_std", 0.18)),
        num_paired_trials=int(overrides.get("num_paired_trials", 5)),
        seed=int(overrides.get("seed", 42)),
    )

    # Concise Harbor-style failure/timeout notes fed back to AlphaEvolve Gemini
    insights: list[tuple[str, str]] = [
        (
            "harbor_paired_summary",
            f"paired_delta_score={eval_metrics['paired_delta_score']:+.2f} pts | "
            f"Candidate Pass Rate={eval_metrics['candidate_pass_rate_pct']:.1f}% vs "
            f"Concurrent Baseline={eval_metrics['baseline_pass_rate_pct']:.1f}% "
            f"(split={harbor_split}, {eval_metrics['total_paired_trials']} paired trials, "
            f"length={eval_metrics['skill_char_length']} chars)",
        )
    ]

    cf = eval_metrics["candidate_failures"]
    if cf["timeout"] > 0:
        insights.append(
            (
                "harbor_timeout_note",
                f"{cf['timeout']} trials hit agent/walltime timeout! Add an explicit protocol: "
                "(1) write a minimal valid baseline output to the target path FIRST before refining, "
                "(2) add `--signal=B:USR1@120` + `trap` to flush checkpoints before walltime expiry, and "
                "(3) **Hard Stop When Green** — exit immediately once acceptance checks pass.",
            )
        )
    if cf["oom"] > 0:
        insights.append(
            (
                "harbor_oom_note",
                f"{cf['oom']} trials failed with cgroup OOM_KILL! Instruct the agent to inspect input "
                "dataset byte size (`du -sb` / `stat`) and compute `--mem` dynamically with a `3x` working-set "
                "safety multiplier clamped to node limits instead of using default partition memory.",
            )
        )
    if cf["nfs_thrash"] > 0:
        insights.append(
            (
                "harbor_storage_note",
                f"{cf['nfs_thrash']} trials stalled on `/home` NFS metadata IOPS! Instruct the agent to stage/unpack "
                "archives onto local high-IOPS NVMe `/scratch`, redirect `TMPDIR`, `XDG_CACHE_HOME`, `PIP_CACHE_DIR`, "
                "and `R_LIBS_USER` away from `/home`, and stream cold outputs to `gs://` object storage.",
            )
        )
    if cf["partition"] > 0:
        insights.append(
            (
                "harbor_partition_spot_note",
                f"{cf['partition']} trials failed partition/Spot hygiene! Enforce `--gpus=0` and CPU/batch partition "
                "routing for non-GPU workloads (never GPU/ultramem pools), cap job arrays (`--array=1-N%32`), and "
                "add `--requeue` + idempotent `[ -f checkpoint ]` resume guards on preemptible Spot workers.",
            )
        )
    if cf["auth_expiry"] > 0:
        insights.append(
            (
                "harbor_credential_ttl_note",
                f"{cf['auth_expiry']} multi-hour (`>8h`) trials failed mid-run due to 8-hour short-lived OAuth token "
                "expiry! Require non-interactive background token-broker credential renewal on jobs exceeding 8 hours.",
            )
        )

    return make_evaluation_result(
        primary_metric="paired_delta_score",
        overall_score=float(eval_metrics["paired_delta_score"]),
        valid=True,
        gate_passed=True,
        stage_reached="stage_3_harbor_paired_eval",
        split=harbor_split,
        metrics=eval_metrics,
        insights=insights,
        secondary_scores={
            "candidate_pass_rate_pct": eval_metrics["candidate_pass_rate_pct"],
            "baseline_pass_rate_pct": eval_metrics["baseline_pass_rate_pct"],
        },
    )
