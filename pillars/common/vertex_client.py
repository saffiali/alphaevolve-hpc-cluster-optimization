"""Zero-hardcoded-credential Vertex AI Gemini REST client for AlphaEvolve code mutation.

Supports authentication across Cloud Run / Cloud Batch (GCE Metadata Server),
explicit environment variables, and local developer `gcloud` CLI sessions without
embedding any project IDs, accounts, or tokens in source code.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any

from pillars.common.models import EVOLVE_END_MARKER, EVOLVE_START_MARKER

METADATA_ROOT = "http://metadata.google.internal/computeMetadata/v1"
METADATA_HEADERS = {"Metadata-Flavor": "Google"}


class VertexAIClientError(RuntimeError):
    """Raised when Vertex AI project resolution, authentication, or generation fails."""


def _is_cloud_managed_runtime() -> bool:
    """Return True when running inside Cloud Run, Cloud Batch, or App Engine."""
    return any(
        os.environ.get(k)
        for k in ("K_SERVICE", "CLOUD_RUN_JOB", "BATCH_TASK_INDEX", "BATCH_JOB_UID")
    )


def resolve_gcp_project_id(explicit_project_id: str | None = None) -> str:
    """Resolve active GCP project ID dynamically from arguments, env, gcloud, or metadata."""
    if explicit_project_id and explicit_project_id.strip():
        return explicit_project_id.strip()

    for env_key in ("GOOGLE_CLOUD_PROJECT", "GCP_PROJECT_ID", "GCLOUD_PROJECT"):
        val = os.environ.get(env_key, "").strip()
        if val:
            return val

    if not _is_cloud_managed_runtime():
        # Prefer active developer gcloud configuration before VM host metadata
        env = dict(os.environ)
        env.pop("CLOUDSDK_AUTH_ACCESS_TOKEN_FILE", None)
        try:
            proc = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True,
                text=True,
                env=env,
                timeout=8,
                check=False,
            )
            proj = proc.stdout.strip()
            if proc.returncode == 0 and proj and proj != "(unset)":
                return proj
        except Exception:
            pass

    # Cloud Run / Cloud Batch metadata server fallback
    try:
        req = urllib.request.Request(
            f"{METADATA_ROOT}/project/project-id",
            headers=METADATA_HEADERS,
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            proj = resp.read().decode("utf-8").strip()
            if proj:
                return proj
    except Exception:
        pass

    raise VertexAIClientError(
        "Unable to resolve GCP project ID. Set GOOGLE_CLOUD_PROJECT environment variable."
    )


def resolve_access_token() -> str:
    """Obtain short-lived OAuth2 bearer token via env, gcloud CLI, or Cloud Run/Batch metadata."""
    env_tok = os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN", "").strip()
    if env_tok:
        return env_tok

    if not _is_cloud_managed_runtime():
        env = dict(os.environ)
        env.pop("CLOUDSDK_AUTH_ACCESS_TOKEN_FILE", None)
        try:
            proc = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True,
                text=True,
                env=env,
                timeout=12,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
        except Exception:
            pass

    # Cloud Run / Cloud Batch metadata server
    try:
        req = urllib.request.Request(
            f"{METADATA_ROOT}/instance/service-accounts/default/token",
            headers=METADATA_HEADERS,
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            tok = str(payload.get("access_token", "")).strip()
            if tok:
                return tok
    except Exception as exc:
        raise VertexAIClientError(f"Failed to acquire GCP access token: {exc}") from exc

    raise VertexAIClientError("Failed to acquire non-empty GCP access token.")


def extract_code_from_llm_response(raw_text: str) -> str:
    """Extract clean Python code for an EVOLVE-BLOCK from a Gemini response."""
    text = raw_text.strip()
    # Check if markers are explicitly included
    if EVOLVE_START_MARKER in text and EVOLVE_END_MARKER in text:
        s_pos = text.index(EVOLVE_START_MARKER) + len(EVOLVE_START_MARKER)
        e_pos = text.index(EVOLVE_END_MARKER, s_pos)
        inner = text[s_pos:e_pos].strip("\r\n")
        if inner.strip():
            return inner

    # Otherwise extract fenced ```python ... ``` or ``` ... ``` block
    fenced = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        # Pick longest code block
        best = max(fenced, key=len).strip("\r\n")
        inner_lines = [
            ln
            for ln in best.splitlines()
            if ln.strip() not in (EVOLVE_START_MARKER, EVOLVE_END_MARKER)
        ]
        return "\n".join(inner_lines)

    return text


class VertexGeminiMutator:
    """Calls Vertex AI Gemini (`gemini-2.5-flash` by default) to evolve `# EVOLVE-BLOCK` code."""

    def __init__(
        self,
        project_id: str | None = None,
        location: str | None = None,
        model: str | None = None,
        timeout_sec: float = 45.0,
    ) -> None:
        self.project_id = resolve_gcp_project_id(project_id)
        self.location = (
            location
            or os.environ.get("GOOGLE_CLOUD_REGION")
            or os.environ.get("VERTEX_AI_LOCATION")
            or "us-central1"
        )
        self.model = (
            model
            or os.environ.get("VERTEX_AI_MODEL")
            or "gemini-2.5-flash"
        )
        self.timeout_sec = timeout_sec
        self._cached_token: str | None = None
        self._token_ts: float = 0.0

    def _get_token(self) -> str:
        now = time.monotonic()
        if self._cached_token is None or (now - self._token_ts) > 1200.0:
            self._cached_token = resolve_access_token()
            self._token_ts = now
        return self._cached_token

    def generate_mutated_block(
        self,
        *,
        problem_description: str,
        surrounding_context: str,
        parent_block: str,
        parent_score: float,
        parent_insights: list[dict[str, str]],
        inspiration_elites: list[tuple[float, str, str]] | None = None,
        mutation_directive: str = "",
        temperature: float = 0.65,
        max_output_tokens: int = 8192,
        thinking_budget: int = 1024,
    ) -> dict[str, Any]:
        """Request an evolved `# EVOLVE-BLOCK` mutation from Vertex AI Gemini."""
        insights_str = "\n".join(
            f"  - [{item.get('label', 'metric')}]: {item.get('text', '')}"
            for item in parent_insights
        ) or "  - Baseline run completed."

        elites_section = ""
        if inspiration_elites:
            chunks: list[str] = []
            for idx, (sc, niche, code) in enumerate(inspiration_elites[:2], start=1):
                chunks.append(
                    f"### Inspiration Elite #{idx} (niche={niche}, score={sc:.4f})\n```python\n{code.strip()}\n```"
                )
            elites_section = "\n\n## High-Scoring Inspiration Elites From Other Islands\n" + "\n\n".join(chunks)

        system_instruction = (
            "You are AlphaEvolve, an evolutionary scientific and systems code optimization engine. "
            "Your task is to mutate ONLY the Python code inside `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` "
            "to strictly maximize the evaluator primary score while obeying every hard constraint floor and "
            "sandbox rule. NEVER import forbidden modules (`os`, `sys`, `subprocess`, `socket`, `ctypes`, "
            "`open`, `eval`, `exec`). Return ONLY the replacement code inside a single ```python ... ``` block "
            "wrapped in `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` at column 0."
        )

        user_prompt = f"""## Optimization Problem Specification
{problem_description.strip()}

## Immutable Harness Context (Read-Only Surrounding Module)
```python
{surrounding_context.strip()}
```

## Current Parent Candidate (Score = {parent_score:.4f})
Diagnostic evaluator insights from parent run:
{insights_str}

```python
{EVOLVE_START_MARKER}
{parent_block.strip()}
{EVOLVE_END_MARKER}
```{elites_section}

## Mutation Goal for This Generation
{mutation_directive or "Design an algorithmic improvement that boosts the primary score without violating any hard SLA, physics accuracy, or sandbox constraint."}

Output ONLY the complete replacement `# EVOLVE-BLOCK-START` ... `# EVOLVE-BLOCK-END` Python code block (preserving required function names and signatures, starting markers at column 0, with zero prose outside the code block)."""

        endpoint = (
            f"https://{self.location}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project_id}/locations/{self.location}/"
            f"publishers/google/models/{self.model}:generateContent"
        )
        gen_cfg: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        }
        if thinking_budget >= 0:
            gen_cfg["thinkingConfig"] = {"thinkingBudget": int(thinking_budget)}

        body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": gen_cfg,
        }

        last_err: Exception | None = None
        for attempt in range(3):
            t0 = time.perf_counter()
            try:
                token = self._get_token()
                req = urllib.request.Request(
                    endpoint,
                    data=json.dumps(body).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    raw_resp = json.loads(resp.read().decode("utf-8"))
                latency_ms = (time.perf_counter() - t0) * 1000.0

                candidates = raw_resp.get("candidates") or []
                if not candidates:
                    raise VertexAIClientError(f"Empty candidates from Vertex AI: {raw_resp}")
                parts = candidates[0].get("content", {}).get("parts", [])
                # Concatenate non-thought text parts
                text_chunks = [
                    p.get("text", "")
                    for p in parts
                    if not p.get("thought", False) and p.get("text")
                ]
                combined_text = "\n".join(text_chunks).strip()
                if not combined_text:
                    raise VertexAIClientError("Vertex AI response contained empty text parts.")

                mutated_block = extract_code_from_llm_response(combined_text)
                usage = raw_resp.get("usageMetadata", {})
                return {
                    "mutated_block": mutated_block,
                    "latency_ms": round(latency_ms, 1),
                    "model": self.model,
                    "location": self.location,
                    "prompt_tokens": int(usage.get("promptTokenCount", 0)),
                    "candidates_tokens": int(usage.get("candidatesTokenCount", 0)),
                    "total_tokens": int(usage.get("totalTokenCount", 0)),
                }
            except urllib.error.HTTPError as exc:
                err_body = exc.read().decode("utf-8", errors="replace")[:500]
                last_err = VertexAIClientError(f"HTTP {exc.code} from Vertex AI: {err_body}")
                if exc.code == 401:
                    self._cached_token = None
                elif exc.code == 400 and "thinkingConfig" in err_body:
                    body["generationConfig"].pop("thinkingConfig", None)
                    continue
                time.sleep(1.0 * (attempt + 1))
            except Exception as exc:
                last_err = exc
                time.sleep(1.0 * (attempt + 1))

        raise VertexAIClientError(f"Vertex AI generation failed after retries: {last_err}")
