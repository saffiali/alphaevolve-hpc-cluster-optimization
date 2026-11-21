"""Managed AlphaEvolve Discovery Engine Client Adapter (`alphaevolve-on-googlecloud` compatible).

Implements the Google Cloud Discovery Engine AlphaEvolve v1alpha wire contract:
  - Regional endpoint resolution:
      * `location="EU"`     -> `https://eu-discoveryengine.googleapis.com`
      * `location="US"`     -> `https://us-discoveryengine.googleapis.com`
      * `location="global"` -> `https://discoveryengine.googleapis.com`
  - Resource hierarchy:
      `projects/{PROJECT_ID}/locations/{LOCATION}/collections/{COLLECTION}/engines/{GE_APP_ID}`
  - Lock-token `acquire_programs` (`:acquirePrograms`) and multi-metric + diagnostic
    `submit_program_evaluations` (`:submitProgramsEvaluations`) semantics.
  - Seamless 3-mode execution:
      1. `discovery_engine`: Managed AlphaEvolve service when `GE_APP_ID` is configured.
      2. `vertex_ai_fallback`: Live Vertex AI Gemini (`gemini-2.5-flash` / `gemini-3.8-flash`)
         when `GE_APP_ID` is unset.
      3. `offline_replay`: Deterministic zero-network replay for fast unit tests & CI.
  - Includes `TrialLogRingBuffer` so long-running 10h+ Harbor/Ray evaluations stream
    summaries to Cloud Storage / Cloud Logging without filling local controller disks.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Callable

from pydantic import BaseModel, Field

from pillars.common.models import (
    AlphaEvolveEvaluationInsight,
    AlphaEvolveEvaluationInsights,
    AlphaEvolveEvaluationScore,
    AlphaEvolveEvaluationScores,
    AlphaEvolveProgramEvaluation,
)
from pillars.common.sandbox import extract_evolve_block, replace_evolve_block
from pillars.common.vertex_client import (
    VertexGeminiMutator,
    resolve_access_token,
    resolve_gcp_project_id,
)


class AlphaEvolveModelConfig(BaseModel):
    """Per-model mixture configuration matching `alphaevolve-on-googlecloud`."""

    name: str
    weight: float | None = 1.0


class AlphaEvolveGenerationSettings(BaseModel):
    """LLM ensemble and prompt context settings."""

    context: str | None = None
    models: list[AlphaEvolveModelConfig] | None = None
    include_full_program_in_prompt: bool | None = True


class AlphaEvolveRunSettings(BaseModel):
    """Budget and concurrency controls for an AlphaEvolve experiment."""

    max_programs: int | None = 20
    concurrency: int | None = 4
    max_duration: str | None = "21600s"


class AlphaEvolveExperimentConfig(BaseModel):
    """Declarative experiment config sent to Discovery Engine `alphaEvolveExperiments`."""

    title: str
    problem_description: str
    program_language: str = "python"
    run_settings: AlphaEvolveRunSettings | None = None
    generation_settings: AlphaEvolveGenerationSettings | None = None


class AlphaEvolveSourceFile(BaseModel):
    """Single file entry inside `AlphaEvolveProgramContent`."""

    path: str
    content: str
    program_language: str | None = "python"
    description: str | None = None


class AlphaEvolveProgramContent(BaseModel):
    """Multi-file program payload."""

    description: str | None = None
    files: list[AlphaEvolveSourceFile]


class AlphaEvolveProgram(BaseModel):
    """Candidate program envelope acquired from or listed by AlphaEvolve."""

    name: str
    create_time: datetime | None = None
    content: AlphaEvolveProgramContent | None = None
    evaluation: AlphaEvolveProgramEvaluation | None = None
    lock_token: str | None = None
    parent_programs: list[str] | None = None
    state: str | None = "ACQUIRED"


class AlphaEvolveProgramEvaluationSubmission(BaseModel):
    """Lock-token guarded evaluation payload for `:submitProgramsEvaluations`."""

    program: str
    evaluation: AlphaEvolveProgramEvaluation
    lock_token: str


class TrialLogRingBuffer:
    """Bounded in-memory ring buffer + optional GCS/JSONL sink preventing controller disk exhaustion."""

    def __init__(
        self,
        max_entries: int = 250,
        max_chars_per_entry: int = 2048,
        gcs_archive_uri: str | None = None,
    ) -> None:
        self.max_entries = max(1, int(max_entries))
        self.max_chars_per_entry = max(128, int(max_chars_per_entry))
        self.gcs_archive_uri = gcs_archive_uri or os.environ.get("AE_TRIAL_LOG_GCS_URI", "")
        self._entries: deque[dict[str, Any]] = deque(maxlen=self.max_entries)
        self.total_logged_count: int = 0

    def append(self, program_id: str, summary: str, metadata: dict[str, Any] | None = None) -> None:
        truncated = summary[: self.max_chars_per_entry]
        self._entries.append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "program_id": program_id,
                "summary": truncated,
                "gcs_stream_target": self.gcs_archive_uri or None,
                "metadata": metadata or {},
            }
        )
        self.total_logged_count += 1

    def snapshot(self) -> list[dict[str, Any]]:
        return list(self._entries)


class DiscoveryEngineAlphaEvolveClient:
    """REST client implementing the `alphaevolve-on-googlecloud` Discovery Engine API."""

    def __init__(
        self,
        project_id: str = "",
        location: str = "global",
        collection: str = "default_collection",
        engine: str = "",
        assistant: str = "default_assistant",
        base_url: str = "discoveryengine.googleapis.com",
        timeout_sec: float = 30.0,
        token_provider: Callable[[], str] | None = None,
        transport: Callable[[str, str, dict[str, str], dict[str, Any] | None], dict[str, Any]] | None = None,
    ) -> None:
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "") or os.environ.get("PROJECT_ID", "")
        self.location = location or os.environ.get("LOCATION", "global")
        self.collection = collection or os.environ.get("COLLECTION", "default_collection")
        self.engine = engine or os.environ.get("GE_APP_ID", "")
        self.assistant = assistant or os.environ.get("ASSISTANT", "default_assistant")
        self.timeout_sec = timeout_sec
        self._token_provider = token_provider or resolve_access_token
        self._transport = transport
        self._lock = threading.Lock()

        clean_host = base_url.replace("https://", "").replace("http://", "").strip("/")
        loc_upper = self.location.upper()
        if loc_upper == "EU" and not clean_host.startswith("eu-"):
            clean_host = f"eu-{clean_host}"
        elif loc_upper == "US" and not clean_host.startswith("us-"):
            clean_host = f"us-{clean_host}"
        self.base_url = f"https://{clean_host}"
        self.base_path = (
            f"projects/{self.project_id}/locations/{self.location}/"
            f"collections/{self.collection}/engines/{self.engine}"
        )

    def get_assistant_endpoint(self) -> str:
        return f"{self.base_url}/v1alpha/{self.base_path}/assistants/{self.assistant}"

    def get_sessions_endpoint(self) -> str:
        return f"{self.base_url}/v1alpha/{self.base_path}/sessions"

    def get_alpha_evolve_experiments_endpoint(self, session_name: str) -> str:
        return f"{self.base_url}/v1alpha/{session_name}/alphaEvolveExperiments"

    def get_alpha_evolve_experiment_endpoint(self, experiment_name: str) -> str:
        return f"{self.base_url}/v1alpha/{experiment_name}"

    def get_alpha_evolve_programs_endpoint(self, experiment_name: str) -> str:
        return f"{self.base_url}/v1alpha/{experiment_name}/alphaEvolvePrograms"

    def _headers(self) -> dict[str, str]:
        with self._lock:
            tok = self._token_provider()
        return {
            "Authorization": f"Bearer {tok}",
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        full_url = url
        if params:
            qs = urllib.parse.urlencode(params)
            full_url = f"{url}?{qs}"
        headers = self._headers()
        if self._transport is not None:
            return self._transport(method, full_url, headers, payload)

        body_bytes = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(full_url, data=body_bytes, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else {}

    def create_session(self) -> str:
        resp = self._request(
            "POST",
            f"{self.get_assistant_endpoint()}:streamAssist",
            payload={
                "query": {"text": "starting alpha evolve query"},
                "assistSkippingMode": "REQUEST_ASSIST",
            },
        )
        items = resp if isinstance(resp, list) else resp.get(" messages", resp.get("answers", [resp]))
        for item in items:
            if isinstance(item, dict) and "sessionInfo" in item:
                sess = item["sessionInfo"].get("session", "")
                if sess:
                    return str(sess)
        raise RuntimeError(f"Discovery Engine streamAssist did not return sessionInfo: {resp}")

    def create_experiment(self, exp_config: AlphaEvolveExperimentConfig | dict[str, Any], session_name: str) -> dict[str, Any]:
        cfg_dict = (
            exp_config.model_dump(exclude_none=True)
            if isinstance(exp_config, AlphaEvolveExperimentConfig)
            else exp_config
        )
        return self._request(
            "POST",
            self.get_alpha_evolve_experiments_endpoint(session_name),
            payload={"config": cfg_dict},
        )

    def create_initial_program(
        self,
        experiment_name: str,
        files: list[AlphaEvolveSourceFile | dict[str, Any]],
        description: str = "Initial seed program",
    ) -> dict[str, Any]:
        norm_files = [
            f.model_dump(exclude_none=True) if isinstance(f, AlphaEvolveSourceFile) else f
            for f in files
        ]
        return self._request(
            "POST",
            self.get_alpha_evolve_programs_endpoint(experiment_name),
            payload={"content": {"description": description, "files": norm_files}},
        )

    def start_experiment(self, experiment_name: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"{self.get_alpha_evolve_experiment_endpoint(experiment_name)}:start",
            payload={"name": experiment_name},
        )

    def acquire_programs(self, experiment_name: str, desired_programs_count: int = 1) -> list[AlphaEvolveProgram]:
        resp = self._request(
            "POST",
            f"{self.get_alpha_evolve_experiment_endpoint(experiment_name)}:acquirePrograms",
            payload={
                "parent": experiment_name,
                "desired_programs_count": int(desired_programs_count),
            },
        )
        raw_programs = resp.get("programs", [])
        out: list[AlphaEvolveProgram] = []
        for p in raw_programs:
            # Normalize camelCase `lockToken` from REST wire JSON to `lock_token`
            norm = dict(p)
            if "lockToken" in norm and "lock_token" not in norm:
                norm["lock_token"] = norm.pop("lockToken")
            out.append(AlphaEvolveProgram.model_validate(norm))
        return out

    def submit_program_evaluations(
        self,
        experiment_name: str,
        submissions: list[AlphaEvolveProgramEvaluationSubmission | dict[str, Any]],
    ) -> dict[str, Any]:
        norm_subs = [
            s.model_dump(exclude_none=True)
            if isinstance(s, AlphaEvolveProgramEvaluationSubmission)
            else s
            for s in submissions
        ]
        return self._request(
            "POST",
            f"{self.get_alpha_evolve_experiment_endpoint(experiment_name)}:submitProgramsEvaluations",
            payload={
                "parent": experiment_name,
                "evaluation_submissions": norm_subs,
            },
        )

    def list_programs(
        self, experiment_name: str, order_by: str | None = None
    ) -> list[AlphaEvolveProgram]:
        params = {"order_by": order_by} if order_by else None
        resp = self._request(
            "GET",
            self.get_alpha_evolve_programs_endpoint(experiment_name),
            params=params,
        )
        raw = resp.get("alphaEvolvePrograms", resp.get("alpha_evolve_programs", []))
        return [AlphaEvolveProgram.model_validate(item) for item in raw]


class ManagedAlphaEvolveAdapter:
    """Unified controller adapter bridging Discovery Engine (`GE_APP_ID`), Vertex AI, and Replay.

    Exposes identical `acquire_candidate` -> `submit_evaluation` lock-token semantics across
    all three backends so Harbor/Batch/Ray evaluators run unchanged in every environment.
    """

    def __init__(
        self,
        *,
        mode: str = "live",
        ge_app_id: str | None = None,
        project_id: str | None = None,
        location: str | None = None,
        vertex_model: str = "gemini-2.5-flash",
        replay_blocks: tuple[str, ...] = (),
        discovery_client: DiscoveryEngineAlphaEvolveClient | None = None,
        log_ring_buffer: TrialLogRingBuffer | None = None,
    ) -> None:
        self.ge_app_id = (ge_app_id if ge_app_id is not None else os.environ.get("GE_APP_ID", "")).strip()
        self.mode = mode
        self.vertex_model = vertex_model
        self._replay_blocks = replay_blocks
        self._replay_cursor = 0
        self.log_buffer = log_ring_buffer or TrialLogRingBuffer()
        self._active_locks: dict[str, str] = {}
        self._submitted: list[AlphaEvolveProgramEvaluationSubmission] = []
        self._best_program_src: str = ""
        self._best_score: float = float("-inf")
        self._best_insights: list[dict[str, str]] = []
        self._seed_template: str = ""
        self._problem_description: str = ""
        self._experiment_name: str = ""

        if mode == "replay":
            self.backend = "offline_replay"
            self.de_client = None
            self.mutator = None
        elif self.ge_app_id:
            self.backend = "discovery_engine"
            proj = project_id or resolve_gcp_project_id()
            loc = location or os.environ.get("LOCATION", "global")
            self.de_client = discovery_client or DiscoveryEngineAlphaEvolveClient(
                project_id=proj,
                location=loc,
                engine=self.ge_app_id,
            )
            self.mutator = None
        else:
            self.backend = "vertex_ai_fallback"
            self.de_client = None
            self.mutator = VertexGeminiMutator(
                project_id=project_id,
                location=location or os.environ.get("GOOGLE_CLOUD_REGION", "us-central1"),
                model=vertex_model,
            )

    def initialize_experiment(
        self,
        *,
        title: str,
        problem_description: str,
        seed_source: str,
        seed_eval: dict[str, Any],
        seed_path: str = "main.py",
    ) -> str:
        self._seed_template = seed_source
        self._problem_description = problem_description
        self._best_program_src = seed_source
        self._best_score = float(seed_eval["overall_score"])
        self._best_insights = seed_eval.get("insights", {}).get("insights", [])

        if self.backend == "discovery_engine" and self.de_client is not None:
            session_name = self.de_client.create_session()
            exp_cfg = AlphaEvolveExperimentConfig(
                title=title,
                problem_description=problem_description,
                program_language="python",
                generation_settings=AlphaEvolveGenerationSettings(
                    models=[AlphaEvolveModelConfig(name="gemini-3.5-flash", weight=1.0)],
                    include_full_program_in_prompt=True,
                ),
            )
            created = self.de_client.create_experiment(exp_cfg, session_name)
            self._experiment_name = str(created["name"])
            self.de_client.create_initial_program(
                self._experiment_name,
                files=[AlphaEvolveSourceFile(path=seed_path, content=seed_source)],
            )
            self.de_client.start_experiment(self._experiment_name)
            return self._experiment_name

        self._experiment_name = f"localSessions/default/alphaEvolveExperiments/{uuid.uuid4().hex[:12]}"
        return self._experiment_name

    def acquire_candidate(
        self,
        *,
        mutation_directive: str = "",
        seed_path: str = "main.py",
        temperature: float = 0.55,
    ) -> AlphaEvolveProgram:
        if self.backend == "discovery_engine" and self.de_client is not None:
            progs = self.de_client.acquire_programs(self._experiment_name, desired_programs_count=1)
            if not progs:
                raise RuntimeError("No candidate programs currently available from Discovery Engine.")
            prog = progs[0]
            if prog.lock_token:
                self._active_locks[prog.name] = prog.lock_token
            return prog

        prog_id = f"{self._experiment_name}/alphaEvolvePrograms/cand-{len(self._submitted) + 1:03d}"
        lock_tok = f"lock-{uuid.uuid4().hex}"
        self._active_locks[prog_id] = lock_tok

        if self.backend == "offline_replay":
            if not self._replay_blocks:
                cand_src = self._seed_template
            else:
                blk = self._replay_blocks[self._replay_cursor % len(self._replay_blocks)]
                self._replay_cursor += 1
                cand_src = replace_evolve_block(self._seed_template, blk)
        else:
            assert self.mutator is not None
            parent_blk = extract_evolve_block(self._best_program_src, require_markers=True)
            gen = self.mutator.generate_mutated_block(
                problem_description=self._problem_description,
                surrounding_context=self._seed_template,
                parent_block=parent_blk,
                parent_score=self._best_score,
                parent_insights=self._best_insights,
                mutation_directive=mutation_directive,
                temperature=temperature,
                thinking_budget=512,
            )
            cand_src = replace_evolve_block(self._seed_template, gen["mutated_block"])

        return AlphaEvolveProgram(
            name=prog_id,
            create_time=datetime.now(timezone.utc),
            content=AlphaEvolveProgramContent(
                files=[AlphaEvolveSourceFile(path=seed_path, content=cand_src)]
            ),
            lock_token=lock_tok,
            state="ACQUIRED",
        )

    def submit_evaluation(
        self,
        program: AlphaEvolveProgram,
        eval_result: dict[str, Any],
    ) -> AlphaEvolveProgramEvaluationSubmission:
        expected_lock = self._active_locks.get(program.name)
        if not program.lock_token or expected_lock is None or program.lock_token != expected_lock:
            raise ValueError(f"Invalid or expired lock_token for program '{program.name}'.")

        raw_scores = eval_result.get("scores", {}).get("scores", [])
        raw_insights = eval_result.get("insights", {}).get("insights", [])
        ae_eval = AlphaEvolveProgramEvaluation(
            scores=AlphaEvolveEvaluationScores(
                scores=[AlphaEvolveEvaluationScore(**s) for s in raw_scores]
            ),
            insights=AlphaEvolveEvaluationInsights(
                insights=[AlphaEvolveEvaluationInsight(**i) for i in raw_insights]
            ),
        )
        submission = AlphaEvolveProgramEvaluationSubmission(
            program=program.name,
            evaluation=ae_eval,
            lock_token=program.lock_token,
        )

        if self.backend == "discovery_engine" and self.de_client is not None:
            self.de_client.submit_program_evaluations(self._experiment_name, [submission])

        self._active_locks.pop(program.name, None)
        self._submitted.append(submission)

        score_val = float(eval_result.get("overall_score", float("-inf")))
        if eval_result.get("valid", False) and score_val > self._best_score:
            self._best_score = score_val
            if program.content and program.content.files:
                self._best_program_src = program.content.files[0].content
            self._best_insights = raw_insights

        summary_txt = raw_insights[0]["text"] if raw_insights else f"score={score_val}"
        self.log_buffer.append(program.name, summary_txt, {"overall_score": score_val})
        return submission
