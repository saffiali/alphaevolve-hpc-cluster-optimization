"""Unit and adversarial sabotage tests for Managed AlphaEvolve Discovery Engine client adapter."""

from __future__ import annotations

from typing import Any
import pytest

from pillars.agent_skill_optimization.evaluator import evaluate_program as eval_p4
from pillars.common.evolution_runner import OFFLINE_REPLAY_BLOCKS, load_pillar_config
from pillars.common.managed_ae_client import (
    AlphaEvolveExperimentConfig,
    AlphaEvolveSourceFile,
    DiscoveryEngineAlphaEvolveClient,
    ManagedAlphaEvolveAdapter,
    TrialLogRingBuffer,
)


@pytest.mark.parametrize(
    ("loc", "expected_host"),
    [
        ("global", "https://discoveryengine.googleapis.com"),
        ("EU", "https://eu-discoveryengine.googleapis.com"),
        ("eu", "https://eu-discoveryengine.googleapis.com"),
        ("US", "https://us-discoveryengine.googleapis.com"),
        ("us", "https://us-discoveryengine.googleapis.com"),
    ],
)
def test_discovery_engine_regional_endpoint_routing(loc: str, expected_host: str) -> None:
    client = DiscoveryEngineAlphaEvolveClient(
        project_id="sample-hpc-proj",
        location=loc,
        engine="sample-ge-engine",
        token_provider=lambda: "mock-oauth-token",
    )
    assert client.base_url == expected_host
    assert (
        client.get_assistant_endpoint()
        == f"{expected_host}/v1alpha/projects/sample-hpc-proj/locations/{loc}/collections/default_collection/engines/sample-ge-engine/assistants/default_assistant"
    )


def test_discovery_engine_acquire_and_submit_lock_token_contract() -> None:
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def _mock_transport(
        method: str, url: str, headers: dict[str, str], payload: dict[str, Any] | None
    ) -> dict[str, Any]:
        assert headers["Authorization"] == "Bearer test-token"
        calls.append((method, url, payload))
        if url.endswith(":streamAssist"):
            return [{"sessionInfo": {"session": "projects/p/locations/eu/collections/default_collection/engines/e/sessions/s1"}}]
        if url.endswith("/alphaEvolveExperiments"):
            return {"name": "projects/p/locations/eu/collections/default_collection/engines/e/sessions/s1/alphaEvolveExperiments/exp1"}
        if url.endswith(":acquirePrograms"):
            return {
                "programs": [
                    {
                        "name": "projects/p/locations/eu/collections/default_collection/engines/e/sessions/s1/alphaEvolveExperiments/exp1/alphaEvolvePrograms/prog1",
                        "lockToken": "srv-lock-991",
                        "content": {"files": [{"path": "main.py", "content": "# seed"}]},
                    }
                ]
            }
        return {"status": "OK"}

    de_client = DiscoveryEngineAlphaEvolveClient(
        project_id="p",
        location="eu",
        engine="e",
        token_provider=lambda: "test-token",
        transport=_mock_transport,
    )
    sess = de_client.create_session()
    assert sess.endswith("/sessions/s1")

    exp = de_client.create_experiment(
        AlphaEvolveExperimentConfig(
            title="Skill Opt",
            problem_description="Optimize HPC skill",
            program_language="python",
        ),
        sess,
    )
    exp_name = exp["name"]
    de_client.create_initial_program(exp_name, [AlphaEvolveSourceFile(path="main.py", content="x = 1")])
    de_client.start_experiment(exp_name)

    acquired = de_client.acquire_programs(exp_name, desired_programs_count=1)
    assert len(acquired) == 1
    assert acquired[0].lock_token == "srv-lock-991"

    adapter = ManagedAlphaEvolveAdapter(
        mode="live",
        ge_app_id="e",
        project_id="p",
        location="eu",
        discovery_client=de_client,
    )
    assert adapter.backend == "discovery_engine"
    adapter._experiment_name = exp_name
    cand = adapter.acquire_candidate()
    assert cand.lock_token == "srv-lock-991"

    eval_payload = {
        "overall_score": 42.5,
        "valid": True,
        "scores": {"scores": [{"metric": "paired_delta_score", "score": 42.5}]},
        "insights": {"insights": [{"label": "harbor", "text": "Passed 7/8 dev tasks"}]},
    }
    sub = adapter.submit_evaluation(cand, eval_payload)
    assert sub.lock_token == "srv-lock-991"
    assert any(u.endswith(":submitProgramsEvaluations") for _, u, _ in calls)

    # Rejects re-submission or tampered lock_token
    with pytest.raises(ValueError, match="lock_token"):
        cand.lock_token = "forged-token"
        adapter.submit_evaluation(cand, eval_payload)


def test_managed_adapter_offline_replay_and_ring_buffer_bound() -> None:
    spec, seed_src = load_pillar_config("agent_skill_optimization")
    seed_eval = eval_p4(candidate_code=seed_src, split="dev")

    ring = TrialLogRingBuffer(max_entries=3, max_chars_per_entry=128, gcs_archive_uri="gs://sample-bucket/harbor-logs")
    adapter = ManagedAlphaEvolveAdapter(
        mode="replay",
        replay_blocks=OFFLINE_REPLAY_BLOCKS["agent_skill_optimization"],
        log_ring_buffer=ring,
    )
    assert adapter.backend == "offline_replay"
    adapter.initialize_experiment(
        title=spec["title"],
        problem_description=spec["problem_description"],
        seed_source=seed_src,
        seed_eval=seed_eval,
    )

    scores: list[float] = []
    for _ in range(4):
        prog = adapter.acquire_candidate()
        assert prog.content is not None
        res = eval_p4(candidate_code=prog.content.files[0].content, split="dev")
        adapter.submit_evaluation(prog, res)
        scores.append(res["overall_score"])

    # Ring buffer strictly bounded to max_entries=3 even after 4 trials (disk-fill protection)
    assert len(ring.snapshot()) == 3
    assert ring.total_logged_count == 4
    assert max(scores) > 50.0
