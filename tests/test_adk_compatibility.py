"""Compatibility contracts for the pinned Google ADK runtime stack."""

from pathlib import Path

import pytest
from google.adk.artifacts import FileArtifactService
from google.adk.errors.input_validation_error import InputValidationError
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_set import EvalSet
from google.genai import types
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVAL_SET_PATH = REPOSITORY_ROOT / "tests" / "eval" / "adk_compatibility.evalset.json"
EVAL_CONFIG_PATH = REPOSITORY_ROOT / "tests" / "eval" / "adk_compatibility.config.json"
EVAL_README_PATH = REPOSITORY_ROOT / "tests" / "eval" / "README.md"


async def test_file_artifacts_reject_identity_path_traversal(
    tmp_path: Path,
) -> None:
    """Keep caller-controlled identities inside the configured artifact root."""
    artifact_root = tmp_path / "artifacts"
    service = FileArtifactService(root_dir=artifact_root)

    for user_id, session_id in (
        ("../escaped-user", "session"),
        ("user", "../escaped-session"),
        (r"..\escaped-user", "session"),
    ):
        with pytest.raises(InputValidationError):
            await service.save_artifact(
                app_name="agent",
                user_id=user_id,
                session_id=session_id,
                filename="proof.txt",
                artifact=types.Part(text="must stay inside the artifact root"),
            )

    assert list(tmp_path.rglob("proof.txt")) == []


async def test_file_artifacts_survive_service_recreation(tmp_path: Path) -> None:
    """Exercise the real file backend across a fresh service instance."""
    artifact_root = tmp_path / "artifacts"
    original = FileArtifactService(root_dir=artifact_root)

    version = await original.save_artifact(
        app_name="agent",
        user_id="user",
        session_id="session",
        filename="report.txt",
        artifact=types.Part(text="persistent artifact"),
    )
    reloaded = await FileArtifactService(root_dir=artifact_root).load_artifact(
        app_name="agent",
        user_id="user",
        session_id="session",
        filename="report.txt",
        version=version,
    )

    assert version == 0
    assert reloaded is not None
    assert reloaded.text == "persistent artifact"


def test_adk_compatibility_eval_files_use_supported_schema() -> None:
    """Validate the committed behavioral check with ADK's real models."""
    eval_set = EvalSet.model_validate_json(EVAL_SET_PATH.read_text(encoding="utf-8"))
    eval_config = EvalConfig.model_validate_json(
        EVAL_CONFIG_PATH.read_text(encoding="utf-8")
    )

    assert eval_set.eval_set_id == "adk_compatibility"
    assert [case.eval_id for case in eval_set.eval_cases] == ["example_tool_once"]
    assert eval_config.model_dump(mode="json")["criteria"] == {
        "tool_trajectory_avg_score": {
            "threshold": 1.0,
            "match_type": "EXACT",
        },
        "response_match_score": 0.8,
    }

    instructions = EVAL_README_PATH.read_text(encoding="utf-8")
    assert "OPENROUTER_API_KEY" in instructions
    assert "google-adk[eval]==1.36.2" in instructions
    assert "MEM0_EMBEDDER_DIMS=" in instructions
    assert "src/agent tests/eval/adk_compatibility.evalset.json" in instructions
    assert "--print_detailed_results" in instructions


def test_adk_and_genai_instrumentors_support_resolved_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Instrument the resolved ADK and GenAI SDKs without exporting telemetry."""
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    adk_instrumentor = GoogleADKInstrumentor()
    genai_instrumentor = GoogleGenAiSdkInstrumentor()

    try:
        adk_instrumentor.instrument()
        genai_instrumentor.instrument()

        assert adk_instrumentor.is_instrumented_by_opentelemetry
        assert genai_instrumentor.is_instrumented_by_opentelemetry
    finally:
        genai_instrumentor.uninstrument()
        adk_instrumentor.uninstrument()
