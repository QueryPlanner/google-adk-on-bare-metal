"""Compatibility contracts for the pinned Google ADK runtime stack."""

import ast
import json
import os
import runpy
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import FunctionType
from typing import cast

import pytest
from google.adk.artifacts import FileArtifactService
from google.adk.errors.input_validation_error import InputValidationError
from google.adk.evaluation.eval_case import (
    IntermediateData,
    Invocation,
    get_all_tool_calls,
)
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_metrics import (
    EvalMetricResult,
    EvalMetricResultPerInvocation,
    EvalStatus,
)
from google.adk.evaluation.eval_result import EvalCaseResult, EvalSetResult
from google.adk.evaluation.eval_set import EvalSet
from google.genai import types
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVAL_SET_PATH = REPOSITORY_ROOT / "tests" / "eval" / "adk_compatibility.evalset.json"
EVAL_CONFIG_PATH = REPOSITORY_ROOT / "tests" / "eval" / "adk_compatibility.config.json"
EVAL_README_PATH = REPOSITORY_ROOT / "tests" / "eval" / "README.md"
PRODUCTION_EVAL_PATH = REPOSITORY_ROOT / "tests" / "eval" / "production_adk_eval.py"
EVAL_ISOLATION_ENVIRONMENT = {
    "ADK_DISABLE_LOAD_DOTENV": "true",
    "GOOGLE_API_KEY": "",
    "MEM0_EMBEDDER_DIMS": "",
    "MEM0_EMBEDDER_MODEL": "__disabled_for_adk_compatibility_eval__",
    "OTEL_SDK_DISABLED": "true",
    "PYTEST_ADDOPTS": "",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "PYTEST_PLUGINS": "",
    "ROOT_AGENT_MODEL": "google/gemini-2.5-flash",
}


def _make_metric_results(
    *,
    response_status: EvalStatus = EvalStatus.PASSED,
    response_score: float | None = 0.9,
) -> list[EvalMetricResult]:
    return [
        EvalMetricResult(
            metric_name="tool_trajectory_avg_score",
            threshold=1.0,
            score=1.0,
            eval_status=EvalStatus.PASSED,
        ),
        EvalMetricResult(
            metric_name="response_match_score",
            threshold=0.8,
            score=response_score,
            eval_status=response_status,
        ),
    ]


def _make_structured_result(
    *,
    overall_response_status: EvalStatus = EvalStatus.PASSED,
    invocation_response_status: EvalStatus = EvalStatus.PASSED,
    response_score: float | None = 0.9,
    tool_name: str = "example_tool",
    tool_call_id: str | None = None,
) -> EvalSetResult:
    prompt = types.Content(
        role="user",
        parts=[
            types.Part(
                text=(
                    "Call example_tool exactly once. Then reply with exactly: "
                    "Successfully used example_tool."
                )
            )
        ],
    )
    expected_invocation = Invocation(
        invocation_id="example_tool_once_invocation",
        user_content=prompt,
        final_response=types.Content(
            role="model",
            parts=[types.Part(text="Successfully used example_tool.")],
        ),
        intermediate_data=IntermediateData(
            tool_uses=[types.FunctionCall(name="example_tool", args={})]
        ),
    )
    actual_invocation = Invocation(
        user_content=types.Content(
            role=prompt.role,
            parts=prompt.parts,
        ),
        final_response=types.Content(
            role="model",
            parts=[types.Part(text="Successfully used example_tool.")],
        ),
        intermediate_data=IntermediateData(
            tool_uses=[
                types.FunctionCall(
                    id=tool_call_id,
                    name=tool_name,
                    args={},
                )
            ]
        ),
    )
    case_result = EvalCaseResult(
        eval_set_id="adk_compatibility",
        eval_id="example_tool_once",
        final_eval_status=EvalStatus.PASSED,
        overall_eval_metric_results=_make_metric_results(
            response_status=overall_response_status,
            response_score=response_score,
        ),
        eval_metric_result_per_invocation=[
            EvalMetricResultPerInvocation(
                actual_invocation=actual_invocation,
                expected_invocation=expected_invocation,
                eval_metric_results=_make_metric_results(
                    response_status=invocation_response_status,
                    response_score=response_score,
                ),
            )
        ],
        session_id="synthetic-session",
    )
    return EvalSetResult(
        eval_set_result_id="synthetic-result",
        eval_set_id="adk_compatibility",
        eval_case_results=[case_result],
    )


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
    conversation = eval_set.eval_cases[0].conversation
    assert conversation is not None
    assert len(conversation) == 1
    assert [
        tool.model_dump(mode="json", exclude_none=True)
        for tool in get_all_tool_calls(conversation[0].intermediate_data)
    ] == [{"args": {}, "name": "example_tool"}]
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
    assert "uv sync --locked --no-default-groups --group eval" in instructions
    assert "pytest --noconftest --confcutdir=tests/eval" in instructions
    assert "tests/eval/production_adk_eval.py" in instructions
    assert "--show-capture=no" in instructions
    assert "--with" not in instructions
    assert "--print_detailed_results" not in instructions


def test_production_eval_is_explicit_synchronous_and_non_skipping() -> None:
    """Keep the paid gate outside normal collection and impossible to skip."""
    source = PRODUCTION_EVAL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(PRODUCTION_EVAL_PATH))
    synchronous_tests = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    asynchronous_tests = [
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name.startswith("test_")
    ]

    assert PRODUCTION_EVAL_PATH.name == "production_adk_eval.py"
    assert len(synchronous_tests) == 1
    assert synchronous_tests[0].name == "test_production_adk_compatibility"
    assert synchronous_tests[0].decorator_list == []
    assert asynchronous_tests == []
    assert "pytest.skip" not in source
    assert "skipif" not in source
    assert "xfail" not in source
    assert "--print_detailed_results" not in source
    assert "CliRunner().invoke" in source
    assert "logging.disable(logging.CRITICAL)" in source
    assert 'warnings.simplefilter("ignore")' in source
    assert "TemporaryDirectory" in source
    assert "ignore_patterns(" in source
    assert '".adk",' in source
    assert "_require_passing_summary(result.output)" in source
    assert "_require_passing_structured_result(isolated_agent)" in source


def test_production_eval_summary_is_exact_and_non_vacuous() -> None:
    """Reject empty, failed, duplicated, and over-counted ADK summaries."""
    namespace = runpy.run_path(str(PRODUCTION_EVAL_PATH))
    require_passing_summary = cast(
        Callable[[str], None],
        namespace["_require_passing_summary"],
    )
    passing = (
        "Eval Run Summary\nadk_compatibility:\n  Tests passed: 1\n  Tests failed: 0\n"
    )
    spoofed_before_marker = (
        "model output\n"
        "adk_compatibility:\n"
        "  Tests passed: 1\n"
        "  Tests failed: 0\n"
        "Eval Run Summary\n"
    )

    require_passing_summary(passing)
    for invalid in (
        "",
        passing.replace("Tests passed: 1", "Tests passed: 0"),
        passing.replace("Tests failed: 0", "Tests failed: 1"),
        passing.replace("Tests passed: 1", "Tests passed: 2"),
        passing + passing,
        passing + "another_eval:\n" + "  Tests passed: 1\n" + "  Tests failed: 0\n",
        spoofed_before_marker,
        passing + "unexpected terminal output\n",
    ):
        with pytest.raises(
            AssertionError,
            match="did not pass exactly one case",
        ):
            require_passing_summary(invalid)


def test_production_eval_rejects_drifted_eval_case(tmp_path: Path) -> None:
    """Pin the prompt, golden response, and session input used for promotion."""
    namespace = runpy.run_path(str(PRODUCTION_EVAL_PATH))
    require_committed_contract = cast(
        FunctionType,
        namespace["_require_committed_eval_contract"],
    )
    function_globals = cast(
        dict[str, object],
        require_committed_contract.__globals__,
    )
    drifted_eval_set = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    invocation = drifted_eval_set["eval_cases"][0]["conversation"][0]
    invocation["user_content"]["parts"][0]["text"] = "Use any tool."
    invocation["final_response"]["parts"][0]["text"] = "Any response."
    drifted_eval_set["eval_cases"][0]["session_input"] = {
        "app_name": "agent",
        "user_id": "different-user",
        "state": {"weakened": True},
    }
    drifted_path = tmp_path / "drifted.evalset.json"
    drifted_path.write_text(json.dumps(drifted_eval_set), encoding="utf-8")
    function_globals["EVAL_SET_PATH"] = drifted_path

    with pytest.raises(
        AssertionError,
        match="evaluation assets are invalid",
    ):
        require_committed_contract()


def test_production_eval_rejects_custom_metric_override(tmp_path: Path) -> None:
    """Keep ADK's pinned built-in response matcher authoritative."""
    namespace = runpy.run_path(str(PRODUCTION_EVAL_PATH))
    require_committed_contract = cast(
        FunctionType,
        namespace["_require_committed_eval_contract"],
    )
    function_globals = cast(
        dict[str, object],
        require_committed_contract.__globals__,
    )
    drifted_config = json.loads(EVAL_CONFIG_PATH.read_text(encoding="utf-8"))
    drifted_config["custom_metrics"] = {
        "response_match_score": {"code_config": {"name": "agent.tools.example_tool"}}
    }
    drifted_path = tmp_path / "drifted.config.json"
    drifted_path.write_text(json.dumps(drifted_config), encoding="utf-8")
    function_globals["EVAL_CONFIG_PATH"] = drifted_path

    with pytest.raises(
        AssertionError,
        match="evaluation assets are invalid",
    ):
        require_committed_contract()


@pytest.mark.parametrize(
    ("overall_status", "invocation_status", "response_score"),
    [
        (EvalStatus.NOT_EVALUATED, EvalStatus.PASSED, 0.9),
        (EvalStatus.PASSED, EvalStatus.NOT_EVALUATED, 0.9),
        (EvalStatus.FAILED, EvalStatus.PASSED, 0.9),
        (EvalStatus.PASSED, EvalStatus.PASSED, None),
        (EvalStatus.PASSED, EvalStatus.PASSED, 0.79),
        (EvalStatus.PASSED, EvalStatus.PASSED, 1.01),
        (EvalStatus.PASSED, EvalStatus.PASSED, float("nan")),
        (EvalStatus.PASSED, EvalStatus.PASSED, float("inf")),
    ],
)
def test_production_eval_structured_result_rejects_incomplete_metrics(
    tmp_path: Path,
    overall_status: EvalStatus,
    invocation_status: EvalStatus,
    response_score: float | None,
) -> None:
    """Reject metric failures hidden by ADK's aggregate CLI summary."""
    namespace = runpy.run_path(str(PRODUCTION_EVAL_PATH))
    require_passing_result = cast(
        Callable[[Path], None],
        namespace["_require_passing_structured_result"],
    )
    isolated_agent = tmp_path / "agent"
    history = isolated_agent / ".adk" / "eval_history"
    history.mkdir(parents=True)
    result_path = history / "synthetic.evalset_result.json"
    result_path.write_text(
        _make_structured_result(
            overall_response_status=overall_status,
            invocation_response_status=invocation_status,
            response_score=response_score,
        ).model_dump_json(),
        encoding="utf-8",
    )

    with pytest.raises(
        AssertionError,
        match="structured result is invalid",
    ):
        require_passing_result(isolated_agent)


def test_production_eval_structured_result_is_exact_and_non_vacuous(
    tmp_path: Path,
) -> None:
    """Require one parseable result with one passing case and invocation."""
    namespace = runpy.run_path(str(PRODUCTION_EVAL_PATH))
    require_passing_result = cast(
        Callable[[Path], None],
        namespace["_require_passing_structured_result"],
    )
    isolated_agent = tmp_path / "agent"

    with pytest.raises(
        AssertionError,
        match="structured result is invalid",
    ):
        require_passing_result(isolated_agent)

    history = isolated_agent / ".adk" / "eval_history"
    history.mkdir(parents=True)
    result_path = history / "synthetic.evalset_result.json"
    result_path.write_text(
        _make_structured_result().model_dump_json(),
        encoding="utf-8",
    )

    require_passing_result(isolated_agent)

    (history / "unexpected.evalset_result.json").write_text(
        _make_structured_result().model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(
        AssertionError,
        match="structured result is invalid",
    ):
        require_passing_result(isolated_agent)


@pytest.mark.parametrize("corruption", ["metric_mismatch", "missing_expected"])
def test_production_eval_structured_result_rejects_internal_inconsistency(
    tmp_path: Path,
    corruption: str,
) -> None:
    """Reject artifacts the pinned one-invocation evaluator cannot produce."""
    namespace = runpy.run_path(str(PRODUCTION_EVAL_PATH))
    require_passing_result = cast(
        Callable[[Path], None],
        namespace["_require_passing_structured_result"],
    )
    structured_result = _make_structured_result()
    invocation_result = structured_result.eval_case_results[
        0
    ].eval_metric_result_per_invocation[0]
    if corruption == "metric_mismatch":
        invocation_result.eval_metric_results[1].score = 0.8
    else:
        invocation_result.expected_invocation = None

    isolated_agent = tmp_path / "agent"
    history = isolated_agent / ".adk" / "eval_history"
    history.mkdir(parents=True)
    (history / "synthetic.evalset_result.json").write_text(
        structured_result.model_dump_json(),
        encoding="utf-8",
    )

    with pytest.raises(
        AssertionError,
        match="structured result is invalid",
    ):
        require_passing_result(isolated_agent)


def test_production_eval_structured_result_rejects_wrong_actual_tool(
    tmp_path: Path,
) -> None:
    """Require the real invocation to match the committed exact trajectory."""
    namespace = runpy.run_path(str(PRODUCTION_EVAL_PATH))
    require_passing_result = cast(
        Callable[[Path], None],
        namespace["_require_passing_structured_result"],
    )
    isolated_agent = tmp_path / "agent"
    history = isolated_agent / ".adk" / "eval_history"
    history.mkdir(parents=True)
    (history / "synthetic.evalset_result.json").write_text(
        _make_structured_result(tool_name="unexpected_tool").model_dump_json(),
        encoding="utf-8",
    )

    with pytest.raises(
        AssertionError,
        match="structured result is invalid",
    ):
        require_passing_result(isolated_agent)


def test_production_eval_structured_result_allows_provider_tool_call_id(
    tmp_path: Path,
) -> None:
    """Ignore provider transport IDs exactly as ADK's trajectory metric does."""
    namespace = runpy.run_path(str(PRODUCTION_EVAL_PATH))
    require_passing_result = cast(
        Callable[[Path], None],
        namespace["_require_passing_structured_result"],
    )
    isolated_agent = tmp_path / "agent"
    history = isolated_agent / ".adk" / "eval_history"
    history.mkdir(parents=True)
    (history / "synthetic.evalset_result.json").write_text(
        _make_structured_result(tool_call_id="provider-call-id").model_dump_json(),
        encoding="utf-8",
    )

    require_passing_result(isolated_agent)


@pytest.mark.parametrize(
    ("provider_key", "root_model", "expected_error"),
    [
        (
            None,
            "google/gemini-2.5-flash",
            "OPENROUTER_API_KEY is required for the production evaluation",
        ),
        (
            "production-eval-provider-secret-canary",
            "openrouter/provider/not-the-committed-model",
            "ROOT_AGENT_MODEL does not match the production evaluation contract",
        ),
    ],
)
def test_production_eval_preflight_fails_closed_without_leaking_credentials(
    provider_key: str | None,
    root_model: str,
    expected_error: str,
) -> None:
    """Execute the explicit pytest boundary without reaching a model."""
    environment = os.environ.copy()
    for name in (
        *EVAL_ISOLATION_ENVIRONMENT,
        "OPENROUTER_API_KEY",
    ):
        environment.pop(name, None)
    environment["PYTEST_ADDOPTS"] = "--collect-only"
    environment["PYTEST_PLUGINS"] = "untrusted_ambient_plugin"
    for name in tuple(environment):
        if name.startswith("COV_CORE_"):
            environment.pop(name)
    environment.update(EVAL_ISOLATION_ENVIRONMENT)
    environment["ROOT_AGENT_MODEL"] = root_model
    if provider_key is not None:
        environment["OPENROUTER_API_KEY"] = provider_key

    result = subprocess.run(  # noqa: S603 - fixed interpreter and repository path
        [
            sys.executable,
            "-m",
            "pytest",
            "--noconftest",
            "--confcutdir=tests/eval",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            str(PRODUCTION_EVAL_PATH),
            "-q",
            "--tb=line",
            "--disable-warnings",
            "--show-capture=no",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    output = result.stdout + result.stderr

    assert result.returncode == 1
    assert expected_error in output
    if provider_key is not None:
        assert provider_key not in output


def test_production_eval_ignores_external_skip_conftest(tmp_path: Path) -> None:
    """Prevent a neighboring conftest from skipping the paid gate."""
    eval_directory = tmp_path / "tests" / "eval"
    eval_directory.mkdir(parents=True)
    copied_gate = eval_directory / PRODUCTION_EVAL_PATH.name
    copied_gate.write_text(
        PRODUCTION_EVAL_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (eval_directory / "conftest.py").write_text(
        "def pytest_collection_modifyitems(items):\n"
        "    for item in items:\n"
        "        item.add_marker('skip')\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    for name in (*EVAL_ISOLATION_ENVIRONMENT, "OPENROUTER_API_KEY"):
        environment.pop(name, None)
    for name in tuple(environment):
        if name.startswith("COV_CORE_"):
            environment.pop(name)
    environment.update(EVAL_ISOLATION_ENVIRONMENT)

    result = subprocess.run(  # noqa: S603 - fixed interpreter and temp path
        [
            sys.executable,
            "-m",
            "pytest",
            "--noconftest",
            "--confcutdir=tests/eval",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            str(copied_gate),
            "-q",
            "--tb=line",
            "--disable-warnings",
            "--show-capture=no",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    output = result.stdout + result.stderr

    assert result.returncode == 1
    assert "OPENROUTER_API_KEY is required for the production evaluation" in output
    assert "skipped" not in output


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
