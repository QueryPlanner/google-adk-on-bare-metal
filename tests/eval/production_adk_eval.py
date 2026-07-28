"""Explicit real-model gate for a production VM deployment."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import tempfile
import warnings
from pathlib import Path

from click.testing import CliRunner
from google.adk.cli.cli_tools_click import main as adk_cli
from google.adk.evaluation.eval_case import get_all_tool_calls
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_metrics import EvalMetricResult, EvalStatus
from google.adk.evaluation.eval_result import EvalSetResult
from google.adk.evaluation.eval_set import EvalSet

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVAL_SET_PATH = REPOSITORY_ROOT / "tests" / "eval" / "adk_compatibility.evalset.json"
EVAL_CONFIG_PATH = REPOSITORY_ROOT / "tests" / "eval" / "adk_compatibility.config.json"
EXPECTED_ENVIRONMENT = {
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
EXPECTED_CRITERIA = {
    "tool_trajectory_avg_score": {
        "threshold": 1.0,
        "match_type": "EXACT",
    },
    "response_match_score": 0.8,
}
EXPECTED_TOOL_CALL = {"args": {}, "name": "example_tool"}
EXPECTED_INVOCATION = {
    "invocation_id": "example_tool_once_invocation",
    "user_content": {
        "role": "user",
        "parts": [
            {
                "text": (
                    "Call example_tool exactly once. Then reply with exactly: "
                    "Successfully used example_tool."
                )
            }
        ],
    },
    "final_response": {
        "role": "model",
        "parts": [{"text": "Successfully used example_tool."}],
    },
    "intermediate_data": {"tool_uses": [EXPECTED_TOOL_CALL]},
}
EXPECTED_EVAL_SET = {
    "eval_set_id": "adk_compatibility",
    "name": "Google ADK compatibility",
    "description": "Verifies the template's real LiteLLM tool-call path.",
    "eval_cases": [
        {
            "eval_id": "example_tool_once",
            "conversation": [EXPECTED_INVOCATION],
            "session_input": {
                "app_name": "agent",
                "user_id": "adk-compatibility-eval",
                "state": {},
            },
        }
    ],
}
EXPECTED_EVAL_CONFIG = {"criteria": EXPECTED_CRITERIA}
EXPECTED_METRIC_THRESHOLDS = {
    "tool_trajectory_avg_score": 1.0,
    "response_match_score": 0.8,
}
SUMMARY_MARKER = "Eval Run Summary\n"
EXPECTED_SUMMARY = "adk_compatibility:\n  Tests passed: 1\n  Tests failed: 0"


def _require_runtime_contract() -> str:
    for name, expected in EXPECTED_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise AssertionError(
                f"{name} does not match the production evaluation contract"
            )

    provider_key = os.environ.get("OPENROUTER_API_KEY")
    if provider_key is None or not provider_key.strip():
        raise AssertionError(
            "OPENROUTER_API_KEY is required for the production evaluation"
        )
    return provider_key


def _require_committed_eval_contract() -> None:
    try:
        eval_set_document = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
        eval_config_document = json.loads(EVAL_CONFIG_PATH.read_text(encoding="utf-8"))
        eval_set = EvalSet.model_validate(eval_set_document)
        eval_config = EvalConfig.model_validate(eval_config_document)
    except (OSError, ValueError):
        raise AssertionError("The production evaluation assets are invalid") from None

    if (
        eval_set_document != EXPECTED_EVAL_SET
        or eval_config_document != EXPECTED_EVAL_CONFIG
    ):
        raise AssertionError("The production evaluation assets are invalid")

    if eval_set.eval_set_id != "adk_compatibility":
        raise AssertionError("The production EvalSet identity is invalid")
    if [case.eval_id for case in eval_set.eval_cases] != ["example_tool_once"]:
        raise AssertionError("The production EvalSet must contain exactly one case")

    conversation = eval_set.eval_cases[0].conversation
    if conversation is None or len(conversation) != 1:
        raise AssertionError("The production EvalSet must contain one invocation")
    expected_tools = [
        {"args": tool.args, "name": tool.name}
        for tool in get_all_tool_calls(conversation[0].intermediate_data)
    ]
    if expected_tools != [EXPECTED_TOOL_CALL]:
        raise AssertionError("The production EvalSet tool trajectory is invalid")
    if eval_config.model_dump(mode="json")["criteria"] != EXPECTED_CRITERIA:
        raise AssertionError("The production evaluation criteria are invalid")


def _require_passing_summary(output: str) -> None:
    if output.count(SUMMARY_MARKER) != 1:
        raise AssertionError(
            "The production ADK evaluation did not pass exactly one case"
        )
    summary = output.rsplit(SUMMARY_MARKER, maxsplit=1)[1].strip()
    if summary != EXPECTED_SUMMARY:
        raise AssertionError(
            "The production ADK evaluation did not pass exactly one case"
        )


def _require_passing_metrics(metrics: list[EvalMetricResult]) -> None:
    metrics_by_name = {metric.metric_name: metric for metric in metrics}
    if len(metrics) != len(EXPECTED_METRIC_THRESHOLDS) or set(metrics_by_name) != set(
        EXPECTED_METRIC_THRESHOLDS
    ):
        raise AssertionError("The production ADK structured result is invalid")

    for name, threshold in EXPECTED_METRIC_THRESHOLDS.items():
        metric = metrics_by_name[name]
        if (
            metric.eval_status != EvalStatus.PASSED
            or metric.threshold != threshold
            or metric.score is None
            or not math.isfinite(metric.score)
            or metric.score < threshold
            or metric.score > 1.0
        ):
            raise AssertionError("The production ADK structured result is invalid")


def _require_passing_structured_result(isolated_agent: Path) -> None:
    result_paths = sorted(
        (isolated_agent / ".adk" / "eval_history").glob("*.evalset_result.json")
    )
    if len(result_paths) != 1:
        raise AssertionError("The production ADK structured result is invalid")

    try:
        eval_set_result = EvalSetResult.model_validate_json(
            result_paths[0].read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        raise AssertionError(
            "The production ADK structured result is invalid"
        ) from None

    if (
        eval_set_result.eval_set_id != "adk_compatibility"
        or len(eval_set_result.eval_case_results) != 1
    ):
        raise AssertionError("The production ADK structured result is invalid")

    case_result = eval_set_result.eval_case_results[0]
    if (
        case_result.eval_set_id != "adk_compatibility"
        or case_result.eval_id != "example_tool_once"
        or case_result.final_eval_status != EvalStatus.PASSED
        or not case_result.session_id.strip()
        or len(case_result.eval_metric_result_per_invocation) != 1
    ):
        raise AssertionError("The production ADK structured result is invalid")

    invocation_result = case_result.eval_metric_result_per_invocation[0]
    expected_invocation = invocation_result.expected_invocation
    if (
        expected_invocation is None
        or expected_invocation.model_dump(
            mode="json",
            exclude_none=True,
            exclude_defaults=True,
        )
        != EXPECTED_INVOCATION
        or invocation_result.actual_invocation.user_content.model_dump(
            mode="json",
            exclude_none=True,
            exclude_defaults=True,
        )
        != EXPECTED_INVOCATION["user_content"]
    ):
        raise AssertionError("The production ADK structured result is invalid")

    final_response = invocation_result.actual_invocation.final_response
    if final_response is None or not any(
        part.text and part.text.strip() for part in final_response.parts or []
    ):
        raise AssertionError("The production ADK structured result is invalid")

    actual_tools = [
        {"args": tool.args, "name": tool.name}
        for tool in get_all_tool_calls(
            invocation_result.actual_invocation.intermediate_data
        )
    ]
    if actual_tools != [EXPECTED_TOOL_CALL]:
        raise AssertionError("The production ADK structured result is invalid")

    _require_passing_metrics(case_result.overall_eval_metric_results)
    _require_passing_metrics(invocation_result.eval_metric_results)
    overall_metrics = {
        metric.metric_name: (
            metric.eval_status,
            metric.threshold,
            metric.score,
        )
        for metric in case_result.overall_eval_metric_results
    }
    invocation_metrics = {
        metric.metric_name: (
            metric.eval_status,
            metric.threshold,
            metric.score,
        )
        for metric in invocation_result.eval_metric_results
    }
    if overall_metrics != invocation_metrics:
        raise AssertionError("The production ADK structured result is invalid")


def test_production_adk_compatibility() -> None:
    """Require one real ADK pass without exposing captured model output."""
    provider_key = _require_runtime_contract()
    _require_committed_eval_contract()

    with tempfile.TemporaryDirectory(prefix="adk-production-eval-") as temp:
        isolated_agent = Path(temp) / "agent"
        shutil.copytree(
            REPOSITORY_ROOT / "src" / "agent",
            isolated_agent,
            ignore=shutil.ignore_patterns(
                ".adk",
                ".env",
                ".env.*",
                "__pycache__",
                "*.pyc",
            ),
        )
        previous_logging_disable = logging.root.manager.disable
        try:
            logging.disable(logging.CRITICAL)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = CliRunner().invoke(
                    adk_cli,
                    [
                        "eval",
                        str(isolated_agent),
                        str(EVAL_SET_PATH),
                        "--config_file_path",
                        str(EVAL_CONFIG_PATH),
                    ],
                )
        finally:
            logging.disable(previous_logging_disable)

        if provider_key in result.output:
            raise AssertionError("The production evaluation exposed its credential")
        if result.exit_code != 0:
            raise AssertionError("The production ADK evaluation command failed")
        _require_passing_structured_result(isolated_agent)

    _require_passing_summary(result.output)
