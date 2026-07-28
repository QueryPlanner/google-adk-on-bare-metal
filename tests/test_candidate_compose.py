"""Static and canonical contracts for the standalone candidate service."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

from agent.compose_env import write_compose_environment

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_COMPOSE_PATH = REPOSITORY_ROOT / "compose.candidate.yaml"
CANDIDATE_IMAGE = "agent:candidate-contract"
FORBIDDEN_SERVICE_KEYS = {
    "build",
    "command",
    "devices",
    "entrypoint",
    "env_file",
    "expose",
    "ipc",
    "networks",
    "pid",
    "ports",
    "privileged",
    "volumes",
}
CANDIDATE_ENVIRONMENT_ALLOWLIST = (
    "AGENT_NAME",
    "ROOT_AGENT_MODEL",
    "LOG_LEVEL",
    "TELEMETRY_NAMESPACE",
    "K_REVISION",
    "CANDIDATE_ENV_CANARY",
)
REQUIRED_ENVIRONMENT = {
    name: f"${{{name}:?Set {name} for the candidate}}"
    for name in CANDIDATE_ENVIRONMENT_ALLOWLIST
}
ALLOWED_ENVIRONMENT = {
    "AGENT_NAME": "candidate-static-agent",
    "ROOT_AGENT_MODEL": "gemini-2.5-flash",
    "LOG_LEVEL": "INFO",
    "TELEMETRY_NAMESPACE": "candidate-static",
    "K_REVISION": "candidate-static",
    "CANDIDATE_ENV_CANARY": "candidate-static-canary",
}
FIXED_ENVIRONMENT = {
    "ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS": "false",
    "ADK_DISABLE_LOAD_DOTENV": "true",
    "AGENT_DIR": "/app/src",
    "AGENT_ENGINE": "",
    "ALLOW_ORIGINS": "[]",
    "DATABASE_URL": "",
    "GEMINI_API_KEY": "",
    "GOOGLE_API_KEY": "",
    "GOOGLE_APPLICATION_CREDENTIALS": "",
    "GOOGLE_CLOUD_LOCATION": "",
    "GOOGLE_CLOUD_PROJECT": "",
    "GOOGLE_GENAI_USE_VERTEXAI": "false",
    "HOST": "127.0.0.1",
    "LANGFUSE_BASE_URL": "",
    "LANGFUSE_PUBLIC_KEY": "",
    "LANGFUSE_SECRET_KEY": "",
    "MEM0_LLM_API_KEY": "",
    "MEM0_QDRANT_HOST": "",
    "MEM0_QDRANT_PORT": "",
    "OPENROUTER_API_KEY": "",
    "OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE": "",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "",
    "OTEL_EXPORTER_OTLP_TRACES_HEADERS": "",
    "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "",
    "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT": "",
    "OTEL_GATEWAY_BEARER_TOKEN_FILE": "",
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "false",
    "OTEL_LOGS_EXPORTER": "none",
    "OTEL_METRICS_EXPORTER": "none",
    "OTEL_PYTHON_EXPORTER_OTLP_HTTP_CREDENTIAL_PROVIDER": "",
    "OTEL_PYTHON_EXPORTER_OTLP_HTTP_TRACES_CREDENTIAL_PROVIDER": "",
    "OTEL_SDK_DISABLED": "true",
    "OTEL_TRACES_EXPORTER": "none",
    "PORT": "8080",
    "RELOAD_AGENTS": "false",
    "SERVE_WEB_INTERFACE": "false",
    "VERTEXAI_LOCATION": "",
    "VERTEXAI_PROJECT": "",
}
EXPECTED_ENVIRONMENT = ALLOWED_ENVIRONMENT | FIXED_ENVIRONMENT
ARBITRARY_ENVIRONMENT = {
    "PYTHONPATH": "/hostile/python/path",
    "LD_PRELOAD": "/hostile/preload.so",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "https://hostile-collector.example.test",
}
AMBIENT_SECRET_VALUES = {
    "AMBIENT_SECRET": "ambient-process-secret-canary",
    "GH_TOKEN": "ambient-github-secret-canary",
    "HTTPS_PROXY": "https://ambient-proxy-secret-canary.example.test",
}


def _raw_candidate_configuration() -> dict[str, Any]:
    """Load the committed YAML without Compose applying defaults."""
    configuration = yaml.safe_load(CANDIDATE_COMPOSE_PATH.read_text(encoding="utf-8"))
    assert isinstance(configuration, dict)
    return configuration


def _minimal_process_environment(tmp_path: Path) -> dict[str, str]:
    """Return only the process settings needed to invoke Compose."""
    return {
        "COMPOSE_DISABLE_ENV_FILE": "1",
        "HOME": os.environ.get("HOME", str(tmp_path)),
        "LANG": "C",
        "PATH": os.environ["PATH"],
    }


def _docker_with_compose(tmp_path: Path) -> str:
    """Resolve the portable Compose boundary or skip like existing unit tests."""
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker CLI is unavailable")

    result = subprocess.run(  # noqa: S603 - resolved Docker, fixed arguments
        [docker, "compose", "version"],
        env=_minimal_process_environment(tmp_path),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("Docker Compose CLI is unavailable")
    return docker


def _write_candidate_environment(
    tmp_path: Path,
    *,
    missing_name: str | None = None,
) -> tuple[Path, set[str]]:
    """Write allowed values plus controls that the candidate must reject."""
    hostile_controls = {
        name: f'hostile-{index}-$VALUE "quote" \\ slash'
        for index, name in enumerate(FIXED_ENVIRONMENT)
    }
    file_environment = (
        ALLOWED_ENVIRONMENT | hostile_controls | ARBITRARY_ENVIRONMENT
    ).copy()
    if missing_name in file_environment:
        del file_environment[missing_name]

    environment_path = tmp_path / "candidate.env"
    write_compose_environment(
        environment_path,
        tuple(file_environment),
        file_environment,
    )
    assert environment_path.stat().st_mode & 0o777 == 0o600
    forbidden_values = set(hostile_controls.values()) | set(
        ARBITRARY_ENVIRONMENT.values()
    )
    return environment_path, forbidden_values


def _render_candidate_configuration(
    tmp_path: Path,
) -> tuple[dict[str, Any], str, set[str]]:
    """Render canonical JSON with a private env file and minimal host state."""
    docker = _docker_with_compose(tmp_path)
    environment_path, hostile_values = _write_candidate_environment(tmp_path)
    environment = (
        _minimal_process_environment(tmp_path)
        | AMBIENT_SECRET_VALUES
        | {
            "ENV_FILE": str(environment_path),
            "IMAGE": CANDIDATE_IMAGE,
        }
    )
    result = subprocess.run(  # noqa: S603 - resolved Docker, fixed arguments
        [
            docker,
            "compose",
            "--project-name",
            "candidate-compose-contract",
            "--env-file",
            str(environment_path),
            "-f",
            str(CANDIDATE_COMPOSE_PATH),
            "config",
            "--format",
            "json",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    configuration = json.loads(result.stdout)
    assert isinstance(configuration, dict)
    return configuration, result.stdout, hostile_values


def test_raw_candidate_is_one_standalone_service() -> None:
    """Keep the candidate independent from the production Compose model."""
    document = CANDIDATE_COMPOSE_PATH.read_text(encoding="utf-8")
    configuration = _raw_candidate_configuration()

    assert set(configuration) == {"services", "x-candidate-environment-file"}
    assert configuration["x-candidate-environment-file"] == (
        "${ENV_FILE:?Set ENV_FILE to the candidate environment file}"
    )
    assert set(configuration["services"]) == {"agent"}
    assert (
        "# Interpolated values come only from a private serializer-produced "
        "allowlist." in document
    )


def test_raw_candidate_requires_private_inputs_and_inherits_image_process() -> None:
    """Require caller-owned inputs while preserving image ENTRYPOINT and CMD."""
    service = _raw_candidate_configuration()["services"]["agent"]

    assert service["image"] == "${IMAGE:?Set IMAGE to the candidate image}"
    assert service["pull_policy"] == "never"
    assert "env_file" not in service
    assert "entrypoint" not in service
    assert "command" not in service


def test_raw_candidate_has_exact_isolation_and_privilege_model() -> None:
    """Reject production resources and every supported privilege escape."""
    service = _raw_candidate_configuration()["services"]["agent"]

    assert FORBIDDEN_SERVICE_KEYS.isdisjoint(service)
    assert service["network_mode"] == "none"
    assert service["user"] == "1000:1000"
    assert service["cap_drop"] == ["ALL"]
    assert service["security_opt"] == ["no-new-privileges:true"]
    assert service["restart"] == "no"
    assert service["stop_grace_period"] == "10s"


def test_raw_candidate_has_exact_environment_safety_model() -> None:
    """Neutralize dotenv, data, model, memory, and telemetry boundaries."""
    service = _raw_candidate_configuration()["services"]["agent"]

    assert tuple(REQUIRED_ENVIRONMENT) == CANDIDATE_ENVIRONMENT_ALLOWLIST
    assert service["environment"] == REQUIRED_ENVIRONMENT | FIXED_ENVIRONMENT
    assert "MEM0_ENABLED" not in service["environment"]


def test_raw_candidate_healthcheck_is_short_bounded_and_proxy_free() -> None:
    """Probe only loopback readiness without honoring proxy configuration."""
    healthcheck = _raw_candidate_configuration()["services"]["agent"]["healthcheck"]
    probe = healthcheck["test"]

    assert probe[:3] == ["CMD", "python", "-c"]
    assert len(probe) == 4
    assert "urllib.request.ProxyHandler({})" in probe[3]
    assert "http://127.0.0.1:8080/ready" in probe[3]
    assert "timeout=1" in probe[3]
    assert "0.0.0.0" not in probe[3]  # noqa: S104 - rejected public bind
    assert healthcheck == {
        "test": probe,
        "interval": "2s",
        "timeout": "2s",
        "retries": 10,
        "start_period": "20s",
    }


@pytest.mark.parametrize(
    "missing_name",
    ["IMAGE", "ENV_FILE", *CANDIDATE_ENVIRONMENT_ALLOWLIST],
)
def test_compose_rejects_missing_required_input(
    tmp_path: Path,
    missing_name: str,
) -> None:
    """Fail canonical rendering when either caller-owned input is absent."""
    docker = _docker_with_compose(tmp_path)
    environment_path, hostile_values = _write_candidate_environment(
        tmp_path,
        missing_name=missing_name,
    )
    environment = _minimal_process_environment(tmp_path) | {
        "ENV_FILE": str(environment_path),
        "IMAGE": CANDIDATE_IMAGE,
    }
    environment.pop(missing_name, None)

    result = subprocess.run(  # noqa: S603 - resolved Docker, fixed arguments
        [
            docker,
            "compose",
            "--project-name",
            "candidate-required-input-contract",
            "--env-file",
            str(environment_path),
            "-f",
            str(CANDIDATE_COMPOSE_PATH),
            "config",
            "--format",
            "json",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert missing_name in output
    assert all(value not in output for value in hostile_values)


def test_canonical_candidate_has_exact_safety_model(tmp_path: Path) -> None:
    """Verify Compose resolves the same standalone, no-egress contract."""
    configuration, _, _ = _render_candidate_configuration(tmp_path)
    service = configuration["services"]["agent"]

    assert set(configuration["services"]) == {"agent"}
    assert service["image"] == CANDIDATE_IMAGE
    assert service["pull_policy"] == "never"
    assert service["environment"] == EXPECTED_ENVIRONMENT
    assert service["network_mode"] == "none"
    assert service["user"] == "1000:1000"
    assert service["cap_drop"] == ["ALL"]
    assert service["security_opt"] == ["no-new-privileges:true"]
    assert service["restart"] == "no"
    assert service["stop_grace_period"] == "10s"
    canonical_forbidden_keys = FORBIDDEN_SERVICE_KEYS - {"command", "entrypoint"}
    assert canonical_forbidden_keys.isdisjoint(service)
    assert service["command"] is None
    assert service["entrypoint"] is None

    healthcheck = service["healthcheck"]
    assert healthcheck["test"][:3] == ["CMD", "python", "-c"]
    assert "urllib.request.ProxyHandler({})" in healthcheck["test"][3]
    assert "http://127.0.0.1:8080/ready" in healthcheck["test"][3]
    assert healthcheck["interval"] == "2s"
    assert healthcheck["timeout"] == "2s"
    assert healthcheck["retries"] == 10
    assert healthcheck["start_period"] == "20s"


def test_canonical_candidate_contains_no_ambient_secret_values(tmp_path: Path) -> None:
    """Prove host and env-file canaries cannot enter the candidate model."""
    _, canonical_json, hostile_values = _render_candidate_configuration(tmp_path)

    forbidden_values = hostile_values | set(AMBIENT_SECRET_VALUES.values())
    assert all(value not in canonical_json for value in forbidden_values)
