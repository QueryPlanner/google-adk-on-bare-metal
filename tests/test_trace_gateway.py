"""Static and resolved-Compose contracts for the optional trace gateway."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASE_COMPOSE_PATH = REPOSITORY_ROOT / "compose.yaml"
GATEWAY_COMPOSE_PATH = REPOSITORY_ROOT / "compose.trace-gateway.yaml"
COLLECTOR_CONFIG_PATH = REPOSITORY_ROOT / "deploy" / "otel-collector" / "config.yaml"
CERTIFICATE_EXTENSION_PATH = (
    REPOSITORY_ROOT / "deploy" / "otel-collector" / "server-cert.ext"
)
AGENT_ENV_EXAMPLE_PATH = (
    REPOSITORY_ROOT / "deploy" / "otel-collector" / "agent.env.example"
)
COLLECTOR_ENV_EXAMPLE_PATH = (
    REPOSITORY_ROOT / "deploy" / "otel-collector" / "collector.env.example"
)
COMPOSE_ENV_EXAMPLE_PATH = (
    REPOSITORY_ROOT / "deploy" / "otel-collector" / "compose.env.example"
)
OBSERVABILITY_GUIDE_PATH = REPOSITORY_ROOT / "docs" / "base-infra" / "observability.md"
ARCHITECTURE_DECISION_PATH = (
    REPOSITORY_ROOT / "docs" / "adr" / "0001-private-tls-trace-gateway.md"
)
COMPOSE_WORKFLOW_PATH = (
    REPOSITORY_ROOT / ".github" / "workflows" / "test-docker-compose.yml"
)
GITIGNORE_PATH = REPOSITORY_ROOT / ".gitignore"
SETUP_SCRIPT_PATH = REPOSITORY_ROOT / "setup.sh"
COLLECTOR_IMAGE = (
    "ghcr.io/open-telemetry/opentelemetry-collector-releases/"
    "opentelemetry-collector-contrib:0.157.0@"
    "sha256:f2f01157055a9b2aab9df7118e1f1c9abf345e99b23bc7a2bc791db374a7d0f6"
)
SAFE_SPAN_ATTRIBUTES = [
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.experimental.reasoning_tokens",
    "gen_ai.usage.experimental.system_instruction_tokens",
]


def _load_collector_configuration() -> dict[str, Any]:
    configuration = yaml.safe_load(COLLECTOR_CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(configuration, dict)
    return configuration


def _write_synthetic_gateway_files(tmp_path: Path) -> Path:
    """Create secret-free files needed only for Compose interpolation."""
    agent_env = tmp_path / "agent.env"
    collector_env = tmp_path / "collector.env"
    secret_directory = tmp_path / "secrets"
    secret_directory.mkdir()
    agent_env.write_text(
        "\n".join(
            [
                "ROOT_AGENT_MODEL=google/gemini-test",
                "OPENROUTER_API_KEY=synthetic-model-boundary",
                "LANGFUSE_BASE_URL=https://vendor-fallback.example.test",
                "LANGFUSE_PUBLIC_KEY=hostile-public-key",
                "LANGFUSE_SECRET_KEY=hostile-secret-key",
                (
                    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="
                    "https://vendor-fallback.example.test/v1/traces"
                ),
                "OTEL_EXPORTER_OTLP_TRACES_HEADERS=authorization=hostile-header",
                "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=grpc",
                ("OTEL_PYTHON_EXPORTER_OTLP_HTTP_CREDENTIAL_PROVIDER=hostile-provider"),
                (
                    "OTEL_PYTHON_EXPORTER_OTLP_HTTP_TRACES_CREDENTIAL_PROVIDER="
                    "hostile-trace-provider"
                ),
                "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=false",
                "",
            ]
        ),
        encoding="utf-8",
    )
    collector_env.write_text(
        "OTEL_GATEWAY_LANGFUSE_AUTHORITY=langfuse.example.test\n",
        encoding="utf-8",
    )
    secret_paths = {
        name: secret_directory / name
        for name in (
            "ca.pem",
            "server.pem",
            "server.key",
            "receiver.token",
            "langfuse-basic.token",
        )
    }
    for name, path in secret_paths.items():
        path.write_text(f"synthetic-{name}\n", encoding="utf-8")

    compose_env = tmp_path / "compose.env"
    compose_env.write_text(
        "\n".join(
            [
                "AGENT_NAME=resolved-gateway-agent",
                "OTEL_GATEWAY_SECRET_GID=10002",
                f"OTEL_GATEWAY_AGENT_ENV_FILE={agent_env}",
                f"OTEL_GATEWAY_COLLECTOR_ENV_FILE={collector_env}",
                f"OTEL_GATEWAY_CA_FILE={secret_paths['ca.pem']}",
                (f"OTEL_GATEWAY_SERVER_CERTIFICATE_FILE={secret_paths['server.pem']}"),
                f"OTEL_GATEWAY_SERVER_KEY_FILE={secret_paths['server.key']}",
                f"OTEL_GATEWAY_RECEIVER_TOKEN_FILE={secret_paths['receiver.token']}",
                (
                    "OTEL_GATEWAY_LANGFUSE_TOKEN_FILE="
                    f"{secret_paths['langfuse-basic.token']}"
                ),
                "OTEL_GATEWAY_HEALTH_PORT=23133",
                "OTEL_GATEWAY_METRICS_PORT=28888",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return compose_env


def _resolved_gateway_configuration(tmp_path: Path) -> dict[str, Any]:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker CLI is unavailable")
    compose_env = _write_synthetic_gateway_files(tmp_path)
    environment = {
        "COMPOSE_DISABLE_ENV_FILE": "1",
        "HOME": os.environ.get("HOME", str(tmp_path)),
        "IMAGE": "agent:trace-gateway-contract",
        "LANG": "C",
        "PATH": os.environ["PATH"],
    }
    result = subprocess.run(  # noqa: S603 - resolved Docker, fixed arguments
        [
            docker,
            "compose",
            "--project-name",
            "trace-gateway-contract",
            "--env-file",
            str(compose_env),
            "-f",
            str(BASE_COMPOSE_PATH),
            "-f",
            str(GATEWAY_COMPOSE_PATH),
            "config",
            "--format",
            "json",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    configuration = yaml.safe_load(result.stdout)
    assert isinstance(configuration, dict)
    return configuration


def test_collector_pipeline_is_fail_closed_and_trace_only() -> None:
    """Keep one exact lossy privacy pipeline with no debug payload exporter."""
    configuration = _load_collector_configuration()
    receiver = configuration["receivers"]["otlp"]["protocols"]["http"]
    processors = configuration["processors"]
    pipeline = configuration["service"]["pipelines"]["traces"]

    assert set(configuration["receivers"]) == {"otlp"}
    assert set(configuration["exporters"]) == {"otlp_http/langfuse"}
    assert set(configuration["service"]["pipelines"]) == {"traces"}
    assert set(configuration["receivers"]["otlp"]["protocols"]) == {"http"}
    assert receiver == {
        "endpoint": "0.0.0.0:4318",
        "auth": {"authenticator": "bearertokenauth/gateway"},
        "tls": {
            "cert_file": "/run/secrets/otel_gateway_server_certificate",
            "key_file": "/run/secrets/otel_gateway_server_key",
            "min_version": "1.2",
        },
    }
    assert pipeline == {
        "receivers": ["otlp"],
        "processors": [
            "memory_limiter",
            "filter/privacy",
            "transform/privacy",
            "redaction/privacy",
            "groupbyattrs/privacy",
            "batch",
        ],
        "exporters": ["otlp_http/langfuse"],
    }
    assert processors["filter/privacy"] == {
        "error_mode": "propagate",
        "traces": {
            "span": ["Len(span.links) > 0"],
            "spanevent": ["true"],
        },
    }
    assert processors["memory_limiter"] == {
        "check_interval": "1s",
        "limit_mib": 192,
        "spike_limit_mib": 48,
    }
    assert processors["groupbyattrs/privacy"] == {}
    assert "debug" not in configuration["exporters"]
    assert configuration["service"]["telemetry"]["logs"]["level"] == "info"


def test_collector_transform_has_exact_safe_shape() -> None:
    """Deny unknown fields while retaining structural operation metadata."""
    configuration = _load_collector_configuration()
    processors = configuration["processors"]
    statement_groups = processors["transform/privacy"]["trace_statements"]
    statements_by_context = {
        statement_group["context"]: statement_group["statements"]
        for statement_group in statement_groups
    }
    span_statements = statements_by_context["span"]

    assert processors["transform/privacy"]["error_mode"] == "propagate"
    assert set(statements_by_context) == {"resource", "scope", "span"}
    assert statements_by_context["resource"] == [
        'delete_matching_keys(resource.attributes, ".*")',
        ('set(resource.attributes["service.name"], "google-adk-agent")'),
        ('set(resource.attributes["service.namespace"], "google-adk-on-bare-metal")'),
    ]
    assert statements_by_context["scope"] == [
        'delete_matching_keys(scope.attributes, ".*")',
        'set(scope.name, "google-adk")',
        'set(scope.version, "")',
        "set(scope.dropped_attributes_count, 0)",
    ]
    assert 'set(span.name, "agent.operation")' in span_statements
    assert 'set(span.status.message, "")' in span_statements
    assert 'set(span.trace_state, "")' in span_statements
    assert "set(span.flags, 0)" in span_statements
    assert "set(span.dropped_attributes_count, 0)" in span_statements
    assert "set(span.dropped_events_count, 0)" in span_statements
    assert "set(span.dropped_links_count, 0)" in span_statements
    assert 'delete_matching_keys(span.attributes, ".*")' in span_statements
    cache_names = (
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "system_instruction_tokens",
    )
    for attribute, cache_name in zip(
        SAFE_SPAN_ATTRIBUTES,
        cache_names,
        strict=True,
    ):
        assert any(
            statement.startswith(
                f'set(span.cache["{cache_name}"], span.attributes["{attribute}"])'
            )
            and "IsInt" in statement
            for statement in span_statements
        )
        assert (
            f'set(span.attributes["{attribute}"], '
            f'span.cache["{cache_name}"]) where '
            f'span.cache["{cache_name}"] != nil'
        ) in span_statements

    assert processors["redaction/privacy"] == {
        "allow_all_keys": False,
        "allowed_keys": [
            "service.name",
            "service.namespace",
            *SAFE_SPAN_ATTRIBUTES,
        ],
        "summary": "silent",
    }


def test_langfuse_export_is_collector_owned_and_bounded() -> None:
    """Keep exact endpoint/auth ownership and bounded in-memory failure behavior."""
    configuration = _load_collector_configuration()
    extensions = configuration["extensions"]
    exporter = configuration["exporters"]["otlp_http/langfuse"]

    assert extensions["bearertokenauth/gateway"] == {
        "scheme": "Bearer",
        "filename": "/run/secrets/otel_gateway_receiver_token",
    }
    assert extensions["bearertokenauth/langfuse"] == {
        "scheme": "Basic",
        "filename": "/run/secrets/otel_gateway_langfuse_token",
    }
    assert exporter == {
        "traces_endpoint": (
            "https://${env:OTEL_GATEWAY_LANGFUSE_AUTHORITY}/api/public/otel/v1/traces"
        ),
        "headers": {"x-langfuse-ingestion-version": "4"},
        "auth": {"authenticator": "bearertokenauth/langfuse"},
        "tls": {"insecure": False, "min_version": "1.2"},
        "timeout": "2s",
        "retry_on_failure": {
            "enabled": True,
            "initial_interval": "1s",
            "max_interval": "5s",
            "max_elapsed_time": "30s",
        },
        "sending_queue": {
            "enabled": True,
            "num_consumers": 1,
            "queue_size": 256,
        },
    }
    assert "storage" not in exporter["sending_queue"]


def test_resolved_gateway_isolated_and_has_no_agent_vendor_fallback(
    tmp_path: Path,
) -> None:
    """Resolve real Compose and prove credentials and networks stay partitioned."""
    configuration = _resolved_gateway_configuration(tmp_path)
    services = configuration["services"]
    agent = services["agent"]
    collector = services["otel-collector"]
    agent_environment = agent["environment"]
    collector_environment = collector["environment"]

    assert agent_environment["AGENT_NAME"] == "resolved-gateway-agent"
    assert agent_environment["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] == (
        "https://otel-collector:4318/v1/traces"
    )
    assert agent_environment["OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE"] == (
        "/run/secrets/otel_gateway_ca"
    )
    assert agent_environment["OTEL_GATEWAY_BEARER_TOKEN_FILE"] == str(
        Path("/run/secrets") / "otel_gateway_receiver_token"
    )
    assert agent_environment["OTEL_EXPORTER_OTLP_TRACES_HEADERS"] == ""
    assert {
        key: value
        for key, value in agent_environment.items()
        if key.startswith("LANGFUSE_")
    } == {
        "LANGFUSE_BASE_URL": "",
        "LANGFUSE_PUBLIC_KEY": "",
        "LANGFUSE_SECRET_KEY": "",
    }
    assert not any(
        "vendor-fallback" in str(value) or "hostile-" in str(value)
        for value in agent_environment.values()
    )
    assert agent_environment["OTEL_PYTHON_EXPORTER_OTLP_HTTP_CREDENTIAL_PROVIDER"] == ""
    assert (
        agent_environment["OTEL_PYTHON_EXPORTER_OTLP_HTTP_TRACES_CREDENTIAL_PROVIDER"]
        == ""
    )
    assert agent["group_add"] == ["10002"]
    assert collector_environment == {
        "OTEL_GATEWAY_LANGFUSE_AUTHORITY": "langfuse.example.test"
    }
    assert set(agent["networks"]) == {"default", "otel_gateway_telemetry"}
    assert set(collector["networks"]) == {
        "otel_gateway_telemetry",
        "otel_gateway_egress",
    }
    assert configuration["networks"]["otel_gateway_telemetry"]["internal"] is True
    assert (
        configuration["networks"]["otel_gateway_egress"].get("internal", False) is False
    )
    assert "depends_on" not in agent
    assert {secret["source"] for secret in agent["secrets"]} == {
        "otel_gateway_ca",
        "otel_gateway_receiver_token",
    }
    assert "otel_gateway_langfuse_token" not in {
        secret["source"] for secret in agent["secrets"]
    }


def test_resolved_collector_is_pinned_hardened_and_host_closed(
    tmp_path: Path,
) -> None:
    """Keep the optional second process bounded and publish no OTLP receiver."""
    collector = _resolved_gateway_configuration(tmp_path)["services"]["otel-collector"]

    assert collector["image"] == COLLECTOR_IMAGE
    assert collector["group_add"] == ["10002"]
    assert collector["read_only"] is True
    assert collector["user"] == "10001:10001"
    assert collector["cap_drop"] == ["ALL"]
    assert collector["security_opt"] == ["no-new-privileges:true"]
    assert collector["pids_limit"] == 100
    assert collector["deploy"]["resources"]["limits"] == {
        "cpus": 0.5,
        "memory": "268435456",
        "pids": 100,
    }
    assert collector["stop_grace_period"] == "10s"
    assert collector["restart"] == "always"
    assert {
        (port["target"], port["published"], port["host_ip"])
        for port in collector["ports"]
    } == {
        (13133, "23133", "127.0.0.1"),
        (8888, "28888", "127.0.0.1"),
    }
    assert all(port["target"] != 4318 for port in collector["ports"])
    assert "healthcheck" not in collector


def test_gateway_examples_and_runbook_keep_secrets_out_of_source() -> None:
    """Make the operator handoff explicit without shipping credential values."""
    agent_example = AGENT_ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
    collector_example = COLLECTOR_ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
    compose_example = COMPOSE_ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
    certificate_extension = CERTIFICATE_EXTENSION_PATH.read_text(encoding="utf-8")
    guide = OBSERVABILITY_GUIDE_PATH.read_text(encoding="utf-8")
    normalized_guide = " ".join(guide.split())
    decision = ARCHITECTURE_DECISION_PATH.read_text(encoding="utf-8")
    setup_script = SETUP_SCRIPT_PATH.read_text(encoding="utf-8")
    agent_assignment_names = {
        line.partition("=")[0]
        for line in agent_example.splitlines()
        if line and not line.startswith("#")
    }

    assert not any(name.startswith("LANGFUSE_") for name in agent_assignment_names)
    assert "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT" not in agent_assignment_names
    assert "LANGFUSE_SECRET_KEY" not in collector_example
    assert "LANGFUSE_PUBLIC_KEY" not in collector_example
    assert "OTEL_GATEWAY_LANGFUSE_AUTHORITY=cloud.langfuse.com" in collector_example
    assert "OTEL_GATEWAY_LANGFUSE_TRACES_ENDPOINT" not in collector_example
    assert "OTEL_GATEWAY_LANGFUSE_TOKEN_FILE=" in compose_example
    assert "OTEL_GATEWAY_SECRET_GID=10002" in compose_example
    assert "subjectAltName=DNS:otel-collector" in certificate_extension
    assert "COMPOSE_DISABLE_ENV_FILE=1 docker compose" in guide
    assert "validate --config=file:/etc/otelcol-contrib/config.yaml" in guide
    assert "Docker Compose 2.24.4 or newer" in normalized_guide
    assert "getent group otel-gateway" in guide
    assert "id -nG | tr ' ' '\\n' | grep -qx otel-gateway" in guide
    assert "chmod 640" in guide
    assert "OTEL_GATEWAY_SECRET_GID" in guide
    assert "single terminal newline" in guide
    assert "deprecated field-1000 scope envelope" in guide
    assert "deprecated field-1000 scope envelope" in decision
    assert "Upgrade the Collector in its own pull request." in guide
    assert "update the tag and digest together" in guide
    assert "hosted TLS/redaction canary" in guide
    assert "do not substitute a moving tag" in guide
    assert (
        "RUN_TRACE_GATEWAY_INTEGRATION=1 \\\n"
        "  uv run pytest tests/test_trace_gateway_runtime.py -q"
    ) in guide
    assert "sudo chgrp" not in guide
    assert "Raw values can still exist in" in guide
    assert "host-loopback Collector was rejected" in decision
    assert "256 MiB" in decision
    assert 'OTEL_GATEWAY_GROUP="otel-gateway"' in setup_script
    assert 'groupadd --system "$OTEL_GATEWAY_GROUP"' in setup_script
    assert 'usermod -aG "docker,$OTEL_GATEWAY_GROUP" "$AGENT_USER"' in setup_script
    assert "\nsecrets/otel-gateway/\n" in (
        f"\n{GITIGNORE_PATH.read_text(encoding='utf-8')}\n"
    )

    git = shutil.which("git")
    assert git is not None
    ignore_result = subprocess.run(  # noqa: S603 - resolved Git, fixed arguments
        [
            git,
            "check-ignore",
            "--quiet",
            "--no-index",
            "secrets/otel-gateway/server.key",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    assert ignore_result.returncode == 0


def test_compose_ci_runs_opt_in_real_gateway_canary() -> None:
    """Require one bounded, secret-free hosted proof on every main change."""
    workflow = COMPOSE_WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "  trace-gateway:\n" in workflow
    assert "name: Validate trace-redaction gateway" in workflow
    assert "timeout-minutes: 30" in workflow
    assert 'RUN_TRACE_GATEWAY_INTEGRATION: "1"' in workflow
    assert (
        "TRACE_GATEWAY_TEST_PROJECT: >-\n"
        "        adk-gateway-${{ github.run_id }}-${{ github.run_attempt }}"
    ) in workflow
    assert (
        "astral-sh/setup-uv@d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86 # v5"
    ) in workflow
    assert "run: uv python install 3.13" in workflow
    assert "run: uv sync --locked" in workflow
    assert "run: uv run pytest tests/test_trace_gateway_runtime.py -q" in workflow
    assert "pull_request_target" not in workflow
    assert "secrets." not in workflow


def test_root_setup_provisions_unprivileged_gateway_secret_group() -> None:
    """Keep the documented agent-runner path executable without runtime sudo."""
    setup_script = SETUP_SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'OTEL_GATEWAY_GROUP="otel-gateway"' in setup_script
    group_position = setup_script.index('groupadd --system "$OTEL_GATEWAY_GROUP"')
    user_position = setup_script.index('AGENT_USER="agent-runner"')
    membership_position = setup_script.index(
        'usermod -aG "docker,$OTEL_GATEWAY_GROUP" "$AGENT_USER"'
    )
    completion_position = setup_script.index('log "Setup complete!')

    assert user_position < group_position < membership_position < completion_position

    syntax = subprocess.run(  # noqa: S603 - fixed repository shell program
        ["/bin/bash", "-n", str(SETUP_SCRIPT_PATH)],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr
