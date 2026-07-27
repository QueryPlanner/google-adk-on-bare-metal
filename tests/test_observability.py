"""Tests for explicit OpenTelemetry environment composition."""

import base64
import os
from pathlib import Path

import pytest

from agent.utils.config import ObservabilityEnv
from agent.utils.observability import configure_otel_resource


@pytest.fixture(autouse=True)
def isolate_observability_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Prevent observability tests from reading the repository dotenv."""
    monkeypatch.chdir(tmp_path)


def test_generic_otel_settings_are_published_without_disclosure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test explicit vendor-neutral settings are materialized for the SDK."""
    header = "Authorization=generic-header-secret-canary"
    monkeypatch.setenv("UNRELATED_SENTINEL", "preserved")
    settings = ObservabilityEnv.model_validate(
        {
            "TELEMETRY_NAMESPACE": "test-namespace",
            "K_REVISION": "test-revision",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "https://otel.example.test",
            "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
            "OTEL_EXPORTER_OTLP_HEADERS": header,
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true",
        }
    )

    configure_otel_resource("test-agent", settings)

    resource_attributes = os.environ["OTEL_RESOURCE_ATTRIBUTES"]
    assert "service.name=test-agent" in resource_attributes
    assert "service.namespace=test-namespace" in resource_attributes
    assert "service.version=test-revision" in resource_attributes
    assert "service.instance.id=worker-" in resource_attributes
    assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "https://otel.example.test"
    assert os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] == "http/protobuf"
    assert os.environ["OTEL_EXPORTER_OTLP_HEADERS"] == header
    assert os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] == "true"
    assert os.environ["UNRELATED_SENTINEL"] == "preserved"
    assert header not in capsys.readouterr().out


def test_langfuse_derives_only_missing_otel_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test complete Langfuse credentials derive a safe OTLP configuration."""
    public_key = "pk-lf-public-secret-canary"
    secret_key = "sk-lf-private-secret-canary"  # noqa: S105
    settings = ObservabilityEnv.model_validate(
        {
            "LANGFUSE_PUBLIC_KEY": public_key,
            "LANGFUSE_SECRET_KEY": secret_key,
            "LANGFUSE_BASE_URL": "https://langfuse.example.test/",
        }
    )
    expected_auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()

    configure_otel_resource("test-agent", settings)

    assert (
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
        == "https://langfuse.example.test/api/public/otel"
    )
    assert os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] == "http/protobuf"
    assert (
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"]
        == f"Authorization=Basic {expected_auth}"
    )
    output = capsys.readouterr().out
    assert public_key not in output
    assert secret_key not in output
    assert expected_auth not in output


def test_explicit_otel_values_win_over_langfuse_derivation() -> None:
    """Test explicit OTLP fields outrank Langfuse convenience defaults."""
    explicit_header = "Authorization=explicit-secret-canary"
    settings = ObservabilityEnv.model_validate(
        {
            "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
            "LANGFUSE_SECRET_KEY": "sk-lf-test",
            "LANGFUSE_BASE_URL": "https://langfuse.example.test",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "https://collector.example.test",
            "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
            "OTEL_EXPORTER_OTLP_HEADERS": explicit_header,
        }
    )

    configure_otel_resource("test-agent", settings)

    assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "https://collector.example.test"
    assert os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] == "grpc"
    assert os.environ["OTEL_EXPORTER_OTLP_HEADERS"] == explicit_header


def test_dotenv_blank_otel_values_allow_langfuse_derivation(
    tmp_path: Path,
) -> None:
    """Test blank optional dotenv entries are normalized after selection."""
    public_key = "pk-lf-dotenv-blank"
    secret_key = "sk-lf-dotenv-blank"  # noqa: S105
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                f"LANGFUSE_PUBLIC_KEY={public_key}",
                f"LANGFUSE_SECRET_KEY={secret_key}",
                "LANGFUSE_BASE_URL=https://dotenv.langfuse.test/",
                "OTEL_EXPORTER_OTLP_ENDPOINT=",
                "OTEL_EXPORTER_OTLP_PROTOCOL=",
                "OTEL_EXPORTER_OTLP_HEADERS=",
                "",
            ]
        ),
        encoding="utf-8",
    )
    settings = ObservabilityEnv()
    expected_auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()

    configure_otel_resource("test-agent", settings)

    assert (
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
        == "https://dotenv.langfuse.test/api/public/otel"
    )
    assert os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] == "http/protobuf"
    assert (
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"]
        == f"Authorization=Basic {expected_auth}"
    )


def test_process_blank_otel_values_block_dotenv_and_derive_langfuse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test blank process values do not fall through to explicit dotenv OTLP."""
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "LANGFUSE_BASE_URL=https://dotenv.langfuse.test",
                "OTEL_EXPORTER_OTLP_ENDPOINT=https://dotenv.collector.test",
                "OTEL_EXPORTER_OTLP_PROTOCOL=grpc",
                "OTEL_EXPORTER_OTLP_HEADERS=dotenv-header",
                "",
            ]
        ),
        encoding="utf-8",
    )
    public_key = "pk-lf-process-blank"
    secret_key = "sk-lf-process-blank"  # noqa: S105
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", public_key)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", secret_key)
    monkeypatch.setenv("LANGFUSE_BASE_URL", " ")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", " ")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "\t")
    settings = ObservabilityEnv()
    expected_auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()

    assert settings.otel_exporter_otlp_endpoint is None
    assert settings.otel_exporter_otlp_protocol is None
    assert settings.otel_exporter_otlp_headers is None

    configure_otel_resource("test-agent", settings)

    assert (
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
        == "https://cloud.langfuse.com/api/public/otel"
    )
    assert os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] == "http/protobuf"
    assert (
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"]
        == f"Authorization=Basic {expected_auth}"
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("LANGFUSE_PUBLIC_KEY", "pk-lf-incomplete"),
        ("LANGFUSE_SECRET_KEY", "sk-lf-incomplete"),
    ],
)
def test_incomplete_langfuse_credentials_do_not_enable_export(
    field_name: str,
    value: str,
) -> None:
    """Test one Langfuse key never creates a partially authenticated exporter."""
    settings = ObservabilityEnv.model_validate({field_name: value})

    configure_otel_resource("test-agent", settings)

    assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in os.environ
    assert "OTEL_EXPORTER_OTLP_PROTOCOL" not in os.environ
    assert "OTEL_EXPORTER_OTLP_HEADERS" not in os.environ
    assert os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] == "false"
