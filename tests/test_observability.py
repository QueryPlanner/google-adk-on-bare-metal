"""Tests for private trace-only OpenTelemetry composition and shutdown."""

import base64
import logging
import os
import threading
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.sdk.resources import (
    SERVICE_INSTANCE_ID,
    SERVICE_NAME,
    SERVICE_NAMESPACE,
    SERVICE_VERSION,
    OTELResourceDetector,
)
from opentelemetry.util.re import parse_env_headers

from agent.utils.config import ObservabilityEnv
from agent.utils.observability import (
    OTEL_FORCE_FLUSH_TIMEOUT_MILLIS,
    _force_flush_traces,
    configure_otel_resource,
    install_otel_flush_lifespan,
    setup_logging,
)

_OBSERVABILITY_ENVIRONMENT_KEYS = (
    "ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
    "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
    "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT",
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
    "OTEL_RESOURCE_ATTRIBUTES",
    "OTEL_SERVICE_NAME",
)


@pytest.fixture(autouse=True)
def isolate_observability_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[None]:
    """Prevent dotenv reads and restore env changes made by production code."""
    monkeypatch.chdir(tmp_path)
    original_values = {
        key: os.environ.get(key) for key in _OBSERVABILITY_ENVIRONMENT_KEYS
    }
    for key in _OBSERVABILITY_ENVIRONMENT_KEYS:
        os.environ.pop(key, None)

    yield

    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


def test_explicit_trace_settings_are_materialized_without_generic_export(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Publish only validated HTTP trace settings and both capture controls."""
    header = "Authorization=Bearer%20header-secret-canary"
    settings = ObservabilityEnv.model_validate(
        {
            "TELEMETRY_NAMESPACE": "test-namespace",
            "K_REVISION": "test-revision",
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": (
                "https://collector.example.test/otel/v1/traces"
            ),
            "OTEL_EXPORTER_OTLP_TRACES_HEADERS": header,
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true",
        }
    )
    os.environ["UNRELATED_SENTINEL"] = "preserved"

    configure_otel_resource("test-agent", settings)

    resource_attributes = os.environ["OTEL_RESOURCE_ATTRIBUTES"]
    assert "service.name=test-agent" in resource_attributes
    assert "service.namespace=test-namespace" in resource_attributes
    assert "service.version=test-revision" in resource_attributes
    assert "service.instance.id=worker-" in resource_attributes
    assert os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] == (
        "https://collector.example.test/otel/v1/traces"
    )
    assert os.environ["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] == "http/protobuf"
    assert os.environ["OTEL_EXPORTER_OTLP_TRACES_HEADERS"] == header
    assert os.environ["OTEL_EXPORTER_OTLP_TRACES_TIMEOUT"] == "2.0"
    assert os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] == "true"
    assert os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] == "true"
    assert os.environ["UNRELATED_SENTINEL"] == "preserved"
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in os.environ
    assert "OTEL_EXPORTER_OTLP_PROTOCOL" not in os.environ
    assert "OTEL_EXPORTER_OTLP_HEADERS" not in os.environ
    assert header not in capsys.readouterr().out
    os.environ.pop("UNRELATED_SENTINEL")


def test_disabled_export_clears_stale_trace_settings_and_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid configuration leakage across repeated in-process app factories."""
    settings = ObservabilityEnv()
    for key in (
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
        "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT",
    ):
        monkeypatch.setenv(key, "stale-secret-canary")
    monkeypatch.setenv("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", "true")

    configure_otel_resource("test-agent", settings)

    assert "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT" not in os.environ
    assert "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL" not in os.environ
    assert "OTEL_EXPORTER_OTLP_TRACES_HEADERS" not in os.environ
    assert "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT" not in os.environ
    assert os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] == "false"
    assert os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] == "false"


def test_resource_values_cannot_inject_attributes_or_control_characters() -> None:
    """Percent-encode validated resource values before ADK parses them."""
    agent_name = "agent,forged.agent=value"
    namespace = "namespace=blue,forged.namespace=value"
    revision = "revision\nforged.revision=value"
    settings = ObservabilityEnv.model_validate(
        {
            "TELEMETRY_NAMESPACE": namespace,
            "K_REVISION": revision,
        }
    )
    os.environ["OTEL_SERVICE_NAME"] = "stale-service-name"

    configure_otel_resource(agent_name, settings)

    serialized_attributes = os.environ["OTEL_RESOURCE_ATTRIBUTES"]
    detected_attributes = OTELResourceDetector().detect().attributes
    assert "\n" not in serialized_attributes
    assert "OTEL_SERVICE_NAME" not in os.environ
    assert detected_attributes[SERVICE_NAME] == agent_name
    assert detected_attributes[SERVICE_NAMESPACE] == namespace
    assert detected_attributes[SERVICE_VERSION] == revision
    assert str(detected_attributes[SERVICE_INSTANCE_ID]).startswith("worker-")
    assert set(detected_attributes) == {
        SERVICE_INSTANCE_ID,
        SERVICE_NAME,
        SERVICE_NAMESPACE,
        SERVICE_VERSION,
    }


def test_explicit_endpoint_without_headers_stays_header_free() -> None:
    """Do not invent authentication for a header-free explicit collector."""
    settings = ObservabilityEnv.model_validate(
        {
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": (
                "https://collector.example.test/v1/traces"
            )
        }
    )

    configure_otel_resource("test-agent", settings)

    assert "OTEL_EXPORTER_OTLP_TRACES_HEADERS" not in os.environ
    assert os.environ["OTEL_EXPORTER_OTLP_TRACES_TIMEOUT"] == "2.0"


@pytest.mark.parametrize(
    ("base_url", "expected_endpoint"),
    [
        (
            None,
            "https://cloud.langfuse.com/api/public/otel/v1/traces",
        ),
        (
            "https://langfuse.example.test/prefix/",
            "https://langfuse.example.test/prefix/api/public/otel/v1/traces",
        ),
    ],
)
def test_langfuse_derives_trace_endpoint_and_v4_headers(
    base_url: str | None,
    expected_endpoint: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Derive the current Langfuse OTLP/HTTP contract without disclosure."""
    public_key = "pk-lf-public-secret-canary"
    secret_key = "sk-lf-private-secret-canary"  # noqa: S105
    values = {
        "LANGFUSE_PUBLIC_KEY": public_key,
        "LANGFUSE_SECRET_KEY": secret_key,
        "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT": "1.25",
    }
    if base_url is not None:
        values["LANGFUSE_BASE_URL"] = base_url
    settings = ObservabilityEnv.model_validate(values)

    configure_otel_resource("test-agent", settings)

    parsed_headers = parse_env_headers(os.environ["OTEL_EXPORTER_OTLP_TRACES_HEADERS"])
    assert os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] == expected_endpoint
    assert os.environ["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] == "http/protobuf"
    assert os.environ["OTEL_EXPORTER_OTLP_TRACES_TIMEOUT"] == "1.25"
    assert set(parsed_headers) == {
        "authorization",
        "x-langfuse-ingestion-version",
    }
    authorization_scheme, encoded_credentials = parsed_headers["authorization"].split(
        " ", maxsplit=1
    )
    assert authorization_scheme == "Basic"
    assert base64.b64decode(encoded_credentials).decode() == (
        f"{public_key}:{secret_key}"
    )
    assert parsed_headers["x-langfuse-ingestion-version"] == "4"
    output = capsys.readouterr().out
    assert public_key not in output
    assert secret_key not in output
    assert parsed_headers["authorization"] not in output


def test_setup_logging_uses_info_fallback_and_quiets_urllib3() -> None:
    """Use a stable fallback for an unexpected programmatic log level."""
    with patch("agent.utils.observability.logging.basicConfig") as basic_config:
        setup_logging("not-a-level")

    assert basic_config.call_args.kwargs["level"] == logging.INFO
    assert logging.getLogger("urllib3").level == logging.WARNING


def test_outer_lifespan_flushes_after_adk_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve ADK cleanup ordering before the final provider flush."""
    events: list[object] = []

    @asynccontextmanager
    async def adk_lifespan(app: FastAPI) -> Any:
        events.append(("adk-startup", app))
        try:
            yield {"ready": True}
        finally:
            events.append("adk-cleanup")

    class RecordingProvider:
        def force_flush(self, timeout_millis: int) -> bool:
            events.append(("force-flush", timeout_millis))
            return True

    app = FastAPI(lifespan=adk_lifespan)
    app.get("/")(lambda: {"ok": True})
    monkeypatch.setattr(
        "agent.utils.observability.trace.get_tracer_provider",
        lambda: RecordingProvider(),
    )
    install_otel_flush_lifespan(app)

    with TestClient(app) as client:
        assert client.get("/").json() == {"ok": True}
        assert events[0] == ("adk-startup", app)

    assert events[-2:] == [
        "adk-cleanup",
        ("force-flush", OTEL_FORCE_FLUSH_TIMEOUT_MILLIS),
    ]


@pytest.mark.parametrize(
    ("provider", "expected_message"),
    [
        (
            type(
                "FailedProvider",
                (),
                {"force_flush": lambda self, timeout_millis: False},
            )(),
            "OpenTelemetry trace flush timed out.",
        ),
        (
            type(
                "RaisingProvider",
                (),
                {
                    "force_flush": lambda self, timeout_millis: (_ for _ in ()).throw(
                        RuntimeError("flush-exception-secret-canary")
                    )
                },
            )(),
            "OpenTelemetry trace flush failed.",
        ),
    ],
)
def test_flush_failures_log_only_stable_messages(
    provider: object,
    expected_message: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Never leak provider exception or exporter material during shutdown."""
    monkeypatch.setattr(
        "agent.utils.observability.trace.get_tracer_provider",
        lambda: provider,
    )
    caplog.set_level(logging.WARNING, logger="agent.utils.observability")

    _force_flush_traces()

    records = [
        record
        for record in caplog.records
        if record.name == "agent.utils.observability"
    ]
    assert [record.getMessage() for record in records] == [expected_message]
    assert "flush-exception-secret-canary" not in caplog.text
    assert all(record.exc_info is None for record in records)


def test_provider_without_flush_is_a_noop(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Allow the API proxy provider when ADK has no active SDK provider."""
    monkeypatch.setattr(
        "agent.utils.observability.trace.get_tracer_provider",
        object,
    )

    _force_flush_traces()

    assert not caplog.records


def test_blocking_provider_cannot_exceed_outer_bound(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Return independently of an SDK processor that ignores its timeout."""
    release_flush = threading.Event()
    observed_timeout: list[int] = []

    class BlockingProvider:
        def force_flush(self, timeout_millis: int) -> bool:
            observed_timeout.append(timeout_millis)
            release_flush.wait()
            return True

    monkeypatch.setattr(
        "agent.utils.observability.trace.get_tracer_provider",
        lambda: BlockingProvider(),
    )
    monkeypatch.setattr(
        "agent.utils.observability.OTEL_FORCE_FLUSH_TIMEOUT_MILLIS",
        20,
    )
    caplog.set_level(logging.WARNING, logger="agent.utils.observability")

    started_at = time.monotonic()
    _force_flush_traces()
    elapsed = time.monotonic() - started_at
    release_flush.set()

    assert observed_timeout == [20]
    assert elapsed < 0.5
    assert "OpenTelemetry trace flush timed out." in caplog.messages
