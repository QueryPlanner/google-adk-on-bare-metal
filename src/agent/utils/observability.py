"""OpenTelemetry and Logging setup for bare-metal adaptation.

This module provides consolidated observability configuration.
"""

import base64
import logging
import os
import sys
import threading
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any, cast
from urllib.parse import quote

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.sdk.resources import (
    SERVICE_INSTANCE_ID,
    SERVICE_NAME,
    SERVICE_NAMESPACE,
    SERVICE_VERSION,
)
from starlette.types import Lifespan

from .config import ObservabilityEnv

OTEL_FORCE_FLUSH_TIMEOUT_MILLIS = 5_000
_OTLP_TRACE_ENVIRONMENT_KEYS = (
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
    "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
    "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT",
)

logger = logging.getLogger(__name__)


def configure_otel_resource(
    agent_name: str,
    settings: ObservabilityEnv,
) -> None:
    """Configure OpenTelemetry resource via environment variables.

    Materialize resource, capture, and validated trace-specific OTLP variables.
    Google ADK remains the sole owner of the global OpenTelemetry providers.

    Args:
        agent_name: Unique service identifier.
        settings: Validated observability settings.
    """
    print("🔭 Setting OpenTelemetry Resource attributes environment variable...")
    instance_id = f"worker-{os.getpid()}-{uuid.uuid4().hex}"
    resource_attributes = {
        SERVICE_INSTANCE_ID: instance_id,
        SERVICE_NAME: agent_name,
        SERVICE_NAMESPACE: settings.telemetry_namespace,
        SERVICE_VERSION: settings.service_revision,
    }
    os.environ["OTEL_RESOURCE_ATTRIBUTES"] = ",".join(
        f"{key}={quote(value, safe='')}" for key, value in resource_attributes.items()
    )
    os.environ.pop("OTEL_SERVICE_NAME", None)
    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = str(
        settings.otel_capture_message_content
    ).lower()
    os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] = str(
        settings.otel_capture_message_content
    ).lower()

    for environment_key in _OTLP_TRACE_ENVIRONMENT_KEYS:
        os.environ.pop(environment_key, None)

    explicit_endpoint = settings.otel_exporter_otlp_traces_endpoint
    if explicit_endpoint is not None:
        os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = explicit_endpoint
        os.environ["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] = "http/protobuf"
        os.environ["OTEL_EXPORTER_OTLP_TRACES_TIMEOUT"] = str(
            settings.effective_otel_exporter_otlp_traces_timeout
        )
        explicit_headers = settings.otel_exporter_otlp_traces_headers
        if explicit_headers is not None:
            os.environ["OTEL_EXPORTER_OTLP_TRACES_HEADERS"] = (
                explicit_headers.get_secret_value()
            )
        return

    public_key = settings.langfuse_public_key
    secret_key = settings.langfuse_secret_key
    if public_key and secret_key:
        print("💡 Langfuse keys detected. Configuring OTLP exporter...")
        base_url = settings.effective_langfuse_base_url.rstrip("/")
        auth_string = f"{public_key.get_secret_value()}:{secret_key.get_secret_value()}"
        encoded_auth = base64.b64encode(auth_string.encode()).decode()
        authorization = quote(f"Basic {encoded_auth}", safe="")
        os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = (
            f"{base_url}/api/public/otel/v1/traces"
        )
        os.environ["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] = "http/protobuf"
        os.environ["OTEL_EXPORTER_OTLP_TRACES_TIMEOUT"] = str(
            settings.effective_otel_exporter_otlp_traces_timeout
        )
        os.environ["OTEL_EXPORTER_OTLP_TRACES_HEADERS"] = (
            f"Authorization={authorization},x-langfuse-ingestion-version=4"
        )
        print("✅ Langfuse OTLP exporter configured.")


def install_otel_flush_lifespan(app: FastAPI) -> None:
    """Flush ADK-owned queued spans after its application cleanup completes."""
    adk_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def otel_flush_lifespan(
        app_instance: FastAPI,
    ) -> AsyncIterator[Mapping[str, Any] | None]:
        try:
            async with adk_lifespan(app_instance) as state:
                yield state
        finally:
            _force_flush_traces()

    app.router.lifespan_context = cast(Lifespan[FastAPI], otel_flush_lifespan)


def _force_flush_traces() -> None:
    """Bound one final flush without replacing or shutting down the provider."""
    provider = trace.get_tracer_provider()
    force_flush = getattr(provider, "force_flush", None)
    if not callable(force_flush):
        return

    flush_finished = threading.Event()
    flush_failed = False
    flush_completed: bool | None = None

    def run_force_flush() -> None:
        """Run a potentially blocking SDK flush outside the application loop."""
        nonlocal flush_completed, flush_failed
        try:
            flush_completed = force_flush(
                timeout_millis=OTEL_FORCE_FLUSH_TIMEOUT_MILLIS
            )
        except Exception:
            flush_failed = True
        finally:
            flush_finished.set()

    flush_thread = threading.Thread(
        target=run_force_flush,
        name="otel-force-flush",
        daemon=True,
    )
    flush_thread.start()
    if not flush_finished.wait(OTEL_FORCE_FLUSH_TIMEOUT_MILLIS / 1_000):
        logger.warning("OpenTelemetry trace flush timed out.")
        return

    if flush_failed:
        logger.warning("OpenTelemetry trace flush failed.")
    elif flush_completed is False:
        logger.warning("OpenTelemetry trace flush timed out.")


def setup_logging(log_level: str) -> None:
    """Set up basic logging.

    Args:
        log_level: Logging verbosity level as string
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set levels for some noisy libraries if needed
    logging.getLogger("urllib3").setLevel(logging.WARNING)
