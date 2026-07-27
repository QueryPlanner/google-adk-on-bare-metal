"""OpenTelemetry and Logging setup for bare-metal adaptation.

This module provides consolidated observability configuration.
"""

import base64
import logging
import os
import sys
import uuid

from opentelemetry.sdk.resources import (
    SERVICE_INSTANCE_ID,
    SERVICE_NAME,
    SERVICE_NAMESPACE,
    SERVICE_VERSION,
)

from .config import ObservabilityEnv


def configure_otel_resource(
    agent_name: str,
    settings: ObservabilityEnv,
) -> None:
    """Configure OpenTelemetry resource via environment variables.

    Materialize only the standard variables consumed by the OpenTelemetry SDK.
    Explicit OTLP settings win; missing values are derived from Langfuse when both
    Langfuse keys are configured.

    Args:
        agent_name: Unique service identifier.
        settings: Validated observability settings.
    """
    print("🔭 Setting OpenTelemetry Resource attributes environment variable...")
    instance_id = f"worker-{os.getpid()}-{uuid.uuid4().hex}"
    os.environ["OTEL_RESOURCE_ATTRIBUTES"] = (
        f"{SERVICE_INSTANCE_ID}={instance_id},"
        f"{SERVICE_NAME}={agent_name},"
        f"{SERVICE_NAMESPACE}={settings.telemetry_namespace},"
        f"{SERVICE_VERSION}={settings.service_revision}"
    )
    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = str(
        settings.otel_capture_message_content
    ).lower()

    if settings.otel_exporter_otlp_endpoint is not None:
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = settings.otel_exporter_otlp_endpoint
    if settings.otel_exporter_otlp_protocol is not None:
        os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = settings.otel_exporter_otlp_protocol
    if settings.otel_exporter_otlp_headers is not None:
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = (
            settings.otel_exporter_otlp_headers.get_secret_value()
        )

    # Automatically configure Langfuse if keys are present
    # -------------------------------------------------------------------------
    # VENDOR NEUTRALITY NOTE:
    # This block is a convenience helper for Langfuse.
    # To use a different OTLP backend (Jaeger, Honeycomb, etc.):
    #   1. Remove/Unset LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY.
    #   2. Set standard OTel env vars:
    #      - OTEL_EXPORTER_OTLP_ENDPOINT
    #      - OTEL_EXPORTER_OTLP_HEADERS (if auth is needed)
    #      - OTEL_EXPORTER_OTLP_PROTOCOL
    # -------------------------------------------------------------------------
    public_key = settings.langfuse_public_key
    secret_key = settings.langfuse_secret_key
    if public_key and secret_key:
        print("💡 Langfuse keys detected. Configuring OTLP exporter...")

        # 1. Derive an endpoint only when no explicit OTLP endpoint was supplied.
        base_url = settings.langfuse_base_url.rstrip("/")
        if settings.otel_exporter_otlp_endpoint is None:
            os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{base_url}/api/public/otel"

        # 2. Derive an auth header only when no explicit OTLP header was supplied.
        if settings.otel_exporter_otlp_headers is None:
            auth_str = (
                f"{public_key.get_secret_value()}:{secret_key.get_secret_value()}"
            )
            encoded_auth = base64.b64encode(auth_str.encode("utf-8")).decode("utf-8")
            os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = (
                f"Authorization=Basic {encoded_auth}"
            )

        # 3. Derive the required Langfuse protocol unless one was explicit.
        if settings.otel_exporter_otlp_protocol is None:
            os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"

        print("✅ Langfuse OTLP exporter configured.")


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
