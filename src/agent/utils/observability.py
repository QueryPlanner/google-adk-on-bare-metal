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


def configure_otel_resource(agent_name: str) -> None:
    """Configure OpenTelemetry resource via environment variables.

    Additionally configures Langfuse OTLP exporter if credentials are provided.

    Args:
        agent_name: Unique service identifier
    """
    print("🔭 Setting OpenTelemetry Resource attributes environment variable...")
    instance_id = f"worker-{os.getpid()}-{uuid.uuid4().hex}"
    os.environ["OTEL_RESOURCE_ATTRIBUTES"] = (
        f"{SERVICE_INSTANCE_ID}={instance_id},"
        f"{SERVICE_NAME}={agent_name},"
        f"{SERVICE_NAMESPACE}={os.getenv('TELEMETRY_NAMESPACE', 'local')},"
        f"{SERVICE_VERSION}={os.getenv('K_REVISION', 'local')}"
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
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if public_key and secret_key:
        print("💡 Langfuse keys detected. Configuring OTLP exporter...")

        # 1. Set Endpoint (default to EU if not specified)
        base_url = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com").rstrip(
            "/"
        )
        if "OTEL_EXPORTER_OTLP_ENDPOINT" not in os.environ:
            os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{base_url}/api/public/otel"

        # 2. Generate Auth Header
        auth_str = f"{public_key}:{secret_key}"
        encoded_auth = base64.b64encode(auth_str.encode("utf-8")).decode("utf-8")
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {encoded_auth}"

        # 3. Ensure Protocol is set to http/protobuf (required by Langfuse)
        if "OTEL_EXPORTER_OTLP_PROTOCOL" not in os.environ:
            os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"

        endpoint = os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
        print(f"✅ Langfuse OTLP configured for endpoint: {endpoint}")


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
