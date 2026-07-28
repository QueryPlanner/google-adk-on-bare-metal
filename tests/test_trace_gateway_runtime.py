"""Real Docker proof for the optional private trace-redaction gateway."""

from __future__ import annotations

import base64
import gzip
import json
import os
import re
import shutil
import subprocess
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from opentelemetry.proto.trace.v1.trace_pb2 import Span, Status

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASE_COMPOSE_PATH = REPOSITORY_ROOT / "compose.yaml"
GATEWAY_COMPOSE_PATH = REPOSITORY_ROOT / "compose.trace-gateway.yaml"
COLLECTOR_IMAGE = (
    "ghcr.io/open-telemetry/opentelemetry-collector-releases/"
    "opentelemetry-collector-contrib:0.157.0@"
    "sha256:f2f01157055a9b2aab9df7118e1f1c9abf345e99b23bc7a2bc791db374a7d0f6"
)
MINIMUM_COMPOSE_VERSION = (2, 24, 4)
SAFE_RESOURCE_ATTRIBUTES = {
    "service.name": "google-adk-agent",
    "service.namespace": "google-adk-on-bare-metal",
}
SAFE_SPAN_ATTRIBUTES = {
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.experimental.reasoning_tokens",
    "gen_ai.usage.experimental.system_instruction_tokens",
}
SAFE_SPAN_NAMES = {
    "agent.cache",
    "agent.compaction",
    "agent.data",
    "agent.invoke",
    "agent.llm",
    "agent.model",
    "agent.operation",
    "agent.request",
    "agent.tool",
}
GATEWAY_TOKEN = "runtime-gateway-token-canary"  # noqa: S105
BASIC_PAYLOAD = base64.b64encode(b"runtime-public-canary:runtime-secret-canary").decode(
    "ascii"
)
SYNTHETIC_MODEL_KEY = "synthetic-runtime-model-boundary"  # noqa: S105
HOSTILE_ENV_CANARIES = (
    "hostile-public-canary",
    "hostile-secret-canary",
    "hostile-header",
    "hostile-provider",
    "hostile-trace-provider",
)
RAW_CANARIES = (
    "SECRET_TOOLNAME_CANARY",
    "TRACE_STATE_CANARY",
    "STATIC_DESCRIPTION_CANARY",
    "STRUCTURED_GENAI_CANARY",
    "CUSTOM_ATTRIBUTE_CANARY",
    "BAD_TOKEN_TYPE_CANARY",
    "EXCEPTION_MESSAGE_CANARY",
    "EXCEPTION_STACKTRACE_CANARY",
    "STATUS_MESSAGE_CANARY",
    "LINKED_SPAN_NAME_CANARY",
    "LINK_TRACE_STATE_CANARY",
    "LINK_ATTRIBUTE_CANARY",
    "UNKNOWN_SPAN_NAME_CANARY",
    "RESOURCE_NAME_CANARY",
    "RESOURCE_NAMESPACE_CANARY",
    "RESOURCE_VERSION_CANARY",
    "RESOURCE_PRIVATE_CANARY",
    "SCOPE_NAME_CANARY",
    "SCOPE_VERSION_CANARY",
    "SCOPE_ATTR_CANARY",
    "ENTITY_SCHEMA_CANARY",
    "ENTITY_TYPE_CANARY",
    "ENTITY_ID_KEY_CANARY",
    "ENTITY_DESCRIPTION_KEY_CANARY",
    "RESOURCE_SCHEMA_CANARY",
    "SCOPE_SCHEMA_CANARY",
    "DEPRECATED_SCOPE_CANARY",
    "SDK_NAME_CANARY",
    "SDK_PRIVATE_CANARY",
    "SDK_DESCRIPTION_CANARY",
    "SDK_EXCEPTION_CANARY",
    "SDK_STATUS_CANARY",
)
SENSITIVE_LOG_VALUES = (
    GATEWAY_TOKEN,
    BASIC_PAYLOAD,
    SYNTHETIC_MODEL_KEY,
    *HOSTILE_ENV_CANARIES,
    *RAW_CANARIES,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_TRACE_GATEWAY_INTEGRATION") != "1",
    reason="real trace-gateway Docker proof is opt-in",
)

CAPTURE_SERVER = textwrap.dedent(
    r"""
    from __future__ import annotations

    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import itertools
    import json
    from pathlib import Path
    import ssl

    OUTPUT = Path("/tmp/trace-capture")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    COUNTER = itertools.count(1)
    MAX_BODY = 16 * 1024 * 1024
    EXPECTED_AUTHORIZATION = (
        "Basic "
        + Path("/run/secrets/otel_gateway_downstream_token")
        .read_text(encoding="ascii")
        .strip()
    )


    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self) -> None:
            if self.path != "/api/public/otel/v1/traces":
                self.send_error(404)
                return
            if self.headers.get("Authorization") != EXPECTED_AUTHORIZATION:
                self.send_error(401)
                return
            if self.headers.get("x-langfuse-ingestion-version") != "4":
                self.send_error(400)
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > MAX_BODY:
                self.send_error(413)
                return
            body = self.rfile.read(content_length)
            index = next(COUNTER)
            stem = f"{index:06d}"
            (OUTPUT / f"{stem}.bin").write_bytes(body)
            metadata = {
                "authorization_ok": True,
                "content_encoding": self.headers.get("Content-Encoding"),
                "content_type": self.headers.get("Content-Type"),
                "ingestion_version": self.headers.get(
                    "x-langfuse-ingestion-version"
                ),
                "path": self.path,
            }
            (OUTPUT / f"{stem}.json").write_text(
                json.dumps(metadata, sort_keys=True),
                encoding="utf-8",
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/x-protobuf")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            return


    server = ThreadingHTTPServer(("0.0.0.0", 4318), Handler)
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.minimum_version = ssl.TLSVersion.TLSv1_2
    tls.load_cert_chain(
        certfile="/run/secrets/otel_gateway_downstream_certificate",
        keyfile="/run/secrets/otel_gateway_downstream_key",
    )
    server.socket = tls.wrap_socket(server.socket, server_side=True)
    server.serve_forever()
    """
).strip()

CRAFTED_SENDER = textwrap.dedent(
    r"""
    from __future__ import annotations

    import os
    from http.client import HTTPException, HTTPSConnection
    from pathlib import Path
    import ssl
    import sys
    from urllib.error import HTTPError, URLError
    from urllib.request import HTTPSHandler, ProxyHandler, Request, build_opener

    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )
    from opentelemetry.proto.common.v1.common_pb2 import (
        AnyValue,
        EntityRef,
        InstrumentationScope,
        KeyValue,
    )
    from opentelemetry.proto.resource.v1.resource_pb2 import Resource
    from opentelemetry.proto.trace.v1.trace_pb2 import (
        ResourceSpans,
        ScopeSpans,
        Span,
        Status,
    )


    def text_value(key: str, value: str) -> KeyValue:
        return KeyValue(key=key, value=AnyValue(string_value=value))


    def int_value(key: str, value: int) -> KeyValue:
        return KeyValue(key=key, value=AnyValue(int_value=value))


    def varint(value: int) -> bytes:
        output = bytearray()
        while value > 0x7F:
            output.append((value & 0x7F) | 0x80)
            value >>= 7
        output.append(value)
        return bytes(output)


    def hidden_key_int(key: str, value: int) -> bytes:
        visible = int_value(key, value).SerializeToString()
        return visible + b"\x18" + varint(31_337)


    safe_span = Span(
        trace_id=b"\x01" * 16,
        span_id=b"\x02" * 8,
        parent_span_id=b"\x0b" * 8,
        name="execute_tool SECRET_TOOLNAME_CANARY",
        kind=Span.SPAN_KIND_CLIENT,
        trace_state="vendor=TRACE_STATE_CANARY",
        flags=255,
        start_time_unix_nano=1_700_000_000_000_000_000,
        end_time_unix_nano=1_700_000_000_123_456_789,
        dropped_attributes_count=91,
        dropped_events_count=92,
        dropped_links_count=93,
        attributes=[
            text_value("gen_ai.agent.description", "STATIC_DESCRIPTION_CANARY"),
            text_value(
                "gcp.vertex.agent.llm_request",
                '{"prompt":"STRUCTURED_GENAI_CANARY"}',
            ),
            text_value("custom.private", "CUSTOM_ATTRIBUTE_CANARY"),
            text_value("gen_ai.usage.output_tokens", "BAD_TOKEN_TYPE_CANARY"),
            int_value("gen_ai.usage.experimental.reasoning_tokens", 7),
            int_value("gen_ai.usage.experimental.system_instruction_tokens", 17),
        ],
        events=[
            Span.Event(
                time_unix_nano=1_700_000_000_100_000_000,
                name="exception",
                attributes=[
                    text_value(
                        "exception.message",
                        "EXCEPTION_MESSAGE_CANARY",
                    ),
                    text_value(
                        "exception.stacktrace",
                        "EXCEPTION_STACKTRACE_CANARY",
                    ),
                ],
            )
        ],
        status=Status(
            code=Status.STATUS_CODE_ERROR,
            message="STATUS_MESSAGE_CANARY",
        ),
    )
    hidden_attribute = hidden_key_int(
        "gen_ai.usage.input_tokens",
        123,
    )
    hidden_attribute_message = KeyValue.FromString(hidden_attribute)
    clean_attribute = KeyValue()
    clean_attribute.CopyFrom(hidden_attribute_message)
    clean_attribute.DiscardUnknownFields()
    if (
        hidden_attribute_message.SerializeToString()
        == clean_attribute.SerializeToString()
    ):
        raise SystemExit("hidden-key-field-precondition-failed")
    safe_span.attributes.append(hidden_attribute_message)

    linked_span = Span(
        trace_id=b"\x03" * 16,
        span_id=b"\x04" * 8,
        name="linked LINKED_SPAN_NAME_CANARY",
        start_time_unix_nano=1_700_000_001_000_000_000,
        end_time_unix_nano=1_700_000_001_100_000_000,
        links=[
            Span.Link(
                trace_id=b"\x05" * 16,
                span_id=b"\x06" * 8,
                trace_state="linked=LINK_TRACE_STATE_CANARY",
                attributes=[
                    text_value("link.private", "LINK_ATTRIBUTE_CANARY"),
                ],
            )
        ],
    )
    fallback_span = Span(
        trace_id=b"\x07" * 16,
        span_id=b"\x08" * 8,
        name="unknown UNKNOWN_SPAN_NAME_CANARY",
        start_time_unix_nano=1_700_000_002_000_000_000,
        end_time_unix_nano=1_700_000_002_100_000_000,
    )
    llm_span = Span(
        trace_id=b"\x09" * 16,
        span_id=b"\x0a" * 8,
        name="call_llm",
        start_time_unix_nano=1_700_000_003_000_000_000,
        end_time_unix_nano=1_700_000_003_100_000_000,
        attributes=[int_value("gen_ai.usage.output_tokens", 456)],
    )
    resource_spans = ResourceSpans(
        resource=Resource(
            attributes=[
                text_value("service.name", "RESOURCE_NAME_CANARY"),
                text_value("service.namespace", "RESOURCE_NAMESPACE_CANARY"),
                text_value("service.version", "RESOURCE_VERSION_CANARY"),
                text_value("resource.private", "RESOURCE_PRIVATE_CANARY"),
            ],
            dropped_attributes_count=94,
            entity_refs=[
                EntityRef(
                    schema_url="https://ENTITY_SCHEMA_CANARY.invalid",
                    type="ENTITY_TYPE_CANARY",
                    id_keys=["ENTITY_ID_KEY_CANARY"],
                    description_keys=["ENTITY_DESCRIPTION_KEY_CANARY"],
                )
            ],
        ),
        scope_spans=[
            ScopeSpans(
                scope=InstrumentationScope(
                    name="SCOPE_NAME_CANARY",
                    version="SCOPE_VERSION_CANARY",
                    attributes=[
                        text_value("scope.private", "SCOPE_ATTR_CANARY"),
                    ],
                    dropped_attributes_count=95,
                ),
                spans=[safe_span, linked_span, fallback_span, llm_span],
                schema_url="https://SCOPE_SCHEMA_CANARY.invalid",
            )
        ],
        schema_url="https://RESOURCE_SCHEMA_CANARY.invalid",
    )
    deprecated_scope = ScopeSpans(
        scope=InstrumentationScope(name="DEPRECATED_SCOPE_CANARY"),
    ).SerializeToString()
    resource_bytes = resource_spans.SerializeToString()
    resource_bytes += (
        varint((1000 << 3) | 2)
        + varint(len(deprecated_scope))
        + deprecated_scope
    )
    payload = b"\x0a" + varint(len(resource_bytes)) + resource_bytes
    round_trip_request = ExportTraceServiceRequest.FromString(payload)
    embedded_attributes = (
        round_trip_request.resource_spans[0]
        .scope_spans[0]
        .spans[0]
        .attributes
    )
    embedded_input_tokens = [
        attribute
        for attribute in embedded_attributes
        if attribute.key == "gen_ai.usage.input_tokens"
    ]
    if len(embedded_input_tokens) != 1:
        raise SystemExit("hidden-key-field-embedding-failed")
    clean_embedded_attribute = KeyValue()
    clean_embedded_attribute.CopyFrom(embedded_input_tokens[0])
    clean_embedded_attribute.DiscardUnknownFields()
    if (
        embedded_input_tokens[0].SerializeToString()
        == clean_embedded_attribute.SerializeToString()
    ):
        raise SystemExit("hidden-key-field-round-trip-failed")

    mode = os.environ["CANARY_MODE"]
    token = (
        Path("/run/secrets/otel_gateway_receiver_token")
        .read_text(encoding="ascii")
        .strip()
    )
    authorization = (
        "Bearer wrong-runtime-token"
        if mode == "wrong-token"
        else f"Bearer {token}"
    )
    request = Request(
        "https://otel-collector:4318/v1/traces",
        data=payload,
        headers={
            "Authorization": authorization,
            "Content-Type": "application/x-protobuf",
        },
        method="POST",
    )
    context = (
        ssl.create_default_context()
        if mode == "wrong-ca"
        else ssl.create_default_context(
            cafile="/run/secrets/otel_gateway_ca",
        )
    )

    if mode == "wrong-token":
        connection = HTTPSConnection(
            "otel-collector",
            4318,
            context=context,
            timeout=5,
        )
        stage = "connect"
        try:
            connection.connect()
            # An empty protobuf message is a valid OTLP request. If the invalid
            # token were accepted, this would return 200 instead of 401.
            auth_payload = ExportTraceServiceRequest().SerializeToString()
            if auth_payload != b"":
                raise SystemExit("empty-otlp-precondition-failed")
            stage = "request"
            connection.request(
                "POST",
                "/v1/traces",
                body=auth_payload,
                headers={
                    "Authorization": authorization,
                    "Content-Type": "application/x-protobuf",
                },
            )
            stage = "response"
            response = connection.getresponse()
            try:
                stage = "status"
                status = response.status
            finally:
                stage = "response-close"
                response.close()
            stage = "complete"
        except (HTTPException, OSError):
            raise SystemExit(f"auth-{stage}-failed") from None
        finally:
            try:
                connection.close()
            except OSError:
                raise SystemExit("auth-connection-close-failed") from None
        if status != 401:
            raise SystemExit("unexpected-auth-status")
        print("auth-rejected")
        raise SystemExit(0)

    try:
        opener = build_opener(
            ProxyHandler({}),
            HTTPSHandler(context=context),
        )
        with opener.open(request, timeout=5) as response:
            status = response.status
    except HTTPError:
        raise SystemExit("unexpected-http-result") from None
    except URLError as error:
        reason = str(error.reason)
        if mode == "wrong-ca" and "CERTIFICATE_VERIFY_FAILED" in reason:
            print("ca-rejected")
            raise SystemExit(0) from None
        raise SystemExit("unexpected-tls-result") from None

    if mode != "success" or status != 200:
        raise SystemExit("unexpected-success")
    print("crafted-exported")
    """
).strip()

SDK_EXPORTER = textwrap.dedent(
    r"""
    from __future__ import annotations

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import OTELResourceDetector
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Status, StatusCode

    from agent.utils.config import ObservabilityEnv
    from agent.utils.observability import configure_otel_resource

    settings = ObservabilityEnv()
    configure_otel_resource("runtime-sdk-agent", settings)
    provider = TracerProvider(resource=OTELResourceDetector().detect())
    exporter = OTLPSpanExporter()
    processor = BatchSpanProcessor(
        exporter,
        schedule_delay_millis=100,
        max_export_batch_size=1,
    )
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("runtime-sdk-canary", "1.0")
    with tracer.start_as_current_span("execute_tool SDK_NAME_CANARY") as span:
        span.set_attribute("custom.private", "SDK_PRIVATE_CANARY")
        span.set_attribute("gen_ai.agent.description", "SDK_DESCRIPTION_CANARY")
        span.set_attribute("gen_ai.usage.input_tokens", 321)
        span.record_exception(RuntimeError("SDK_EXCEPTION_CANARY"))
        span.set_status(Status(StatusCode.ERROR, "SDK_STATUS_CANARY"))
    if not provider.force_flush(5_000):
        raise SystemExit("sdk-force-flush-failed")
    provider.shutdown()
    print("sdk-exported")
    """
).strip()

HEALTH_PROBE = textwrap.dedent(
    r"""
    import json
    from urllib.request import ProxyHandler, build_opener

    opener = build_opener(ProxyHandler({}))
    for path in ("/live", "/ready"):
        with opener.open(
            f"http://127.0.0.1:8080{path}",
            timeout=3,
        ) as response:
            if response.status != 200:
                raise SystemExit("health-status-failed")
            json.load(response)
    print("health-ok")
    """
).strip()

GATEWAY_HEALTH_PROBE = textwrap.dedent(
    r"""
    from urllib.request import ProxyHandler, build_opener

    opener = build_opener(ProxyHandler({}))
    with opener.open("http://otel-collector:13133/", timeout=2) as response:
        if response.status != 200:
            raise SystemExit("gateway-health-failed")
    print("gateway-ready")
    """
).strip()

CAPTURE_COUNT = textwrap.dedent(
    r"""
    from pathlib import Path

    print(len(list(Path("/tmp/trace-capture").glob("*.bin"))))
    """
).strip()

GATEWAY_EXPORTER_OVERLAY = textwrap.dedent(
    """
    exporters:
      otlp_http/langfuse:
        tls:
          ca_file: /run/secrets/otel_gateway_downstream_ca
    """
).strip()

INTEGRATION_COMPOSE = textwrap.dedent(
    """
    services:
      agent:
        ports: !override []
        healthcheck:
          disable: true
        restart: "no"

      otel-collector:
        command:
          - --config=file:/etc/otelcol-contrib/config.yaml
          - --config=file:/etc/otelcol-contrib/test-exporter.yaml
        configs:
          - source: otel_gateway_test_exporter
            target: /etc/otelcol-contrib/test-exporter.yaml
        secrets:
          - source: otel_gateway_downstream_ca
            target: otel_gateway_downstream_ca
        ports: !override []
        restart: "no"
        depends_on:
          otel-downstream:
            condition: service_started

      otel-downstream:
        image: ${IMAGE:?Set the integration agent image}
        entrypoint:
          - python
          - /app/trace-capture-server.py
        command: []
        volumes:
          - type: bind
            source: ${TRACE_GATEWAY_TEST_CAPTURE_SERVER:?Set capture server path}
            target: /app/trace-capture-server.py
            read_only: true
            bind:
              create_host_path: false
        secrets:
          - source: otel_gateway_downstream_certificate
            target: otel_gateway_downstream_certificate
          - source: otel_gateway_downstream_key
            target: otel_gateway_downstream_key
          - source: otel_gateway_downstream_token
            target: otel_gateway_downstream_token
        group_add:
          - ${OTEL_GATEWAY_SECRET_GID:?Set the test secret GID}
        tmpfs:
          - /tmp:size=16m,mode=1777
        networks:
          - otel_gateway_egress
        read_only: true
        user: "1000:1000"
        cap_drop:
          - ALL
        security_opt:
          - no-new-privileges:true
        pids_limit: 100
        restart: "no"

    configs:
      otel_gateway_test_exporter:
        file: ${TRACE_GATEWAY_TEST_EXPORTER_CONFIG:?Set exporter config path}

    secrets:
      otel_gateway_downstream_ca:
        file: ${TRACE_GATEWAY_TEST_CA_FILE:?Set downstream CA path}
      otel_gateway_downstream_certificate:
        file: ${TRACE_GATEWAY_TEST_DOWNSTREAM_CERTIFICATE:?Set downstream cert path}
      otel_gateway_downstream_key:
        file: ${TRACE_GATEWAY_TEST_DOWNSTREAM_KEY:?Set downstream key path}
      otel_gateway_downstream_token:
        file: ${OTEL_GATEWAY_LANGFUSE_TOKEN_FILE:?Set downstream token path}
    """
).strip()


@dataclass(frozen=True)
class GatewayHarness:
    """Exact isolated Compose boundary used by the hosted canary."""

    docker: str
    project: str
    compose_env: Path
    integration_compose: Path
    environment: dict[str, str]
    image: str

    @property
    def compose_prefix(self) -> list[str]:
        """Return the project-scoped Compose command prefix."""
        return [
            self.docker,
            "compose",
            "--project-name",
            self.project,
            "--env-file",
            str(self.compose_env),
            "-f",
            str(BASE_COMPOSE_PATH),
            "-f",
            str(GATEWAY_COMPOSE_PATH),
            "-f",
            str(self.integration_compose),
        ]


def _redact_output(output: str) -> str:
    """Keep synthetic credentials and raw canaries out of failure output."""
    redacted = output
    for sensitive_value in SENSITIVE_LOG_VALUES:
        redacted = redacted.replace(sensitive_value, "[redacted]")
    return redacted


def _run(
    command: list[str],
    *,
    environment: dict[str, str],
    input_text: str | None = None,
    check: bool = True,
    timeout: float = 180,
) -> subprocess.CompletedProcess[str]:
    """Run one fixed process boundary with redacted deterministic failures."""
    result = subprocess.run(  # noqa: S603 - resolved tools and fixed arguments
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        command_name = " ".join(command[:3])
        stdout = _redact_output(result.stdout[-4_000:])
        stderr = _redact_output(result.stderr[-4_000:])
        raise AssertionError(
            f"{command_name} failed with {result.returncode}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return result


def _compose(
    harness: GatewayHarness,
    *arguments: str,
    input_text: str | None = None,
    check: bool = True,
    timeout: float = 180,
) -> subprocess.CompletedProcess[str]:
    """Run one exact-project Compose operation."""
    return _run(
        [*harness.compose_prefix, *arguments],
        environment=harness.environment,
        input_text=input_text,
        check=check,
        timeout=timeout,
    )


def _write_text(path: Path, contents: str, *, mode: int = 0o600) -> None:
    """Write one integration-only file with an explicit host mode."""
    path.write_text(f"{contents.rstrip()}\n", encoding="utf-8")
    path.chmod(mode)


def _generate_server_certificate(
    openssl: str,
    *,
    environment: dict[str, str],
    directory: Path,
    ca_certificate: Path,
    ca_key: Path,
    common_name: str,
    create_serial: bool,
) -> tuple[Path, Path]:
    """Create one short-lived server certificate with an exact DNS SAN."""
    key = directory / f"{common_name}.key"
    request = directory / f"{common_name}.csr"
    certificate = directory / f"{common_name}.pem"
    extension = directory / f"{common_name}.ext"
    _write_text(
        extension,
        "\n".join(
            [
                f"subjectAltName=DNS:{common_name}",
                "extendedKeyUsage=serverAuth",
                "keyUsage=digitalSignature,keyEncipherment",
            ]
        ),
    )
    _run(
        [
            openssl,
            "req",
            "-new",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-sha256",
            "-subj",
            f"/CN={common_name}",
            "-keyout",
            str(key),
            "-out",
            str(request),
        ],
        environment=environment,
    )
    signing_arguments = (
        ["-CAcreateserial"]
        if create_serial
        else ["-CAserial", str(directory / "ca.srl")]
    )
    _run(
        [
            openssl,
            "x509",
            "-req",
            "-sha256",
            "-days",
            "1",
            "-in",
            str(request),
            "-CA",
            str(ca_certificate),
            "-CAkey",
            str(ca_key),
            *signing_arguments,
            "-extfile",
            str(extension),
            "-out",
            str(certificate),
        ],
        environment=environment,
    )
    return certificate, key


def _make_harness(tmp_path: Path) -> GatewayHarness:
    """Build secret-free runtime files and short-lived synthetic credentials."""
    docker = shutil.which("docker")
    openssl = shutil.which("openssl")
    assert docker is not None, "Docker CLI is required for the opt-in gateway proof"
    assert openssl is not None, "OpenSSL is required for the opt-in gateway proof"

    project = os.environ.get(
        "TRACE_GATEWAY_TEST_PROJECT",
        f"adk-gateway-{os.getpid()}",
    ).lower()
    assert re.fullmatch(r"adk-gateway-[a-z0-9][a-z0-9_-]{0,48}", project)

    inherited_names = (
        "DOCKER_CONFIG",
        "DOCKER_CONTEXT",
        "DOCKER_HOST",
        "HOME",
        "PATH",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "XDG_RUNTIME_DIR",
    )
    environment = {
        name: os.environ[name] for name in inherited_names if name in os.environ
    }
    environment.update(
        {
            "COMPOSE_DISABLE_ENV_FILE": "1",
            "LANG": "C.UTF-8",
        }
    )

    secrets_directory = tmp_path / "secrets"
    secrets_directory.mkdir(mode=0o700)
    ca_key = secrets_directory / "ca.key"
    ca_certificate = secrets_directory / "ca.pem"
    _run(
        [openssl, "genrsa", "-out", str(ca_key), "2048"],
        environment=environment,
    )
    _run(
        [
            openssl,
            "req",
            "-x509",
            "-new",
            "-sha256",
            "-days",
            "1",
            "-key",
            str(ca_key),
            "-subj",
            "/CN=trace-gateway-runtime-ca",
            "-out",
            str(ca_certificate),
        ],
        environment=environment,
    )
    gateway_certificate, gateway_key = _generate_server_certificate(
        openssl,
        environment=environment,
        directory=secrets_directory,
        ca_certificate=ca_certificate,
        ca_key=ca_key,
        common_name="otel-collector",
        create_serial=True,
    )
    downstream_certificate, downstream_key = _generate_server_certificate(
        openssl,
        environment=environment,
        directory=secrets_directory,
        ca_certificate=ca_certificate,
        ca_key=ca_key,
        common_name="otel-downstream",
        create_serial=False,
    )
    receiver_token = secrets_directory / "receiver.token"
    langfuse_token = secrets_directory / "langfuse-basic.token"
    _write_text(receiver_token, GATEWAY_TOKEN, mode=0o640)
    _write_text(langfuse_token, BASIC_PAYLOAD, mode=0o640)
    for runtime_secret in (
        ca_certificate,
        gateway_certificate,
        gateway_key,
        downstream_certificate,
        downstream_key,
    ):
        runtime_secret.chmod(0o640)

    capture_server = tmp_path / "capture_server.py"
    exporter_config = tmp_path / "gateway-exporter.yaml"
    integration_compose = tmp_path / "compose.integration.yaml"
    _write_text(capture_server, CAPTURE_SERVER, mode=0o644)
    _write_text(exporter_config, GATEWAY_EXPORTER_OVERLAY, mode=0o644)
    _write_text(integration_compose, INTEGRATION_COMPOSE, mode=0o644)

    agent_env = tmp_path / "agent.env"
    collector_env = tmp_path / "collector.env"
    _write_text(
        agent_env,
        "\n".join(
            [
                "ROOT_AGENT_MODEL=google/gemini-runtime-test",
                f"OPENROUTER_API_KEY={SYNTHETIC_MODEL_KEY}",
                "LANGFUSE_BASE_URL=https://hostile-vendor.example.test",
                "LANGFUSE_PUBLIC_KEY=hostile-public-canary",
                "LANGFUSE_SECRET_KEY=hostile-secret-canary",
                (
                    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="
                    "https://hostile-vendor.example.test/v1/traces"
                ),
                "OTEL_EXPORTER_OTLP_TRACES_HEADERS=authorization=hostile-header",
                "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=grpc",
                ("OTEL_PYTHON_EXPORTER_OTLP_HTTP_CREDENTIAL_PROVIDER=hostile-provider"),
                (
                    "OTEL_PYTHON_EXPORTER_OTLP_HTTP_TRACES_CREDENTIAL_PROVIDER="
                    "hostile-trace-provider"
                ),
                "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=false",
            ]
        ),
    )
    _write_text(
        collector_env,
        "OTEL_GATEWAY_LANGFUSE_AUTHORITY=otel-downstream:4318",
    )

    image = f"trace-gateway-agent:{project}"
    compose_env = tmp_path / "compose.env"
    _write_text(
        compose_env,
        "\n".join(
            [
                f"IMAGE={image}",
                "AGENT_NAME=trace-gateway-runtime-agent",
                f"OTEL_GATEWAY_SECRET_GID={os.getgid()}",
                f"OTEL_GATEWAY_AGENT_ENV_FILE={agent_env}",
                f"OTEL_GATEWAY_COLLECTOR_ENV_FILE={collector_env}",
                f"OTEL_GATEWAY_CA_FILE={ca_certificate}",
                f"OTEL_GATEWAY_SERVER_CERTIFICATE_FILE={gateway_certificate}",
                f"OTEL_GATEWAY_SERVER_KEY_FILE={gateway_key}",
                f"OTEL_GATEWAY_RECEIVER_TOKEN_FILE={receiver_token}",
                f"OTEL_GATEWAY_LANGFUSE_TOKEN_FILE={langfuse_token}",
                f"TRACE_GATEWAY_TEST_CAPTURE_SERVER={capture_server}",
                f"TRACE_GATEWAY_TEST_EXPORTER_CONFIG={exporter_config}",
                f"TRACE_GATEWAY_TEST_CA_FILE={ca_certificate}",
                (f"TRACE_GATEWAY_TEST_DOWNSTREAM_CERTIFICATE={downstream_certificate}"),
                f"TRACE_GATEWAY_TEST_DOWNSTREAM_KEY={downstream_key}",
            ]
        ),
    )
    environment["IMAGE"] = image

    return GatewayHarness(
        docker=docker,
        project=project,
        compose_env=compose_env,
        integration_compose=integration_compose,
        environment=environment,
        image=image,
    )


def _assert_supported_compose_version(harness: GatewayHarness) -> None:
    """Require the first Compose release that supports the override tag."""
    result = _run(
        [harness.docker, "compose", "version", "--short"],
        environment=harness.environment,
    )
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", result.stdout)
    assert match is not None, "Docker Compose returned no semantic version"
    version = tuple(int(component) for component in match.groups())
    assert version >= MINIMUM_COMPOSE_VERSION, (
        f"Docker Compose {MINIMUM_COMPOSE_VERSION!r} or newer is required"
    )


def _assert_collector_image_is_multiarch(harness: GatewayHarness) -> None:
    """Resolve the immutable image index and require both VM architectures."""
    result = _run(
        [
            harness.docker,
            "buildx",
            "imagetools",
            "inspect",
            COLLECTOR_IMAGE,
        ],
        environment=harness.environment,
    )
    platforms = set(
        re.findall(r"^\s*Platform:\s*(\S+)\s*$", result.stdout, flags=re.MULTILINE)
    )
    assert {"linux/amd64", "linux/arm64"}.issubset(platforms)


def _validate_and_start_gateway(harness: GatewayHarness) -> None:
    """Validate both config layers with the pinned binary, then start once."""
    _compose(harness, "config", "--quiet")
    _compose(harness, "pull", "otel-collector")
    version = _compose(
        harness,
        "run",
        "--rm",
        "--no-deps",
        "otel-collector",
        "--version",
    )
    assert re.search(r"\b0[.]157[.]0\b", f"{version.stdout}\n{version.stderr}")
    _compose(
        harness,
        "run",
        "--rm",
        "--no-deps",
        "otel-collector",
        "validate",
        "--config=file:/etc/otelcol-contrib/config.yaml",
        "--config=file:/etc/otelcol-contrib/test-exporter.yaml",
    )
    _compose(harness, "build", "agent")
    _compose(harness, "up", "--detach", "--no-build")


def _container_id(harness: GatewayHarness, service: str) -> str:
    """Return the single project-scoped container ID for a fixed service."""
    result = _compose(harness, "ps", "--all", "--quiet", service)
    container_id = result.stdout.strip()
    assert re.fullmatch(r"[0-9a-f]{12,64}", container_id), (
        f"Compose did not create exactly one {service} container"
    )
    return container_id


def _container_inspection(
    harness: GatewayHarness,
    service: str,
) -> dict[str, Any]:
    """Inspect one previously resolved project container."""
    result = _run(
        [harness.docker, "inspect", _container_id(harness, service)],
        environment=harness.environment,
    )
    decoded = json.loads(result.stdout)
    assert isinstance(decoded, list) and len(decoded) == 1
    inspection = decoded[0]
    assert isinstance(inspection, dict)
    return inspection


def _container_environment(inspection: dict[str, Any]) -> dict[str, str]:
    """Convert Docker's environment list without exposing it on failure."""
    raw_environment = inspection["Config"]["Env"]
    assert isinstance(raw_environment, list)
    environment: dict[str, str] = {}
    for item in raw_environment:
        assert isinstance(item, str) and "=" in item
        name, value = item.split("=", maxsplit=1)
        environment[name] = value
    return environment


def _assert_no_host_bindings(inspection: dict[str, Any]) -> None:
    """Require a container with no published host socket."""
    assert not inspection["HostConfig"].get("PortBindings")


def _assert_runtime_isolation(harness: GatewayHarness) -> None:
    """Prove the actual containers retain the resolved hardening boundary."""
    inspections = {
        service: _container_inspection(harness, service)
        for service in ("agent", "otel-collector", "otel-downstream")
    }
    for inspection in inspections.values():
        _assert_no_host_bindings(inspection)

    project = harness.project
    agent = inspections["agent"]
    gateway = inspections["otel-collector"]
    downstream = inspections["otel-downstream"]

    agent_environment = _container_environment(agent)
    assert {
        key: agent_environment[key]
        for key in (
            "LANGFUSE_BASE_URL",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
            "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
            "OTEL_PYTHON_EXPORTER_OTLP_HTTP_CREDENTIAL_PROVIDER",
            "OTEL_PYTHON_EXPORTER_OTLP_HTTP_TRACES_CREDENTIAL_PROVIDER",
        )
    } == {
        "LANGFUSE_BASE_URL": "",
        "LANGFUSE_PUBLIC_KEY": "",
        "LANGFUSE_SECRET_KEY": "",
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS": "",
        "OTEL_PYTHON_EXPORTER_OTLP_HTTP_CREDENTIAL_PROVIDER": "",
        "OTEL_PYTHON_EXPORTER_OTLP_HTTP_TRACES_CREDENTIAL_PROVIDER": "",
    }
    assert agent_environment["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] == (
        "https://otel-collector:4318/v1/traces"
    )
    assert agent_environment["OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE"] == (
        "/run/secrets/otel_gateway_ca"
    )
    assert agent_environment["OTEL_GATEWAY_BEARER_TOKEN_FILE"] == (
        "/run/secrets/otel_gateway_receiver_token"  # noqa: S105
    )
    assert not any(
        "hostile-vendor.example.test" in value
        or "hostile-provider" in value
        or "hostile-header" in value
        for value in agent_environment.values()
    )
    agent_mounts = {
        mount["Destination"]
        for mount in agent["Mounts"]
        if isinstance(mount, dict) and "Destination" in mount
    }
    assert "/run/secrets/otel_gateway_langfuse_token" not in agent_mounts
    assert {
        "/run/secrets/otel_gateway_ca",
        "/run/secrets/otel_gateway_receiver_token",
    }.issubset(agent_mounts)
    assert set(agent["NetworkSettings"]["Networks"]) == {
        f"{project}_default",
        f"{project}_otel_gateway_telemetry",
    }

    gateway_host = gateway["HostConfig"]
    assert gateway["Config"]["Image"] == COLLECTOR_IMAGE
    assert gateway["Config"]["User"] == "10001:10001"
    assert gateway_host["ReadonlyRootfs"] is True
    assert gateway_host["CapDrop"] == ["ALL"]
    assert gateway_host["SecurityOpt"] == ["no-new-privileges:true"]
    assert gateway_host["PidsLimit"] == 100
    assert gateway_host["Memory"] == 256 * 1024 * 1024
    assert gateway_host["NanoCpus"] == 500_000_000
    assert str(os.getgid()) in gateway_host["GroupAdd"]
    assert set(gateway["NetworkSettings"]["Networks"]) == {
        f"{project}_otel_gateway_egress",
        f"{project}_otel_gateway_telemetry",
    }

    downstream_host = downstream["HostConfig"]
    assert downstream["Config"]["User"] == "1000:1000"
    assert downstream_host["ReadonlyRootfs"] is True
    assert downstream_host["CapDrop"] == ["ALL"]
    assert downstream_host["SecurityOpt"] == ["no-new-privileges:true"]
    assert downstream_host["PidsLimit"] == 100
    assert str(os.getgid()) in downstream_host["GroupAdd"]
    assert set(downstream["NetworkSettings"]["Networks"]) == {
        f"{project}_otel_gateway_egress"
    }


def _exec_agent_script(
    harness: GatewayHarness,
    script: str,
    *,
    environment: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Execute one fixed Python canary inside the running agent container."""
    arguments = ["exec", "-T"]
    for name, value in sorted((environment or {}).items()):
        assert re.fullmatch(r"[A-Z][A-Z0-9_]*", name)
        arguments.extend(["--env", f"{name}={value}"])
    arguments.extend(["agent", "python", "-c", script])
    return _compose(harness, *arguments, check=check, timeout=30)


def _wait_for_agent_script(
    harness: GatewayHarness,
    script: str,
    expected_output: str,
    *,
    timeout: float = 60,
) -> None:
    """Poll a container-local probe without relying on published ports."""
    deadline = time.monotonic() + timeout
    last_result: subprocess.CompletedProcess[str] | None = None
    while time.monotonic() < deadline:
        last_result = _exec_agent_script(harness, script, check=False)
        if last_result.returncode == 0 and last_result.stdout.strip().splitlines()[
            -1:
        ] == [expected_output]:
            return
        time.sleep(0.5)

    assert last_result is not None
    stdout = _redact_output(last_result.stdout[-2_000:])
    stderr = _redact_output(last_result.stderr[-2_000:])
    raise AssertionError(
        f"Container-local probe did not report {expected_output!r}\n"
        f"stdout:\n{stdout}\nstderr:\n{stderr}"
    )


def _capture_count(harness: GatewayHarness) -> int:
    """Read the number of completed payloads from the isolated endpoint."""
    result = _compose(
        harness,
        "exec",
        "-T",
        "otel-downstream",
        "python",
        "-c",
        CAPTURE_COUNT,
    )
    lines = result.stdout.strip().splitlines()
    assert lines and lines[-1].isdigit()
    return int(lines[-1])


def _wait_for_stable_capture_count(
    harness: GatewayHarness,
    *,
    timeout: float = 15,
) -> int:
    """Wait through both bounded batches until the endpoint count is stable."""
    deadline = time.monotonic() + timeout
    previous: int | None = None
    stable_samples = 0
    while time.monotonic() < deadline:
        current = _capture_count(harness)
        if current == previous:
            stable_samples += 1
        else:
            previous = current
            stable_samples = 0
        if stable_samples >= 4:
            return current
        time.sleep(0.5)
    raise AssertionError("Trace capture count did not settle")


def _copy_capture(
    harness: GatewayHarness,
    destination: Path,
) -> None:
    """Copy one read-only snapshot out of the exact project container."""
    destination.mkdir()
    _compose(
        harness,
        "cp",
        "otel-downstream:/tmp/trace-capture/.",
        str(destination),
    )


def _decode_capture(
    destination: Path,
) -> tuple[list[ExportTraceServiceRequest], list[bytes]]:
    """Decode every completed capture and verify its transport metadata."""
    requests: list[ExportTraceServiceRequest] = []
    payloads: list[bytes] = []
    for metadata_path in sorted(destination.glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata == {
            "authorization_ok": True,
            "content_encoding": metadata["content_encoding"],
            "content_type": "application/x-protobuf",
            "ingestion_version": "4",
            "path": "/api/public/otel/v1/traces",
        }
        body = metadata_path.with_suffix(".bin").read_bytes()
        encoding = metadata["content_encoding"]
        if encoding is None:
            payload = body
        else:
            assert isinstance(encoding, str) and encoding.casefold() == "gzip"
            payload = gzip.decompress(body)
        message = ExportTraceServiceRequest()
        message.ParseFromString(payload)
        assert message.resource_spans
        requests.append(message)
        payloads.append(payload)
    return requests, payloads


def _exported_spans(requests: list[ExportTraceServiceRequest]) -> list[Span]:
    """Flatten the captured trace batches into their retained spans."""
    return [
        span
        for request in requests
        for resource_spans in request.resource_spans
        for scope_spans in resource_spans.scope_spans
        for span in scope_spans.spans
    ]


def _integer_attributes(span: Span) -> dict[str, int]:
    """Return the exact integer-only allowlist of one redacted span."""
    attributes: dict[str, int] = {}
    for attribute in span.attributes:
        assert attribute.key in SAFE_SPAN_ATTRIBUTES
        assert attribute.key not in attributes
        assert attribute.value.WhichOneof("value") == "int_value"
        attributes[attribute.key] = attribute.value.int_value
    return attributes


def _assert_unknown_fields_absent(message: Any, label: str) -> None:
    """Require serialization to remain identical after unknown-field removal."""
    clean = type(message)()
    clean.CopyFrom(message)
    clean.DiscardUnknownFields()
    assert message.SerializeToString() == clean.SerializeToString(), (
        f"{label} retained unknown protobuf fields"
    )


def _assert_sanitized_payloads(
    requests: list[ExportTraceServiceRequest],
    payloads: list[bytes],
) -> None:
    """Assert exact post-gateway OTLP shape and adversarial canary removal."""
    combined_payload = b"".join(payloads)
    leaked_canaries = [
        index
        for index, canary in enumerate(SENSITIVE_LOG_VALUES)
        if canary.encode() in combined_payload
    ]
    assert not leaked_canaries, (
        f"Synthetic sensitive values reached exported protobuf: {leaked_canaries}"
    )

    for request_index, request in enumerate(requests):
        _assert_unknown_fields_absent(request, f"request {request_index}")
        for resource_index, resource_spans in enumerate(request.resource_spans):
            _assert_unknown_fields_absent(
                resource_spans,
                f"resource-spans {request_index}:{resource_index}",
            )
            assert resource_spans.schema_url == ""
            resource = resource_spans.resource
            _assert_unknown_fields_absent(
                resource,
                f"resource {request_index}:{resource_index}",
            )
            assert resource.dropped_attributes_count == 0
            assert not resource.entity_refs
            resource_attributes: dict[str, str] = {}
            for attribute in resource.attributes:
                _assert_unknown_fields_absent(attribute, "resource attribute")
                _assert_unknown_fields_absent(
                    attribute.value,
                    "resource attribute value",
                )
                assert attribute.key not in resource_attributes
                assert attribute.value.WhichOneof("value") == "string_value"
                resource_attributes[attribute.key] = attribute.value.string_value
            assert resource_attributes == SAFE_RESOURCE_ATTRIBUTES

            for scope_index, scope_spans in enumerate(resource_spans.scope_spans):
                _assert_unknown_fields_absent(
                    scope_spans,
                    f"scope-spans {request_index}:{resource_index}:{scope_index}",
                )
                assert scope_spans.schema_url == ""
                scope = scope_spans.scope
                _assert_unknown_fields_absent(scope, "instrumentation scope")
                assert scope.name == "google-adk"
                assert scope.version == ""
                assert not scope.attributes
                assert scope.dropped_attributes_count == 0
                for span in scope_spans.spans:
                    _assert_unknown_fields_absent(span, "span")
                    _assert_unknown_fields_absent(span.status, "span status")
                    assert span.name in SAFE_SPAN_NAMES
                    assert span.trace_state == ""
                    assert span.flags == 0
                    assert span.dropped_attributes_count == 0
                    assert span.dropped_events_count == 0
                    assert span.dropped_links_count == 0
                    assert not span.events
                    assert not span.links
                    assert span.status.message == ""
                    for attribute in span.attributes:
                        _assert_unknown_fields_absent(attribute, "span attribute")
                        _assert_unknown_fields_absent(
                            attribute.value,
                            "span attribute value",
                        )
                    _integer_attributes(span)

    spans = _exported_spans(requests)
    by_trace_id: dict[bytes, list[Span]] = {}
    for span in spans:
        by_trace_id.setdefault(span.trace_id, []).append(span)

    assert b"\x03" * 16 not in by_trace_id
    assert {
        trace_id: len(by_trace_id.get(trace_id, []))
        for trace_id in (b"\x01" * 16, b"\x07" * 16, b"\x09" * 16)
    } == {
        b"\x01" * 16: 1,
        b"\x07" * 16: 1,
        b"\x09" * 16: 1,
    }

    safe_span = by_trace_id[b"\x01" * 16][0]
    assert safe_span.name == "agent.tool"
    assert safe_span.span_id == b"\x02" * 8
    assert safe_span.parent_span_id == b"\x0b" * 8
    assert safe_span.kind == Span.SPAN_KIND_CLIENT
    assert safe_span.start_time_unix_nano == 1_700_000_000_000_000_000
    assert safe_span.end_time_unix_nano == 1_700_000_000_123_456_789
    assert safe_span.status.code == Status.STATUS_CODE_ERROR
    assert _integer_attributes(safe_span) == {
        "gen_ai.usage.input_tokens": 123,
        "gen_ai.usage.experimental.reasoning_tokens": 7,
        "gen_ai.usage.experimental.system_instruction_tokens": 17,
    }

    fallback_span = by_trace_id[b"\x07" * 16][0]
    assert fallback_span.name == "agent.operation"
    assert _integer_attributes(fallback_span) == {}

    llm_span = by_trace_id[b"\x09" * 16][0]
    assert llm_span.name == "agent.llm"
    assert _integer_attributes(llm_span) == {"gen_ai.usage.output_tokens": 456}

    sdk_spans = [
        span
        for span in spans
        if _integer_attributes(span).get("gen_ai.usage.input_tokens") == 321
    ]
    assert len(sdk_spans) == 1
    sdk_span = sdk_spans[0]
    assert sdk_span.name == "agent.tool"
    assert sdk_span.status.code == Status.STATUS_CODE_ERROR
    assert _integer_attributes(sdk_span) == {"gen_ai.usage.input_tokens": 321}
    canary_spans = [
        span
        for span in spans
        if span.trace_id in {b"\x01" * 16, b"\x07" * 16, b"\x09" * 16}
        or _integer_attributes(span).get("gen_ai.usage.input_tokens") == 321
    ]
    assert len(canary_spans) == 4


def _wait_for_sanitized_canaries(
    harness: GatewayHarness,
    tmp_path: Path,
    *,
    timeout: float = 30,
) -> tuple[list[ExportTraceServiceRequest], list[bytes]]:
    """Poll copied protobufs until both crafted and SDK canaries arrive."""
    deadline = time.monotonic() + timeout
    attempt = 0
    last_request_count = 0
    last_span_count = 0
    while time.monotonic() < deadline:
        snapshot = tmp_path / f"capture-{attempt:03d}"
        attempt += 1
        _copy_capture(harness, snapshot)
        requests, payloads = _decode_capture(snapshot)
        spans = _exported_spans(requests)
        last_request_count = len(requests)
        last_span_count = len(spans)
        has_crafted = any(span.trace_id == b"\x01" * 16 for span in spans)
        has_sdk = any(
            _integer_attributes(span).get("gen_ai.usage.input_tokens") == 321
            for span in spans
        )
        if has_crafted and has_sdk:
            return requests, payloads
        time.sleep(0.5)
    raise AssertionError(
        "Timed out waiting for redacted trace canaries "
        f"({last_request_count} requests, {last_span_count} spans)"
    )


def _assert_logs_are_sanitized(harness: GatewayHarness) -> None:
    """Inspect bounded service logs without echoing sensitive contents."""
    result = _compose(
        harness,
        "logs",
        "--no-color",
        "--tail",
        "300",
        "agent",
        "otel-collector",
        "otel-downstream",
        check=False,
    )
    assert result.returncode == 0
    combined_logs = f"{result.stdout}\n{result.stderr}"
    leaked_indices = [
        index
        for index, sensitive_value in enumerate(SENSITIVE_LOG_VALUES)
        if sensitive_value in combined_logs
    ]
    assert not leaked_indices, (
        f"Synthetic sensitive values reached service logs: {leaked_indices}"
    )


def _assert_failure_isolation_and_shutdown(harness: GatewayHarness) -> None:
    """Keep agent health and shutdown bounded across gateway failures."""
    _compose(harness, "stop", "--timeout", "10", "otel-downstream")
    queued = _exec_agent_script(
        harness,
        CRAFTED_SENDER,
        environment={"CANARY_MODE": "success"},
    )
    assert queued.stdout.strip() == "crafted-exported"
    time.sleep(3)
    _wait_for_agent_script(
        harness,
        GATEWAY_HEALTH_PROBE,
        "gateway-ready",
        timeout=15,
    )
    _wait_for_agent_script(harness, HEALTH_PROBE, "health-ok", timeout=15)

    gateway_stop_started = time.monotonic()
    _compose(harness, "stop", "--timeout", "10", "otel-collector")
    gateway_shutdown_seconds = time.monotonic() - gateway_stop_started
    assert gateway_shutdown_seconds < 15
    gateway_state = _container_inspection(harness, "otel-collector")["State"]
    assert gateway_state["Running"] is False
    assert gateway_state["Status"] == "exited"
    assert gateway_state["OOMKilled"] is False
    assert gateway_state["ExitCode"] == 0
    _wait_for_agent_script(harness, HEALTH_PROBE, "health-ok", timeout=15)

    started = time.monotonic()
    _compose(harness, "stop", "--timeout", "10", "agent", timeout=20)
    shutdown_seconds = time.monotonic() - started
    assert shutdown_seconds < 15
    state = _container_inspection(harness, "agent")["State"]
    assert state["Running"] is False
    assert state["Status"] == "exited"
    assert state["OOMKilled"] is False
    assert state["ExitCode"] == 0


def _teardown_gateway(
    harness: GatewayHarness,
    primary_error: BaseException | None,
) -> None:
    """Remove only the validated synthetic Compose project and its volumes."""
    result = _compose(
        harness,
        "down",
        "--volumes",
        "--remove-orphans",
        "--timeout",
        "30",
        check=False,
        timeout=60,
    )
    if result.returncode == 0:
        return
    message = (
        "Exact-project trace-gateway teardown failed: "
        f"{_redact_output(result.stderr[-2_000:])}"
    )
    if primary_error is not None:
        primary_error.add_note(message)
        return
    raise AssertionError(message)


def test_real_gateway_rejects_and_redacts_otlp(tmp_path: Path) -> None:
    """Run authenticated TLS canaries through the exact pinned gateway."""
    harness = _make_harness(tmp_path)
    primary_error: BaseException | None = None
    try:
        _assert_supported_compose_version(harness)
        _assert_collector_image_is_multiarch(harness)
        _validate_and_start_gateway(harness)
        _wait_for_agent_script(harness, HEALTH_PROBE, "health-ok")
        _wait_for_agent_script(
            harness,
            GATEWAY_HEALTH_PROBE,
            "gateway-ready",
        )
        _assert_runtime_isolation(harness)

        baseline_count = _wait_for_stable_capture_count(harness)
        wrong_ca = _exec_agent_script(
            harness,
            CRAFTED_SENDER,
            environment={"CANARY_MODE": "wrong-ca"},
        )
        assert wrong_ca.stdout.strip() == "ca-rejected"
        wrong_token = _exec_agent_script(
            harness,
            CRAFTED_SENDER,
            environment={"CANARY_MODE": "wrong-token"},
        )
        assert wrong_token.stdout.strip() == "auth-rejected"
        _wait_for_agent_script(
            harness,
            GATEWAY_HEALTH_PROBE,
            "gateway-ready",
            timeout=15,
        )
        assert _wait_for_stable_capture_count(harness) == baseline_count

        crafted = _exec_agent_script(
            harness,
            CRAFTED_SENDER,
            environment={"CANARY_MODE": "success"},
        )
        assert crafted.stdout.strip() == "crafted-exported"
        sdk = _exec_agent_script(harness, SDK_EXPORTER)
        assert sdk.stdout.strip().splitlines()[-1] == "sdk-exported"

        requests, payloads = _wait_for_sanitized_canaries(
            harness,
            tmp_path,
        )
        _assert_sanitized_payloads(requests, payloads)
        _assert_logs_are_sanitized(harness)
        _assert_failure_isolation_and_shutdown(harness)
        _assert_logs_are_sanitized(harness)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        _teardown_gateway(harness, primary_error)
