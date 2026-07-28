"""Black-box runtime verification for trace-only OTLP export."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from opentelemetry.proto.trace.v1.trace_pb2 import Span as OtlpSpan

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
_TOOL_ARGUMENT_CANARY = "runtime-tool-argument-canary"
_TOOL_RESULT_CANARY = "runtime-tool-result-canary"
_TOOL_ERROR_CANARY = "runtime-tool-error-canary"
_PROMPT_CANARY = "runtime-prompt-content-canary"
_RESPONSE_CANARY = "runtime-response-content-canary"
_PROVIDER_KEY_CANARY = "runtime-provider-key-canary"  # noqa: S105
_HEADER_CANARY = "runtime-header-secret-canary"  # noqa: S105
_SERVICE_NAME = "otlp-runtime-agent"
_SERVICE_NAMESPACE = "runtime-tests"
_SERVICE_REVISION = "runtime-test-revision"
_PROCESS_START_TIMEOUT_SECONDS = 20.0
_PROCESS_STOP_TIMEOUT_SECONDS = 12.0
_BOUNDED_SHUTDOWN_SECONDS = 9.0

_CHILD_SOURCE = r"""
import os
import signal
import socket
from contextlib import contextmanager

import uvicorn
from fastapi import FastAPI
from google.adk.events import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from opentelemetry import trace

from agent import server as agent_server


def _qualified_name(value):
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _processor_snapshot():
    provider = trace.get_tracer_provider()
    active_processor = getattr(provider, "_active_span_processor", None)
    processors = getattr(active_processor, "_span_processors", ())
    descriptions = []
    for processor in processors:
        exporter = getattr(processor, "span_exporter", None)
        descriptions.append(
            {
                "processor": _qualified_name(processor),
                "exporter": _qualified_name(exporter) if exporter is not None else None,
            }
        )
    return {
        "provider": _qualified_name(provider),
        "processors": descriptions,
    }


def runtime_private_tool(private_argument: str):
    return {
        "executed": True,
        "length": len(private_argument),
        "private_result": os.environ["OTLP_RUNTIME_TOOL_RESULT_CANARY"],
    }


def runtime_failing_tool(private_argument: str):
    raise RuntimeError(os.environ["OTLP_RUNTIME_TOOL_ERROR_CANARY"])


class RuntimeLlm(BaseLlm):
    async def generate_content_async(self, llm_request, stream=False):
        request_text = llm_request.model_dump_json(exclude_none=True)
        if os.environ["OTLP_RUNTIME_PROMPT_CANARY"] not in request_text:
            raise AssertionError("The runtime prompt did not reach the model boundary")
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text=os.environ["OTLP_RUNTIME_RESPONSE_CANARY"]
                    )
                ],
            ),
            finish_reason=types.FinishReason.STOP,
        )


class NormalExitRuntimeServer(uvicorn.Server):
    # Test-only: let interpreter atexit run after the real SIGTERM lifespan.
    @contextmanager
    def capture_signals(self):
        previous_handler = signal.signal(signal.SIGTERM, self.handle_exit)
        try:
            yield
        finally:
            signal.signal(signal.SIGTERM, previous_handler)


app: FastAPI = agent_server.create_app()


@app.post("/runtime/emit")
async def emit_runtime_tool_span():
    from google.adk.agents import LlmAgent
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.telemetry import _instrumentation
    from google.adk.tools import FunctionTool, ToolContext

    agent = LlmAgent(name="runtime_agent", model="gemini-2.5-flash")
    tool = FunctionTool(runtime_private_tool)
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="runtime",
        user_id="runtime-user",
        session_id="runtime-session",
    )
    invocation_context = InvocationContext(
        session_service=session_service,
        invocation_id="runtime-invocation",
        agent=agent,
        session=session,
    )
    tool_context = ToolContext(
        invocation_context,
        function_call_id="runtime-call",
    )
    arguments = {"private_argument": os.environ["OTLP_RUNTIME_TOOL_CANARY"]}
    async with _instrumentation.record_tool_execution(
        tool,
        agent,
        arguments,
    ) as telemetry_context:
        result = await tool.run_async(
            args=arguments,
            tool_context=tool_context,
        )
        response_part = types.Part.from_function_response(
            name=tool.name,
            response=result,
        )
        response_part.function_response.id = tool_context.function_call_id
        telemetry_context.function_response_event = Event(
            invocation_id=invocation_context.invocation_id,
            author=agent.name,
            content=types.Content(role="user", parts=[response_part]),
        )

    failing_tool = FunctionTool(runtime_failing_tool)
    try:
        async with _instrumentation.record_tool_execution(
            failing_tool,
            agent,
            arguments,
        ):
            await failing_tool.run_async(
                args=arguments,
                tool_context=tool_context,
            )
    except RuntimeError as error:
        expected_error = os.environ["OTLP_RUNTIME_TOOL_ERROR_CANARY"]
        if str(error) != expected_error:
            raise
    else:
        raise AssertionError("The failing runtime tool did not raise")

    inference_agent = LlmAgent(
        name="runtime_inference_agent",
        model=RuntimeLlm(model="runtime-model"),
    )
    inference_events = []
    async with Runner(
        agent=inference_agent,
        app_name="runtime",
        session_service=session_service,
    ) as runner:
        async for event in runner.run_async(
            user_id="runtime-user",
            session_id="runtime-session",
            invocation_id="runtime-inference-invocation",
            new_message=types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=os.environ["OTLP_RUNTIME_PROMPT_CANARY"]
                    )
                ],
            ),
        ):
            inference_events.append(event)

    return {
        "tool_result": result,
        "failure_exercised": True,
        "inference_exercised": bool(inference_events),
        "telemetry": _processor_snapshot(),
    }


listen_socket = socket.socket(
    family=socket.AF_INET,
    type=socket.SOCK_STREAM,
    fileno=int(os.environ["OTLP_RUNTIME_LISTEN_FD"]),
)
config = uvicorn.Config(
    app,
    host="127.0.0.1",
    port=0,
    access_log=False,
    log_level="warning",
)
server_type = (
    NormalExitRuntimeServer
    if os.environ["OTLP_RUNTIME_NORMAL_EXIT"] == "true"
    else uvicorn.Server
)
runtime_server = server_type(config)
runtime_server.run(sockets=[listen_socket])
"""


@dataclass(frozen=True, slots=True)
class _RecordedRequest:
    path: str
    headers: dict[str, str]
    body: bytes


class _TraceCaptureServer(ThreadingHTTPServer):
    """Thread-safe local OTLP/HTTP request recorder."""

    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _TraceCaptureHandler)
        self._requests: list[_RecordedRequest] = []
        self._request_lock = threading.Lock()

    def record(self, request: _RecordedRequest) -> None:
        """Record one complete HTTP request."""
        with self._request_lock:
            self._requests.append(request)

    def snapshot(self) -> tuple[_RecordedRequest, ...]:
        """Return a stable snapshot of all recorded requests."""
        with self._request_lock:
            return tuple(self._requests)


class _TraceCaptureHandler(BaseHTTPRequestHandler):
    """Accept trace export POSTs without producing server logs."""

    server: _TraceCaptureServer

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        """Capture an OTLP request and return a successful empty response."""
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        self.server.record(
            _RecordedRequest(
                path=self.path,
                headers={key.casefold(): value for key, value in self.headers.items()},
                body=body,
            )
        )
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, _format: str, *_args: object) -> None:
        """Suppress request logs so synthetic headers cannot reach output."""


@contextmanager
def _trace_receiver() -> Iterator[_TraceCaptureServer]:
    receiver = _TraceCaptureServer()
    receiver_thread = threading.Thread(
        target=receiver.serve_forever,
        name="test-otlp-receiver",
        daemon=True,
    )
    receiver_thread.start()
    try:
        yield receiver
    finally:
        receiver.shutdown()
        receiver.server_close()
        receiver_thread.join(timeout=5)


@contextmanager
def _stalled_collector() -> Iterator[str]:
    """Expose a deterministic endpoint that accepts TCP but never answers HTTP."""
    stalled_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    stalled_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    stalled_socket.bind(("127.0.0.1", 0))
    stalled_socket.listen()
    host, port = cast(tuple[str, int], stalled_socket.getsockname())
    try:
        yield f"http://{host}:{port}/v1/traces"
    finally:
        stalled_socket.close()


@dataclass(frozen=True, slots=True)
class _RuntimeResult:
    live: dict[str, Any]
    ready: dict[str, Any]
    openapi: dict[str, Any]
    emit: dict[str, Any]
    requests_before_shutdown: tuple[_RecordedRequest, ...]
    return_code: int
    stdout: str
    stderr: str
    shutdown_seconds: float


def _minimal_child_environment(
    *,
    agent_dir: Path,
    capture_content: bool,
    endpoint: str,
    listen_fd: int,
    normal_exit_after_sigterm: bool,
) -> dict[str, str]:
    """Build a credential-isolated environment with no dotenv lookup path."""
    return {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(_SOURCE_ROOT),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUTF8": "1",
        "ADK_DISABLE_LOAD_DOTENV": "true",
        "AGENT_DIR": str(agent_dir),
        "AGENT_NAME": _SERVICE_NAME,
        "ALLOW_ORIGINS": "[]",
        "SERVE_WEB_INTERFACE": "false",
        "RELOAD_AGENTS": "false",
        "LOG_LEVEL": "WARNING",
        "TELEMETRY_NAMESPACE": _SERVICE_NAMESPACE,
        "K_REVISION": _SERVICE_REVISION,
        "OTEL_SERVICE_NAME": "stale-runtime-service-name",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": endpoint,
        "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/protobuf",
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS": (f"x-runtime-auth={_HEADER_CANARY}"),
        "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT": "2",
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": (
            str(capture_content).lower()
        ),
        "OTEL_BSP_SCHEDULE_DELAY": "60000",
        "OTEL_BSP_MAX_QUEUE_SIZE": "64",
        "OTEL_BSP_MAX_EXPORT_BATCH_SIZE": "32",
        "OTEL_TRACES_SAMPLER": "always_on",
        "OPENROUTER_API_KEY": _PROVIDER_KEY_CANARY,
        "OTLP_RUNTIME_LISTEN_FD": str(listen_fd),
        "OTLP_RUNTIME_NORMAL_EXIT": str(normal_exit_after_sigterm).lower(),
        "OTLP_RUNTIME_PROMPT_CANARY": _PROMPT_CANARY,
        "OTLP_RUNTIME_RESPONSE_CANARY": _RESPONSE_CANARY,
        "OTLP_RUNTIME_TOOL_CANARY": _TOOL_ARGUMENT_CANARY,
        "OTLP_RUNTIME_TOOL_RESULT_CANARY": _TOOL_RESULT_CANARY,
        "OTLP_RUNTIME_TOOL_ERROR_CANARY": _TOOL_ERROR_CANARY,
    }


def _redacted_output(stdout: str, stderr: str) -> str:
    combined = f"stdout:\n{stdout}\nstderr:\n{stderr}"
    for canary in (
        _TOOL_ARGUMENT_CANARY,
        _TOOL_RESULT_CANARY,
        _TOOL_ERROR_CANARY,
        _PROMPT_CANARY,
        _RESPONSE_CANARY,
        _PROVIDER_KEY_CANARY,
        _HEADER_CANARY,
    ):
        combined = combined.replace(canary, "<synthetic-canary-redacted>")
    return combined[-8_000:]


def _request_json(
    url: str,
    *,
    method: str = "GET",
) -> dict[str, Any]:
    data = b"{}" if method == "POST" else None
    request = urllib.request.Request(  # noqa: S310 - inherited loopback socket
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=1.0) as response:  # noqa: S310
        assert response.status == 200
        return cast(dict[str, Any], json.loads(response.read()))


def _wait_until_live(
    process: subprocess.Popen[str],
    base_url: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + _PROCESS_START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(
                "Runtime child exited before becoming live.\n"
                f"{_redacted_output(stdout, stderr)}"
            )
        try:
            return _request_json(f"{base_url}/live")
        except (OSError, urllib.error.HTTPError, urllib.error.URLError):
            time.sleep(0.05)
    pytest.fail("Runtime child did not become live before the startup deadline.")


def _exercise_runtime(
    tmp_path: Path,
    *,
    capture_content: bool = False,
    endpoint: str,
    normal_exit_after_sigterm: bool,
    receiver: _TraceCaptureServer | None = None,
) -> _RuntimeResult:
    agent_dir = tmp_path / "runtime-agents"
    agent_dir.mkdir()

    listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_socket.bind(("127.0.0.1", 0))
    listen_socket.listen()
    listen_socket.set_inheritable(True)
    host, port = cast(tuple[str, int], listen_socket.getsockname())
    process: subprocess.Popen[str] | None = None

    try:
        process = subprocess.Popen(  # noqa: S603 - fixed interpreter and source
            [sys.executable, "-c", _CHILD_SOURCE],
            cwd=tmp_path,
            env=_minimal_child_environment(
                agent_dir=agent_dir,
                capture_content=capture_content,
                endpoint=endpoint,
                listen_fd=listen_socket.fileno(),
                normal_exit_after_sigterm=normal_exit_after_sigterm,
            ),
            pass_fds=(listen_socket.fileno(),),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        listen_socket.close()

        base_url = f"http://{host}:{port}"
        live = _wait_until_live(process, base_url)
        ready = _request_json(f"{base_url}/ready")
        openapi = _request_json(f"{base_url}/openapi.json")
        emit = _request_json(f"{base_url}/runtime/emit", method="POST")
        requests_before_shutdown = receiver.snapshot() if receiver else ()

        shutdown_started = time.monotonic()
        process.send_signal(signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=_PROCESS_STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=5)
            pytest.fail(
                "Runtime child exceeded the shutdown deadline.\n"
                f"{_redacted_output(stdout, stderr)}"
            )

        return _RuntimeResult(
            live=live,
            ready=ready,
            openapi=openapi,
            emit=emit,
            requests_before_shutdown=requests_before_shutdown,
            return_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            shutdown_seconds=time.monotonic() - shutdown_started,
        )
    finally:
        if listen_socket.fileno() != -1:
            listen_socket.close()
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


def _assert_health_and_process_result(result: _RuntimeResult) -> None:
    assert result.live == {"status": "alive"}
    assert result.ready == {
        "status": "ready",
        "checks": {"database": "not_configured"},
    }
    assert result.shutdown_seconds < _BOUNDED_SHUTDOWN_SECONDS

    combined_output = f"{result.stdout}\n{result.stderr}"
    openapi_document = json.dumps(result.openapi)
    assert "overriding of current tracerprovider" not in combined_output.casefold()
    canaries = (
        ("tool argument", _TOOL_ARGUMENT_CANARY),
        ("tool result", _TOOL_RESULT_CANARY),
        ("tool error", _TOOL_ERROR_CANARY),
        ("prompt", _PROMPT_CANARY),
        ("response", _RESPONSE_CANARY),
        ("provider key", _PROVIDER_KEY_CANARY),
        ("OTLP header", _HEADER_CANARY),
    )
    output_leaks = [label for label, canary in canaries if canary in combined_output]
    openapi_leaks = [label for label, canary in canaries if canary in openapi_document]
    assert not output_leaks, f"Synthetic canaries reached child output: {output_leaks}"
    assert not openapi_leaks, f"Synthetic canaries reached OpenAPI: {openapi_leaks}"


def _assert_adk_provider_ownership(result: _RuntimeResult) -> None:
    """Verify both ADK processors and exactly one outbound trace exporter."""
    telemetry = cast(dict[str, Any], result.emit["telemetry"])
    assert telemetry["provider"] == "opentelemetry.sdk.trace.TracerProvider"
    processors = cast(list[dict[str, str | None]], telemetry["processors"])
    processor_types = [description["processor"] for description in processors]
    assert (
        processor_types.count("opentelemetry.sdk.trace.export.SimpleSpanProcessor") == 2
    )
    assert (
        processor_types.count("opentelemetry.sdk.trace.export.BatchSpanProcessor") == 1
    )
    assert len(processors) == 3
    assert {description["exporter"] for description in processors} == {
        "google.adk.cli.adk_web_server.ApiServerSpanExporter",
        "google.adk.cli.adk_web_server.InMemoryExporter",
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
    }


def _decoded_trace_requests(
    requests: tuple[_RecordedRequest, ...],
) -> list[ExportTraceServiceRequest]:
    decoded: list[ExportTraceServiceRequest] = []
    for request in requests:
        message = ExportTraceServiceRequest()
        message.ParseFromString(request.body)
        assert message.resource_spans
        decoded.append(message)
    return decoded


def _exported_spans(
    decoded_requests: list[ExportTraceServiceRequest],
) -> list[OtlpSpan]:
    """Flatten every exported OTLP scope into one inspectable span list."""
    return [
        span
        for export_request in decoded_requests
        for resource_spans in export_request.resource_spans
        for scope_spans in resource_spans.scope_spans
        for span in scope_spans.spans
    ]


def _string_attributes(attributes: Any) -> dict[str, str]:
    """Extract string-valued OTLP attributes used by the runtime assertions."""
    return {attribute.key: attribute.value.string_value for attribute in attributes}


@pytest.mark.skipif(
    os.name == "nt",
    reason="Runtime verification requires inherited POSIX sockets and SIGTERM.",
)
def test_runtime_elides_structured_content_during_graceful_shutdown(
    tmp_path: Path,
) -> None:
    """Verify default elision and the documented exception-content boundary."""
    with _trace_receiver() as receiver:
        receiver_host, receiver_port = cast(
            tuple[str, int],
            receiver.server_address,
        )
        endpoint = f"http://{receiver_host}:{receiver_port}/v1/traces"
        result = _exercise_runtime(
            tmp_path,
            endpoint=endpoint,
            normal_exit_after_sigterm=False,
            receiver=receiver,
        )
        requests = receiver.snapshot()

    _assert_health_and_process_result(result)
    # Stock Uvicorn re-raises SIGTERM after its production lifespan completes.
    assert result.return_code == -signal.SIGTERM
    assert result.requests_before_shutdown == ()
    assert result.emit["tool_result"] == {
        "executed": True,
        "length": len(_TOOL_ARGUMENT_CANARY),
        "private_result": _TOOL_RESULT_CANARY,
    }
    assert result.emit["failure_exercised"] is True
    assert result.emit["inference_exercised"] is True

    _assert_adk_provider_ownership(result)

    assert requests
    assert {request.path for request in requests} == {"/v1/traces"}
    assert all(
        request.headers.get("x-runtime-auth") == _HEADER_CANARY for request in requests
    )
    decoded_requests = _decoded_trace_requests(requests)
    exported_spans = _exported_spans(decoded_requests)

    span_names = {span.name for span in exported_spans}
    resources: list[dict[str, str]] = []
    for export_request in decoded_requests:
        for resource_spans in export_request.resource_spans:
            resources.append(
                {
                    attribute.key: attribute.value.string_value
                    for attribute in resource_spans.resource.attributes
                }
            )
    assert "execute_tool runtime_private_tool" in span_names
    assert "execute_tool runtime_failing_tool" in span_names
    assert "call_llm" in span_names
    assert any(
        resource.get("service.name") == _SERVICE_NAME
        and resource.get("service.namespace") == _SERVICE_NAMESPACE
        and resource.get("service.version") == _SERVICE_REVISION
        and resource.get("service.instance.id", "").startswith("worker-")
        for resource in resources
    )

    protobuf_payload = b"".join(request.body for request in requests)
    leaked_payload_canaries = [
        label
        for label, canary in (
            ("tool argument", _TOOL_ARGUMENT_CANARY),
            ("tool result", _TOOL_RESULT_CANARY),
            ("prompt", _PROMPT_CANARY),
            ("response", _RESPONSE_CANARY),
            ("provider key", _PROVIDER_KEY_CANARY),
            ("OTLP header", _HEADER_CANARY),
        )
        if canary.encode() in protobuf_payload
    ]
    assert not leaked_payload_canaries, (
        f"Synthetic canaries reached trace protobuf: {leaked_payload_canaries}"
    )

    failing_span = next(
        span
        for span in exported_spans
        if span.name == "execute_tool runtime_failing_tool"
    )
    assert failing_span.status.message == f"RuntimeError: {_TOOL_ERROR_CANARY}"
    exception_event = next(
        event for event in failing_span.events if event.name == "exception"
    )
    exception_attributes = _string_attributes(exception_event.attributes)
    assert exception_attributes["exception.message"] == _TOOL_ERROR_CANARY
    assert _TOOL_ERROR_CANARY in exception_attributes["exception.stacktrace"]


@pytest.mark.skipif(
    os.name == "nt",
    reason="Runtime verification requires inherited POSIX sockets and SIGTERM.",
)
def test_runtime_content_capture_opt_in_exports_structured_content(
    tmp_path: Path,
) -> None:
    """Prove the explicit opt-in changes real ADK OTLP payloads."""
    with _trace_receiver() as receiver:
        receiver_host, receiver_port = cast(
            tuple[str, int],
            receiver.server_address,
        )
        endpoint = f"http://{receiver_host}:{receiver_port}/v1/traces"
        result = _exercise_runtime(
            tmp_path,
            capture_content=True,
            endpoint=endpoint,
            normal_exit_after_sigterm=False,
            receiver=receiver,
        )
        requests = receiver.snapshot()

    _assert_health_and_process_result(result)
    _assert_adk_provider_ownership(result)
    assert result.return_code == -signal.SIGTERM
    assert result.requests_before_shutdown == ()
    assert requests
    assert {request.path for request in requests} == {"/v1/traces"}
    protobuf_payload = b"".join(request.body for request in requests)
    for canary in (
        _TOOL_ARGUMENT_CANARY,
        _TOOL_RESULT_CANARY,
        _PROMPT_CANARY,
        _RESPONSE_CANARY,
    ):
        assert canary.encode() in protobuf_payload
    for secret_canary in (_PROVIDER_KEY_CANARY, _HEADER_CANARY):
        assert secret_canary.encode() not in protobuf_payload


@pytest.mark.skipif(
    os.name == "nt",
    reason="Runtime verification requires inherited POSIX sockets and SIGTERM.",
)
def test_runtime_health_and_shutdown_are_bounded_when_collector_stalls(
    tmp_path: Path,
) -> None:
    """Bound app-lifespan cleanup when final trace export is deliberately blocked."""
    with _stalled_collector() as endpoint:
        result = _exercise_runtime(
            tmp_path,
            endpoint=endpoint,
            normal_exit_after_sigterm=True,
        )

    _assert_health_and_process_result(result)
    # The test-only signal wrapper proves normal SDK atexit remains bounded too.
    assert result.return_code == 0, _redacted_output(result.stdout, result.stderr)
    assert result.emit["tool_result"]["executed"] is True
