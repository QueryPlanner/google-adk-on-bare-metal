# Trace Observability with OpenTelemetry

This template uses Google ADK's process-wide OpenTelemetry provider for agent
traces. ADK remains the sole owner of that provider and its internal processors;
the template does not install, replace, or shut down a second provider.

Remote export is optional. When configured, the pinned runtime adds one
trace-only `BatchSpanProcessor` and sends OTLP over `http/protobuf`. Without a
remote mode, ADK keeps its internal tracing behavior and no trace leaves the
process through this adapter.

## Supported boundary

The template supports:

- Google ADK spans, plus OpenInference spans only with the content-capture
  opt-in;
- resource identity for one server process;
- optional trace export to Langfuse or another OTLP/HTTP collector; and
- a bounded, best-effort graceful flush.

It does not configure:

- OTLP metrics or logs;
- gRPC export;
- FastAPI or other HTTP server spans;
- log correlation, JSON file logs, Cloud Trace, or Cloud Logging;
- a bundled OpenTelemetry Collector; or
- multi-worker or pre-fork provider coordination.

Application logs continue to go to standard output. Use Docker or systemd log
collection and retention independently from trace export.

## Provider and process lifecycle

The server validates observability settings and publishes only validated
OpenTelemetry resource, content-control, and trace-specific exporter variables
before calling ADK's FastAPI factory. ADK then creates the global
`TracerProvider`, retains its internal processors, and adds one outbound batch
processor when remote export is enabled.

The supported deployment is one process started with:

```bash
uv run python -m agent.server
# or
uv run server
```

Compose uses that same single-process contract. Gunicorn, multiple Uvicorn
workers, and other pre-fork modes are outside the tested boundary because every
process would own a separate global provider and export queue.

During graceful application shutdown, ADK closes its runners first. The outer
application lifespan then gives the provider up to five seconds to flush queued
spans. The template never calls `provider.shutdown()`; the OpenTelemetry SDK's
`atexit` hook owns final shutdown only on normal interpreter exit. Uvicorn's
signal-termination path instead relies on the explicit outer flush. This is
still best effort: an unreachable collector can cause exporter errors or
dropped spans. `SIGKILL`, a host crash, or a deadline shorter than the flush
budget cannot flush queued telemetry. Compose allows a ten-second stop grace
period around the application shutdown path.

## Trace-content contract

`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=false` is the default. The
adapter applies that choice to ADK's current and legacy content controls and
does not enable the OpenInference ADK instrumentor. ADK's structured prompt and
response fields plus tool argument and result attributes are therefore elided
by default. Repeated in-process app factories reconcile the process-global
instrumentor, so a previous opt-in cannot survive a later disabled factory.

Setting the flag to `true` enables both ADK content controls and OpenInference
instrumentation. It changes the emitted span shape and is an explicit
privacy-sensitive opt-in:

```dotenv
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
```

This flag is not an arbitrary redaction layer. OpenTelemetry records unhandled
exceptions as span events and error status by default. Exception messages,
stack traces, and status descriptions can therefore contain runtime content
even while structured message and tool payload fields are elided. Static agent
and tool descriptions also remain operational metadata.

Depending on the executed ADK path, content-disabled traces can still include:

- `service.name`, namespace, version, and process instance identity;
- span names, timestamps, durations, status, exception details, and errors;
- model, agent, and tool names or descriptions, invocation settings, and token
  counts; and
- user, session, invocation, or conversation identifiers.

Treat the collector and its access controls, retention, backups, and downstream
integrations as systems that hold sensitive operational metadata. Do not put API
keys, passwords, personal data, or other secrets into service names, identifiers,
descriptions, exception messages, resource attributes, or custom span
attributes. Sanitize exceptions before they cross an instrumented boundary.

Trace capture controls do not govern application logs. In particular, `DEBUG`
logging can include request, response, or tool data from application callbacks.
Apply separate log-level, collection, access, and retention controls.

## Choose one remote mode

Remote export has three valid states:

1. no remote mode;
2. one complete Langfuse mode; or
3. one complete explicit trace OTLP mode.

Langfuse credentials or base URL and an explicit trace endpoint, protocol, or
headers are mutually exclusive. Partial Langfuse credentials, settings from both
modes, unsupported protocols, and unsafe endpoints or headers fail before the
server binds. The shared trace timeout can accompany either complete mode.
Validation errors are generic and do not echo endpoint, credential, or header
values.

### Mode 1: Langfuse

Set both keys. `LANGFUSE_BASE_URL` is optional and defaults to the EU cloud
host:

```dotenv
LANGFUSE_PUBLIC_KEY=pk-lf-replace-me
LANGFUSE_SECRET_KEY=sk-lf-replace-me
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

The adapter derives:

- `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` as
  `<base>/api/public/otel/v1/traces`;
- `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf`; and
- trace headers logically equivalent to:

  ```text
  Authorization: Basic <base64(public-key:secret-key)>
  x-langfuse-ingestion-version: 4
  ```

The OpenTelemetry environment serialization percent-encodes the authorization
value; for example, the space after `Basic` becomes `%20`. The v4 ingestion
header makes directly ingested spans available through Langfuse's current
ingestion path. Derived credentials are never printed.

Do not set an explicit trace endpoint, protocol, or headers in Langfuse mode.
The shared trace timeout below is allowed. Supplying `LANGFUSE_BASE_URL` without
both keys is an incomplete mode and fails configuration validation.

### Mode 2: Explicit OTLP/HTTP traces

Set the complete trace signal endpoint. The protocol can be omitted because the
adapter materializes `http/protobuf`; if supplied, it is matched
case-insensitively and normalized to that value. Headers are optional:

```dotenv
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://collector.example.com/v1/traces
OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
OTEL_EXPORTER_OTLP_TRACES_HEADERS=authorization=Bearer%20replace-me
```

Trace-specific endpoint semantics differ from generic OTLP base-endpoint
semantics: the URL above is used exactly as configured. It must end once in
`/v1/traces`; the exporter does not append another signal path.

Header values use the OpenTelemetry environment format: comma-separated,
percent-encoded `name=value` pairs. Do not quote or log credential-bearing
header values.

### Shared request timeout

Either complete remote mode may set:

```dotenv
OTEL_EXPORTER_OTLP_TRACES_TIMEOUT=2
```

The value is seconds and must be finite, greater than `0`, and at most `2`.
When the variable is omitted, remote export materializes an effective
two-second timeout. Timeout alone is incomplete and does not enable export.

The HTTP exporter can make one retry after a `ConnectionError`. At the maximum,
the two bounded requests are designed to fit inside the five-second outer flush
and leave additional room within Compose's ten-second termination grace period.

## Endpoint and header safety

Remote collectors must use HTTPS. Plaintext HTTP is allowed only for a
loopback collector used in local development or deterministic tests. This
prevents bearer or Basic credentials and trace payloads from crossing a remote
network without transport encryption.

The adapter rejects:

- schemes other than HTTP or HTTPS;
- remote plaintext HTTP;
- missing hosts or ports outside the URL's valid range;
- malformed DNS names, whitespace, or invalid percent encoding;
- embedded usernames or passwords;
- query strings and fragments;
- endpoints that do not end exactly once in `/v1/traces`;
- `grpc` and any protocol other than `http/protobuf`;
- headers without valid `name=value` fields; and
- control characters, including CRLF.

The four documented trace endpoint, protocol, header, and timeout variables are
the only accepted `OTEL_EXPORTER_OTLP_*` settings. Metric-specific,
log-specific, and other exporter variables are rejected before ADK can
interpret them. The supported settings sources are explicit constructor values,
the process environment, and the current-directory `.env`; per-instance
alternate dotenv, secrets-directory, and CLI source overrides are rejected so
they cannot bypass that allowlist.

Endpoint and header failures use stable, secret-free messages. They must not be
diagnosed by printing the rejected values.

## Failure behavior

The exporter sends batches asynchronously. A collector that becomes
unreachable after startup does not make `/live`, ADK's `/health`, or
database-independent `/ready` fail. Those endpoints do not claim trace
delivery.

This fail-open application behavior is deliberate: telemetry loss must not take
the agent out of service. Export attempts can emit SDK errors and spans can be
dropped while the collector is unavailable. The template provides no
store-and-forward queue, retry durability, or delivery guarantee. Monitor the
collector and application logs separately.

## Migrate from generic OTLP variables

Earlier versions documented these global variables:

```dotenv
OTEL_EXPORTER_OTLP_ENDPOINT=...
OTEL_EXPORTER_OTLP_PROTOCOL=...
OTEL_EXPORTER_OTLP_HEADERS=...
```

Remove all three. They are now rejected rather than ignored because the pinned
ADK runtime interprets a global endpoint as permission to create trace, metric,
and log exporters.

Then choose exactly one replacement:

- Langfuse: set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and optionally
  `LANGFUSE_BASE_URL`; or
- another collector: rename endpoint, protocol, and headers to their exact
  `OTEL_EXPORTER_OTLP_TRACES_*` equivalents and make the endpoint a complete
  HTTPS signal URL ending once in `/v1/traces`.

Either replacement may optionally set the bounded trace timeout documented
above.

Do not leave old and new variables together. After changing a VM `.env`,
recreate the service so the process receives the new environment.

## Resource identity

Exported spans use process-level resource attributes:

| Attribute | Source | Purpose |
| --- | --- | --- |
| `service.name` | `AGENT_NAME` | Identifies the agent service |
| `service.namespace` | `TELEMETRY_NAMESPACE`, default `local` | Groups environments or deployments |
| `service.version` | `K_REVISION`, default `local` | Identifies the deployed revision |
| `service.instance.id` | Process ID plus generated UUID | Distinguishes one server process |

The adapter percent-encodes resource values in the OpenTelemetry environment
format before ADK reads them, so commas, equals signs, or control characters in
a value cannot create extra attributes. It also removes a stale
`OTEL_SERVICE_NAME` override so `AGENT_NAME` remains authoritative.

Do not place secrets or personal data in these values.

## References

- [OpenTelemetry OTLP exporter configuration](https://opentelemetry.io/docs/languages/sdk-configuration/otlp-exporter/)
- [OpenTelemetry exception conventions](https://opentelemetry.io/docs/specs/otel/trace/exceptions/)
- [Langfuse OpenTelemetry ingestion](https://langfuse.com/integrations/native/opentelemetry)
- [Architecture](../architecture.md)
- [Environment variables](environment-variables.md)
