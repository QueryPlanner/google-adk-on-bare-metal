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
- an OpenTelemetry Collector in the base Compose deployment; or
- multi-worker or pre-fork provider coordination.

Application logs continue to go to standard output. Use Docker or systemd log
collection and retention independently from trace export.

Operators can explicitly add the optional private-TLS redaction gateway
documented below. It is not enabled by the base Compose file.

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
# Optional private CA:
OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE=/absolute/path/to/collector-ca.pem
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

The five documented trace endpoint, protocol, header, certificate, and timeout
variables are
the only accepted `OTEL_EXPORTER_OTLP_*` settings. Metric-specific,
log-specific, and other exporter variables are rejected before ADK can
interpret them. The supported settings sources are explicit constructor values,
the process environment, and the current-directory `.env`; per-instance
alternate dotenv, secrets-directory, and CLI source overrides are rejected so
they cannot bypass that allowlist.

Endpoint and header failures use stable, secret-free messages. They must not be
diagnosed by printing the rejected values.

## Optional VM trace-redaction gateway

Use `compose.trace-gateway.yaml` only when traces must cross an outbound privacy
boundary before leaving a single Linux VM. The base direct-export mode remains
the smaller operational choice. The overlay requires Docker Compose 2.24.4 or
newer because it uses the `!override` merge tag.

The overlay adds a pinned Collector with:

- one private internal telemetry network shared with the agent;
- a separate Collector-only egress network;
- TLS 1.2 or newer and bearer authentication on the local OTLP receiver;
- Langfuse Basic authentication loaded only by the Collector;
- no OTLP receiver port published to the host;
- process health on `127.0.0.1:13133` and payload-free internal metrics on
  `127.0.0.1:8888`;
- a 0.5 CPU, 256 MiB, and 100-PID limit; and
- a bounded in-memory queue with no raw-telemetry WAL.

The application receives only the local Collector endpoint, CA, and receiver
token file. The overlay replaces the base agent env file and explicitly clears
all `LANGFUSE_*` settings and operator-provided trace headers, even if those
values are present in that replacement file. Langfuse endpoint and credential
material exist only in the Collector. The application reads the local receiver
token from its mounted file and materializes the derived authorization header
inside its own process because ADK consumes exporter settings through the
OpenTelemetry environment contract; Docker Compose configuration and container
metadata do not contain that token.
The overlay also clears both Python OTLP HTTP credential-provider hooks. The
application rejects nonblank hook values from every settings source and appends
`otel-collector` to both `NO_PROXY` variants while preserving existing entries,
so the authenticated VM hop uses direct Docker DNS.

### What the gateway removes

The fail-closed pipeline drops every span containing a link and every span
event, including exception events. It replaces resource identity with fixed safe
values, normalizes instrumentation scope and known ADK operation names, maps
unknown span names to `agent.operation`, clears `status.message` and trace
state, and removes every span attribute except these integer fields:

- `gen_ai.usage.input_tokens`;
- `gen_ai.usage.output_tokens`;
- `gen_ai.usage.experimental.reasoning_tokens`; and
- `gen_ai.usage.experimental.system_instruction_tokens`.

A second redaction allowlist runs before batching. Non-integer values under an
allowed token key are deleted. The four allowed attributes are reconstructed
under fresh literal keys so hidden protobuf key metadata cannot survive. Span
flags and every dropped-item counter are reset to zero. Trace/span identifiers
and parentage, span kind, timestamps, duration, status code, and the four safe
token counts remain.

The pinned transform processor cannot address every field in newer OTLP
resource envelopes. A final empty-key `groupbyattrs` pass therefore reconstructs
resources and scopes from the already allowlisted fields; this removes
`Resource.entity_refs`, `ResourceSpans.schema_url`, and
`ScopeSpans.schema_url`, plus the deprecated field-1000 scope envelope, before
batching. The policy is deliberately lossy, and new ADK attributes remain
dropped until reviewed. Never enable Collector transform debug logging because
it can print the raw transform context.

This is a minimization boundary for non-adversarial standard SDK
instrumentation, not data-loss prevention against compromised application code.
Retained IDs, timestamps, integer counts, kind, and status code can be
used as covert channels by code deliberately constructing hostile OTLP.

### Prepare gateway files

Create the operator-owned files without copying any repository `.env` that
contains Langfuse credentials. The root-owned `setup.sh` creates the
`otel-gateway` group and adds `agent-runner`. On a VM prepared another way, an
administrator must first run `groupadd --system otel-gateway` and add the
deployment account to that group; the fallback below assumes the default
`agent-runner` account. Log in again before continuing. The remaining commands
run as the unprivileged deployment account:

```bash
# Administrator-only fallback when setup.sh was not used:
getent group otel-gateway >/dev/null \
  || sudo groupadd --system otel-gateway
sudo usermod -aG otel-gateway agent-runner
```

```bash
compose_version="$(docker compose version --short)"
python3 -c '
import re
import sys

match = re.fullmatch(r"v?([0-9]+)[.]([0-9]+)[.]([0-9]+)(?:[-+].*)?", sys.argv[1])
if match is None or tuple(map(int, match.groups())) < (2, 24, 4):
    raise SystemExit("Docker Compose 2.24.4 or newer is required")
' "$compose_version"
unset compose_version

umask 077
getent group otel-gateway >/dev/null
id -nG | tr ' ' '\n' | grep -qx otel-gateway
mkdir -p secrets/otel-gateway
cp deploy/otel-collector/agent.env.example .env.trace-agent
cp deploy/otel-collector/collector.env.example .env.trace-collector
cp deploy/otel-collector/compose.env.example .env.trace-gateway
chmod 600 .env.trace-agent .env.trace-collector .env.trace-gateway

gateway_secret_gid="$(getent group otel-gateway | cut -d: -f3)"
test -n "$gateway_secret_gid"
sed -i \
  "s/^OTEL_GATEWAY_SECRET_GID=.*/OTEL_GATEWAY_SECRET_GID=$gateway_secret_gid/" \
  .env.trace-gateway
unset gateway_secret_gid
```

Set the application and model values in `.env.trace-agent`. Set the non-secret
`AGENT_NAME` and host file paths in `.env.trace-gateway`. In
`.env.trace-collector`, set `OTEL_GATEWAY_LANGFUSE_AUTHORITY` to only the
Langfuse hostname and optional port, with no scheme or path. The Collector
configuration supplies the immutable `https://` scheme and exact
`/api/public/otel/v1/traces` path, so a plaintext exporter URL cannot be
configured. Do not add `LANGFUSE_*`, a vendor endpoint, or a vendor header to
the agent file.

The dedicated numeric group is added to both non-root containers. Runtime
secret files are group-readable, but Compose mounts only the CA and receiver
token into the agent; the server key and Langfuse token remain Collector-only.
This avoids running either service as root and avoids relying on unsupported
UID/GID remapping for bind-mounted local Compose secrets.

Create a VM-local CA and a receiver certificate whose SAN matches the Compose
service name:

```bash
openssl genpkey -algorithm RSA \
  -pkeyopt rsa_keygen_bits:3072 \
  -out secrets/otel-gateway/ca.key
openssl req -x509 -new -sha256 -days 365 \
  -key secrets/otel-gateway/ca.key \
  -subj "/CN=ADK trace gateway CA" \
  -out secrets/otel-gateway/ca.pem
openssl genpkey -algorithm RSA \
  -pkeyopt rsa_keygen_bits:3072 \
  -out secrets/otel-gateway/server.key
openssl req -new -sha256 \
  -key secrets/otel-gateway/server.key \
  -subj "/CN=otel-collector" \
  -out secrets/otel-gateway/server.csr
openssl x509 -req -sha256 -days 90 \
  -in secrets/otel-gateway/server.csr \
  -CA secrets/otel-gateway/ca.pem \
  -CAkey secrets/otel-gateway/ca.key \
  -CAcreateserial \
  -extfile deploy/otel-collector/server-cert.ext \
  -out secrets/otel-gateway/server.pem
openssl rand -hex 32 > secrets/otel-gateway/receiver.token
chmod 600 \
  secrets/otel-gateway/ca.key \
  secrets/otel-gateway/server.csr \
  secrets/otel-gateway/ca.srl
chgrp otel-gateway \
  secrets/otel-gateway/ca.pem \
  secrets/otel-gateway/server.pem \
  secrets/otel-gateway/server.key \
  secrets/otel-gateway/receiver.token
chmod 640 \
  secrets/otel-gateway/ca.pem \
  secrets/otel-gateway/server.pem \
  secrets/otel-gateway/server.key \
  secrets/otel-gateway/receiver.token
```

The Collector's Langfuse auth extension expects the base64 payload after the
`Basic` scheme, not the scheme itself. Generate it from operator-controlled
environment variables without printing either key:

```bash
test -n "${LANGFUSE_PUBLIC_KEY:-}"
test -n "${LANGFUSE_SECRET_KEY:-}"
printf '%s:%s' "$LANGFUSE_PUBLIC_KEY" "$LANGFUSE_SECRET_KEY" \
  | openssl base64 -A \
  > secrets/otel-gateway/langfuse-basic.token
chgrp otel-gateway secrets/otel-gateway/langfuse-basic.token
chmod 640 secrets/otel-gateway/langfuse-basic.token
unset LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY
```

The receiver loader accepts the single terminal newline written by
`openssl rand` and rejects embedded whitespace or additional lines. Do not
commit these env, token, certificate, or private-key files. The repository
ignores `secrets/otel-gateway/` as an additional staging safeguard.

### Validate and start

Resolve the exact two-file deployment and validate the configuration with the
pinned binary before starting:

```bash
COMPOSE_DISABLE_ENV_FILE=1 docker compose \
  --env-file .env.trace-gateway \
  -f compose.yaml \
  -f compose.trace-gateway.yaml \
  config --quiet

COMPOSE_DISABLE_ENV_FILE=1 docker compose \
  --env-file .env.trace-gateway \
  -f compose.yaml \
  -f compose.trace-gateway.yaml \
  run --rm --no-deps otel-collector \
  validate --config=file:/etc/otelcol-contrib/config.yaml

COMPOSE_DISABLE_ENV_FILE=1 docker compose \
  --env-file .env.trace-gateway \
  -f compose.yaml \
  -f compose.trace-gateway.yaml \
  up --build --wait --wait-timeout 180
```

The application does not depend on Collector health. `--wait` proves the
application readiness contract only. Check the Collector separately:

```bash
curl --fail --silent http://127.0.0.1:13133/
curl --fail --silent http://127.0.0.1:8888/metrics \
  | grep -E 'otelcol_(processor|exporter|receiver)'
```

Internal metrics carry component names and counts, not trace payloads. Use them
to inspect receive/export failures, queue pressure, and filter drops. To verify
the transformation policy itself, run the repository's isolated synthetic
canary; it creates temporary credentials, publishes no host ports, and does not
read the deployment `.env`:

```bash
RUN_TRACE_GATEWAY_INTEGRATION=1 \
  uv run pytest tests/test_trace_gateway_runtime.py -q
```

Never enable detailed transform logs against real traffic.

### Failure, rotation, and rollback

If the Collector or Langfuse is unavailable, trace export can retry briefly,
fill the bounded queue, and drop data. `/live`, ADK's `/health`, and
database-independent `/ready` remain healthy. The Collector health extension
reports process health only and does not claim downstream delivery.

Upgrade the Collector in its own pull request. Review the pinned release's
processor and security changes, update the tag and digest together, and require
the manifest index to resolve both `linux/amd64` and `linux/arm64`. The built-in
configuration validation and hosted TLS/redaction canary must pass before
merge, followed by the same synthetic canary on the target VM architecture
during rollout. Dependabot watches the Compose manifest, but its proposed digest
is not sufficient evidence by itself. After merge, pull and recreate only
`otel-collector`, then check loopback health and queue/export metrics before
treating the upgrade as complete. Roll back by restoring the preceding reviewed
tag/digest and recreating that service; do not substitute a moving tag.

Rotate the receiver token, server certificate, and Langfuse token by replacing
the files atomically and recreating both services. Keep the old CA during a
staged certificate rotation so the application trusts the replacement before
the Collector switches certificates.

Rollback to direct mode by stopping the overlay project, choosing one documented
direct export mode in the base `.env`, and starting only `compose.yaml`. Do not
copy Collector-only credentials into the agent env file as part of rollback.

This gateway guarantees only that the removed fields do not appear in the
outbound payload verified after the Collector. Raw values can still exist in
Python, ADK processors, the local HTTPS request, and transient Collector
receiver memory. It does not protect against application code intentionally
encoding data into retained structural or integer fields. See
[ADR 0001](../adr/0001-private-tls-trace-gateway.md) for the transport decision
and tradeoffs.

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
