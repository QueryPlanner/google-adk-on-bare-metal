# Environment Variables

Complete reference for all environment variables used in this project.

## Configuration

### Database

**DATABASE_URL**
- **When:** Optional; required for persistent database-backed sessions
- **Value:** Postgres connection string (e.g., `postgresql://user:pass@localhost:5432/dbname`)
- **Purpose:** Persistent storage for agent sessions and memory

Both `ssl=...` and standard `sslmode=...` query options are normalized for the
consumer that receives the URL. The shared direct-asyncpg and SQLAlchemy path
also accepts `target_session_attrs`, `krbsrvname`, `gsslib`, and `passfile`.
The `host` query option is accepted only when the URL authority has no hostname,
and `port` only when the authority has no port. This supports credentialed
Unix-socket URLs without allowing query options to override an address. Other
query keys fail closed because direct asyncpg and SQLAlchemy otherwise interpret
them differently.
`channel_binding=require` is accepted for compatibility but is removed because
the current asyncpg stack does not support that option.

When `DATABASE_URL` is set in the container, the entrypoint verifies a real
`SELECT 1` before starting the server. Authentication and database-selection
errors fail immediately; transient connection failures are retried within the
following bounds:

**DB_READY_TIMEOUT**
- **Default:** `60`
- **Purpose:** Maximum total number of seconds spent retrying startup readiness

**DB_READY_RETRY_INTERVAL**
- **Default:** `1`
- **Purpose:** Seconds between transient readiness failures

**DB_READY_ATTEMPT_TIMEOUT**
- **Default:** `5`
- **Purpose:** Maximum seconds for one connection, query, and close attempt

**DB_READINESS_PROBE_TIMEOUT**
- **Default:** `2`
- **Range:** Greater than `0` and at most `2`
- **Purpose:** Maximum seconds for one recurring `/ready` PostgreSQL attempt

The HTTP readiness budget is separate from startup retries and is always
shorter than the Compose client's three-second timeout. `/ready` performs one
fresh authenticated `SELECT 1` without retrying. A failure returns HTTP 503
without returning or logging the database URL, credentials, database name, or
exception text.

`/ready` reports only PostgreSQL connectivity. It does not verify ADK's internal
SQLAlchemy pool or schema permissions beyond `SELECT 1`, Agent Engine, the model
provider, tools, or artifact capacity after startup. `/live` and ADK's legacy
`/health` route are process-only.

### API Keys

**GOOGLE_API_KEY**
- **When:** Required if using Google models
- **Value:** AI Studio API Key

**OPENROUTER_API_KEY**
- **When:** Required if using OpenRouter models
- **Value:** OpenRouter API Key

### Google Cloud (Optional)

**GOOGLE_CLOUD_PROJECT**
- **When:** Optional (required for Vertex AI)
- **Value:** Your GCP project ID
- **Purpose:** Identifies the Google Cloud project used by model calls

**GOOGLE_CLOUD_LOCATION**
- **When:** Optional
- **Value:** GCP region (e.g., `us-central1`)
- **Purpose:** Region for Vertex AI calls

### Agent Runtime Configuration

**AGENT_NAME**
- **When:** Required
- **Value:** Unique identifier (e.g., `my-agent`)
- **Purpose:** Identifies the agent service and its trace resource

**AGENT_DIR**
- **When:** Optional
- **Default:** The installed `src` directory detected by the server
- **Purpose:** Directory containing ADK agent packages

The server resolves this directory to an absolute path and always stores local
artifacts beneath `<AGENT_DIR>/.adk/artifacts`; there is no separate artifact
directory setting. Dedicate that root to one ADK application and use one writer
at a time for each `user_id`/`session_id`/`filename` key.

Compose pins `AGENT_DIR` to `/app/src`, and the trusted `agent_artifacts` named
volume is mounted at `/app/src/.adk`. For direct or systemd execution, the
service account must be able to create, write, sync, and remove files beneath
the resolved `.adk` directory. Directory and file modes follow the process
umask; ownership, permissions, umask, capacity, retention, encryption, and
backups are operator responsibilities.

The filesystem backend retains artifacts across normal process or container
recreation only. It does not provide crash consistency, application
authentication, quotas, retention, encryption, or backup automation. Treat the
artifact root as trusted application data. `docker compose down --volumes`
permanently deletes the Compose artifact volume.

**ROOT_AGENT_MODEL**
- **When:** Optional
- **Default:** `gemini-2.5-flash`
- **Purpose:** Model used by the root ADK agent

**LOG_LEVEL**
- **Options:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Default:** `INFO`
- **Purpose:** Controls logging verbosity

**HOST**
- **Default:** `127.0.0.1`
- **Purpose:** Process bind address for direct and systemd runs

Compose overrides `HOST` to `0.0.0.0` inside the container. That internal bind
does not publish the service on the VM.

**PORT**
- **Default:** `8080`
- **Purpose:** Server listening port

**AGENT_PUBLISH_HOST**
- **When:** Docker Compose only
- **Default:** `127.0.0.1`
- **Purpose:** Host-side address used to publish container port 8080

Setting `AGENT_PUBLISH_HOST=0.0.0.0` exposes the unauthenticated ADK API on
every host interface. It is an unsafe escape hatch, not a substitute for an
authenticated HTTPS reverse proxy.

### Feature Flags

**SERVE_WEB_INTERFACE**
- **Default:** `FALSE`
- **Purpose:** Enables ADK web UI at http://127.0.0.1:8080
- **Options:** `TRUE` / `FALSE`

**RELOAD_AGENTS**
- **Default:** `FALSE`
- **Purpose:** Enable agent hot-reloading on file changes (development only)

**OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT**
- **Default:** `FALSE`
- **Purpose:** Privacy-sensitive opt-in for structured prompt/response fields
  and tool argument/result attributes in traces

The server materializes this choice across ADK's current and legacy content
controls. When it is `FALSE`, the OpenInference ADK instrumentor is not enabled,
so the exported span shape is intentionally smaller than opt-in mode. When it is
`TRUE`, both ADK content controls are enabled and OpenInference instrumentation
is added.

`FALSE` is not arbitrary redaction and does not make traces anonymous.
OpenTelemetry exception messages, stack traces, and status descriptions can
contain runtime content. Exported metadata can also include service and process
identity, span names, timing and status, static agent or tool descriptions,
model or tool names, token counts, and ADK user, session, invocation, or
conversation identifiers. Treat the collector as a destination for sensitive
operational metadata and sanitize exception text.

### Observability (Langfuse)

**LANGFUSE_PUBLIC_KEY**
- **When:** Optional; must be set together with `LANGFUSE_SECRET_KEY`
- **Value:** `pk-lf-...`
- **Purpose:** Langfuse project identity used to derive trace authentication

**LANGFUSE_SECRET_KEY**
- **When:** Optional; must be set together with `LANGFUSE_PUBLIC_KEY`
- **Value:** `sk-lf-...`
- **Purpose:** Langfuse secret used to derive trace authentication

**LANGFUSE_BASE_URL**
- **Default in complete Langfuse mode:** `https://cloud.langfuse.com`
- **Options:** A regional Langfuse host such as
  `https://us.cloud.langfuse.com`, or a self-hosted base URL

Complete Langfuse credentials select one atomic configuration mode. The server
derives the full trace endpoint
`<base>/api/public/otel/v1/traces`, fixes the protocol to `http/protobuf`, and
derives these logical request headers without logging their values:

```text
Authorization: Basic <base64(public-key:secret-key)>
x-langfuse-ingestion-version: 4
```

The serialized OpenTelemetry header value is percent-encoded; for example, the
space after `Basic` is represented as `%20`. Langfuse mode must not be combined
with an explicit trace endpoint, protocol, or headers. The shared trace timeout
below may accompany Langfuse. A base URL without both keys, incomplete
credentials, and mixed modes fail before the server binds.

### Observability (Explicit Trace OTLP)

**OTEL_EXPORTER_OTLP_TRACES_ENDPOINT**
- **When:** Optional
- **Value:** Complete trace signal URL ending exactly once in `/v1/traces`
- **Purpose:** Selects an explicit OTLP/HTTP trace collector

**OTEL_EXPORTER_OTLP_TRACES_PROTOCOL**
- **Default when explicit export is enabled:** `http/protobuf`
- **Allowed value:** `http/protobuf`
- **Purpose:** Selects the only transport supported by this template

**OTEL_EXPORTER_OTLP_TRACES_HEADERS**
- **When:** Optional
- **Value:** Comma-separated, percent-encoded `name=value` pairs
- **Purpose:** Supplies trace-collector authentication or routing headers

**OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE**
- **When:** Optional; explicit HTTPS trace mode only
- **Value:** Absolute path to one nonempty, readable CA certificate file of at
  most 1 MiB
- **Purpose:** Adds trust for a private-TLS Collector without enabling a client
  certificate or private key

**OTEL_GATEWAY_BEARER_TOKEN_FILE**
- **When:** Optional; only with an explicit HTTPS endpoint and custom CA
- **Value:** Absolute path to a nonempty bearer-token file of at most 4 KiB
- **Purpose:** Loads private-network receiver authentication from a mounted
  secret without putting the token in Compose configuration or container
  metadata; the process derives the trace `Authorization` header at startup

**OTEL_EXPORTER_OTLP_TRACES_TIMEOUT**
- **When:** Optional; requires either complete Langfuse or explicit trace mode
- **Effective default during remote export:** `2` seconds
- **Allowed range:** Finite, greater than `0`, and at most `2` seconds
- **Purpose:** Bounds each OTLP/HTTP export request

Explicit trace mode must not include any `LANGFUSE_*` value. A remote endpoint
must use HTTPS. Plaintext HTTP is accepted only for a loopback collector.
Endpoints with embedded credentials, queries, or fragments are rejected, as
are malformed headers and CRLF characters. Validation failures use generic
messages and do not echo the endpoint, credentials, or header values.

The timeout is shared by both remote modes and does not select a mode. Timeout
alone fails configuration validation. At the two-second maximum, the HTTP
exporter's single `ConnectionError` retry is designed to fit inside the
five-second outer flush and leave additional room within Compose's ten-second
stop grace period.

`grpc` is not supported. This template configures no OTLP metric or log
exporter, HTTP server instrumentation, Cloud Trace or Cloud Logging exporter,
or Collector in the base Compose deployment.

The endpoint, protocol, headers, certificate, and timeout above are the only
accepted
`OTEL_EXPORTER_OTLP_*` settings. Metric-specific, log-specific, and other
exporter settings are rejected before the server binds.

`OTEL_PYTHON_EXPORTER_OTLP_HTTP_CREDENTIAL_PROVIDER` and
`OTEL_PYTHON_EXPORTER_OTLP_HTTP_TRACES_CREDENTIAL_PROVIDER` are also rejected
when nonblank. Custom exporter sessions can override request routing and are
outside this template's transport contract. The gateway overlay explicitly
clears both hooks.

`OTEL_GATEWAY_BEARER_TOKEN_FILE` is template-specific rather than an
OpenTelemetry SDK variable. It is mutually exclusive with explicit trace
headers, requires the custom CA setting, accepts only strict ASCII bearer-token
syntax with at most one terminal line ending, and never prints the path or token
in validation failures. The
`compose.trace-gateway.yaml` overlay mounts the same secret into the application
and Collector receiver.

Legacy generic `OTEL_EXPORTER_OTLP_ENDPOINT`,
`OTEL_EXPORTER_OTLP_PROTOCOL`, and `OTEL_EXPORTER_OTLP_HEADERS` values are
rejected rather than ignored. The pinned ADK runtime would otherwise interpret a
generic endpoint as permission to configure traces, metrics, and logs. Remove
all three legacy values before selecting either supported mode.

## Environment Variable Precedence

1. **Explicit settings constructor values** (highest priority)
2. **Environment variables**
3. **.env file** (loaded from the current working directory via
   `pydantic-settings`)
4. **Default values** (defined in code)

Run server commands from the repository root when relying on `.env`. Container and
VM-injected environment variables always take priority over values in that file.
Server and observability settings read the current-directory file without copying
its contents into the process environment. Before instrumentation, the server
publishes validated resource and content controls plus any configured
trace-specific `OTEL_EXPORTER_OTLP_TRACES_*` values. It does not publish generic
OTLP exporter values.

Supported ADK and server entry points load agent provider and memory variables
before importing the agent definition while preserving existing process values. A
direct `from agent import app` is intentionally process-only; export provider
variables first when using that low-level import path.

## Security Best Practices

- **Never commit `.env` files** - Already gitignored
- **Rotate credentials** - If `.env` is accidentally committed, rotate all credentials
- **Keep port 8080 on loopback** - Authenticate the entire upstream before
  exposing it through HTTPS
- **Keep the ADK web interface disabled on VMs** - Enable it temporarily through
  an SSH tunnel when needed
