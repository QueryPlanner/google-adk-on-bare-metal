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
- **When:** Optional (required for Vertex AI or Cloud Observability)
- **Value:** Your GCP project ID
- **Purpose:** Identifies the Google Cloud project

**GOOGLE_CLOUD_LOCATION**
- **When:** Optional
- **Value:** GCP region (e.g., `us-central1`)
- **Purpose:** Region for Vertex AI calls

### Agent Runtime Configuration

**AGENT_NAME**
- **When:** Required
- **Value:** Unique identifier (e.g., `my-agent`)
- **Purpose:** Identifies logs and traces

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
- **Purpose:** Capture full prompts/responses in traces. Set to `TRUE` to see conversation content in Langfuse or Jaeger.

### Observability (Langfuse)

**LANGFUSE_PUBLIC_KEY**
- **Value:** `pk-lf-...`
- **Purpose:** Automatically configures OTel to export to Langfuse.

**LANGFUSE_SECRET_KEY**
- **Value:** `sk-lf-...`
- **Purpose:** Authentication for Langfuse OTLP.

**LANGFUSE_BASE_URL**
- **Default:** `https://cloud.langfuse.com`
- **Options:** `https://us.cloud.langfuse.com` or your self-hosted URL.

### Observability (Generic OTLP)

**OTEL_EXPORTER_OTLP_ENDPOINT**
- **When:** Optional
- **Purpose:** Explicit OTLP collector endpoint

**OTEL_EXPORTER_OTLP_PROTOCOL**
- **When:** Optional
- **Purpose:** Explicit OTLP transport protocol, such as `http/protobuf`

**OTEL_EXPORTER_OTLP_HEADERS**
- **When:** Optional
- **Purpose:** Authentication headers for the OTLP collector

Explicit OTLP values take precedence over values derived from Langfuse credentials.

## Environment Variable Precedence

1. **Explicit settings constructor values** (highest priority)
2. **Environment variables**
3. **.env file** (loaded from the current working directory via
   `pydantic-settings`)
4. **Default values** (defined in code)

Run server commands from the repository root when relying on `.env`. Container and
VM-injected environment variables always take priority over values in that file.
Server and observability settings read the current-directory file without copying
its contents into the process environment. The server publishes only the validated
standard `OTEL_*` values required by the OpenTelemetry SDK before instrumentation.

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
