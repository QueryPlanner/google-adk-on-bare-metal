## Architecture (minimal, pragmatic)

### Why this repo exists

Google ADK is useful even without Google Cloud:

- You can run the ADK Dev UI locally on your own infrastructure
- You can use a non-Google model provider via `LiteLlm` and `OpenRouter`
- You can persist sessions in a regular database (Postgres)

### Key choices

- **Entry point**: `python -m agent.server`
  - Wraps `google.adk.cli.fast_api.get_fast_api_app(...)`
  - Uses a Postgres-backed session store when `DATABASE_URL` is configured
  - Lets ADK own the process-wide OpenTelemetry provider and internal processors
  - Optionally configures one trace-only OTLP/HTTP exporter
- **Agents directory**: `src/`
  - ADK Dev UI lists *directories* under `agents_dir`.
- **Main Agent**: `src/agent/agent.py`
  - Contains `root_agent` to keep ADK discovery simple.
- **DB URL normalization**: Shared by startup and HTTP readiness
  - Gives direct asyncpg and ADK's SQLAlchemy service compatible forms of the
    same validated PostgreSQL URL

### Runtime health contracts

- `/health` is ADK's unchanged process-level compatibility endpoint.
- `/live` is the template's explicit process-only endpoint and performs no
  external calls.
- `/ready` performs one bounded fresh `SELECT 1` when PostgreSQL is configured.
  Without PostgreSQL it reports `database: not_configured`.

Compose probes `/ready`, so a later PostgreSQL outage changes container health
without pretending the process has died. The readiness check intentionally does
not create another persistent connection pool. It proves current PostgreSQL
connectivity and credentials, not ADK pool capacity, schema compatibility,
Agent Engine, models, tools, or artifact capacity after startup.

### Trace observability boundary

Resource and privacy settings are validated before the ADK FastAPI factory runs.
ADK then creates the sole global `TracerProvider`, preserves its internal trace
processors, and adds one outbound batch processor only when one complete remote
trace mode is configured. The template does not replace or shut down that
provider.

Remote export supports OTLP traces over `http/protobuf`. It does not configure
OTLP metrics or logs, HTTP server spans, log correlation, Cloud Trace, Cloud
Logging, or a bundled collector. Structured message and tool payload fields are
elided by default, but exception content, static descriptions, and operational
metadata—including session or conversation identifiers—can still be exported.

Every remote request has a two-second effective timeout.
`OTEL_EXPORTER_OTLP_TRACES_TIMEOUT` can supply a finite value greater than zero
and at most two seconds alongside either complete remote mode, but cannot select
a mode by itself. This budget is designed to fit the HTTP exporter's single
`ConnectionError` retry inside the five-second outer flush and leave additional
room within Compose's ten-second termination grace period.

The provider is process-global, so the deployment contract is one server
process started with `python -m agent.server` or `uv run server`. Pre-fork and
multi-worker server modes are outside this template's tested boundary.

### What ADK uses the database for

ADK session persistence stores:

- session rows (IDs + state)
- events (conversation history / tool calls)
- app/user state snapshots

This is what makes the Dev UI “remember” conversations across restarts and allows for persistent agent memory.
