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
  - Configures OpenTelemetry for vendor-neutral tracing (Langfuse auto-config included)
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

### What ADK uses the database for

ADK session persistence stores:

- session rows (IDs + state)
- events (conversation history / tool calls)
- app/user state snapshots

This is what makes the Dev UI “remember” conversations across restarts and allows for persistent agent memory.
