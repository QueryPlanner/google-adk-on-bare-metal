## Architecture (minimal, pragmatic)

### Why this repo exists

Google ADK is useful even without Google Cloud:

- You can run the ADK Dev UI locally on your own infrastructure
- You can use a non-Google model provider via `LiteLlm` and `OpenRouter`
- You can persist sessions in a regular database (Postgres)

### Key choices

- **Entry point**: `python -m agent.server`
  - Wraps `google.adk.cli.fast_api.get_fast_api_app(...)`
  - Forces a Postgres-backed session store via `DATABASE_URL`
  - Configures OpenTelemetry for vendor-neutral tracing (Langfuse auto-config included)
- **Agents directory**: `src/`
  - ADK Dev UI lists *directories* under `agents_dir`.
- **Main Agent**: `src/agent/agent.py`
  - Contains `root_agent` to keep ADK discovery simple.
- **DB URL normalization**: Handled in `server.py`
  - Converts standard Postgres URLs (e.g. `postgresql://`) to asyncpg-compatible ones (`postgresql+asyncpg://`)

### What ADK uses the database for

ADK session persistence stores:

- session rows (IDs + state)
- events (conversation history / tool calls)
- app/user state snapshots

This is what makes the Dev UI “remember” conversations across restarts and allows for persistent agent memory.