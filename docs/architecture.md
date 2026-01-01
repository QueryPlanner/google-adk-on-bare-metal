## Architecture (minimal, pragmatic)

### Why this repo exists

Google ADK is useful even without Google Cloud:

- You can run the ADK Dev UI locally
- You can use a non-Google model provider via `LiteLlm`
- You can persist sessions in a regular database (Postgres)

### Key choices

- **Entry point**: `python -m server`
  - Wraps `google.adk.cli.fast_api.get_fast_api_app(...)`
  - Forces a Postgres-backed session store via `DATABASE_URL`
- **Agents directory**: `./agents`
  - ADK Dev UI lists *directories* under `agents_dir`, so we keep a dedicated
    folder with only real agent apps (to avoid listing `docs/`, `tests/`, etc).
- **Agent module**: `agents/<name>/agent.py`
  - Contains `root_agent` to keep ADK discovery simple.
  - Sub-agents are instantiated via factory functions to ensure clean state on reload.
- **DB URL normalization**: `platform_utils/db_url.py`
  - Converts libpq-style parameters (e.g. `sslmode=require`) to asyncpg-compatible ones

### What ADK uses the database for

ADK session persistence stores:

- session rows (IDs + state)
- events (conversation history / tool calls)
- app/user state snapshots

This is what makes the Dev UI “remember” conversations across restarts.


