## Development

### Goals

- Use Google ADK features locally **without** requiring Google Cloud services.
- Keep the codebase easy to reason about (small, explicit modules).
- Enforce a “green build”: **pytest + mypy + ruff must pass**.

### Repo layout

- `src/agent/server.py`: Main platform entrypoint (FastAPI + ADK + Postgres)
- `src/agent/agent.py`: Agent definition (exports `root_agent`)
- `src/agent/utils/`: Shared helper modules (config + observability)
- `tests/`: Unit and integration tests
- `.env`: Configuration file (API keys, DB URL)

### Environment variables

Create `.env` in the project root:

- `AGENT_NAME`: Unique identifier for the agent service (required)
- `GOOGLE_API_KEY`: Google AI Studio key (optional if using OpenRouter)
- `OPENROUTER_API_KEY`: OpenRouter API key (required for non-Google models)
- `DATABASE_URL`: Postgres URL for sessions (required for persistence)

Observability (Optional):
- `LANGFUSE_PUBLIC_KEY`: Langfuse Public Key
- `LANGFUSE_SECRET_KEY`: Langfuse Secret Key
- `LANGFUSE_BASE_URL`: Langfuse Host (default: EU)

Optional:

- `HOST`: server bind host (default: `127.0.0.1`)
- `PORT`: server port (default: `8080`)
- `LOG_LEVEL`: Logging verbosity (default: `INFO`)
- `SERVE_WEB_INTERFACE`: Whether to serve the ADK web UI (default: `false`)

### Install

```bash
uv sync
```

### Run the server

**Local Python:**
```bash
uv run python -m agent.server
```

**Docker Compose (Recommended for full stack):**
```bash
docker compose up --build --watch
```

### Workflow

1.  **Develop:** Edit files in `src/`.
2.  **Test:** Run unit tests frequently.
3.  **Check:** Run quality checks before committing.

### Code Quality & Standards

We enforce high standards to ensure long-term maintainability.

**Commands (Must be 100% Green):**
```bash
uv run ruff check .          # Linting
uv run ruff format --check . # Formatting check
uv run mypy .                # Type checking
uv run pytest                # Unit tests (100% coverage required)
```

**Standards:**
- **Type Hints:** Strict `mypy`. Use modern Python 3.13+ syntax (e.g., `str | None`, `list[str]`). Pydantic for validation.
- **Code Style:** `ruff` default configuration (88-char lines). Use `pathlib.Path` instead of `os.path` where possible.
- **Docstrings:** Google-style format. Document arguments, return values, and exceptions.
- **Testing:** 100% coverage required for new code.

### Dependency Management

```bash
uv add package-name              # Add runtime dependency
uv add --group dev package-name  # Add dev dependency
uv lock --upgrade                # Update all dependencies
```
