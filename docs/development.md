## Development

### Goals

- Use Google ADK features locally **without** requiring Google Cloud services.
- Keep the codebase easy to reason about (small, explicit modules).
- Enforce a “green build”: **pytest + mypy + ruff must pass**.

### Repo layout

- `server.py`: Main platform entrypoint (FastAPI + ADK + Postgres)
- `agents/<name>/agent.py`: Agent definition (exports `root_agent`)
- `platform_utils/`: Shared helper modules (env + db url normalization)
- `tests/`: unit tests
- `.env`: Configuration file (API keys, DB URL)

### Environment variables

Create `.env` in the project root:

- `OPENROUTER_API_KEY`: OpenRouter API key (required by the agent model)
- `DATABASE_URL`: Postgres URL for sessions (required by the server)

Optional:

- `HOST`: server bind host (default: `127.0.0.1`)
- `PORT`: server port (default: `8000`)
- `AGENTS_DIR`: directory containing agent folders (default: `./agents`)

### Install

```bash
uv sync
```

To install dev tooling as well (pytest/mypy/ruff), sync with the dev extra:

```bash
uv sync --extra dev
```

### Run the server

```bash
uv run python -m server
```

### Checks (must be 100% green)

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run pytest
```


