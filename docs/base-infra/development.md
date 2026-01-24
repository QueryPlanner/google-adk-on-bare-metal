# Development

This document covers development workflows, code quality standards, and testing.

## Prerequisites

- Python 3.13+
- `uv` package manager
- Postgres Database (for persistent sessions)
- Google API Key (or OpenRouter Key)

## Running Locally

### Quick Start

```bash
# Minimal setup for local development
cp .env.example .env
# Edit .env:
# - Set GOOGLE_API_KEY
# - Set DATABASE_URL (e.g., postgresql://user:pass@localhost:5432/db)

# Run server
uv run python -m agent_foundation.server  # API-only (set SERVE_WEB_INTERFACE=TRUE for web UI)
LOG_LEVEL=DEBUG uv run python -m agent_foundation.server  # Debug mode

# Docker Compose (recommended - hot reloading & DB included)
docker compose up --build --watch
```

See [Docker Compose Workflow](./docker-compose-workflow.md) and [Environment Variables](./environment-variables.md).

## Development Workflow

### Feature Branch Development

```bash
# Create branch (feat/, fix/, docs/, refactor/, test/)
git checkout -b feat/your-feature-name

# Develop locally
uv run python -m agent_foundation.server  # Fast iteration
# Or: docker compose up --build --watch  # Matches production

# Quality checks before commit (100% coverage required)
uv run ruff format && uv run ruff check && uv run mypy
uv run pytest --cov --cov-report=term-missing

# Commit (conventional format: 50 char title, list body)
git add . && git commit -m "feat: add new tool"
```

## Code Quality and Testing

```bash
# Quality checks (run before commit)
uv run ruff format && uv run ruff check && uv run mypy

# Tests (100% coverage required)
uv run pytest --cov --cov-report=term-missing

# Specific tests
uv run pytest tests/test_integration.py -v
uv run pytest tests/test_file.py::test_name -v
```

## Standards

**Type Hints:** Strict mypy, complete annotations, modern Python 3.13+ syntax (`|` unions, lowercase generics), Pydantic validation.

**Code Style:** Ruff (88-char lines, auto-fix). Always use `Path` objects (never `os.path`). See `pyproject.toml` for rules.

**Docstrings:** Google-style format. Document args, returns, exceptions.

**Testing:** 100% coverage (excludes server.py, agent.py, scripts). Shared fixtures in `conftest.py`. Duck-typed mocks.

## Dependencies

**Runtime:** Google ADK, pydantic, python-dotenv
**Dev:** pytest, ruff, mypy (PEP 735 `dev` group, auto-installed with `uv run`)

```bash
uv add package-name              # Add runtime dependency
uv add --group dev package-name  # Add dev dependency
uv lock --upgrade                # Update all
uv lock --upgrade-package pkg    # Update specific
```

## Project Structure

```
your-agent-name/
  src/agent_foundation/
    agent.py              # LlmAgent configuration
    callbacks.py          # Agent callbacks
    prompt.py             # Agent prompts
    tools.py              # Custom tools
    server.py             # FastAPI server
    utils/                # Utilities
      config.py           # Configuration and environment parsing
      observability.py    # OpenTelemetry setup
  tests/                  # Test suite
    conftest.py           # Shared fixtures
    test_*.py             # Unit and integration tests
  docs/                   # Documentation
  .env.example            # Environment template
  pyproject.toml          # Project configuration
  docker-compose.yml      # Local development
  Dockerfile              # Container image
  README.md               # Main documentation
```

## Observability

OpenTelemetry exports traces to Cloud Trace and logs to Cloud Logging if configured. Control log level: `LOG_LEVEL=DEBUG uv run python -m agent_foundation.server`

See [Observability Guide](./observability.md) for details.
