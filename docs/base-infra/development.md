# Development

This document covers development workflows, code quality standards, and testing.

## Prerequisites

- Python 3.13+
- `uv` package manager
- Google Cloud SDK (gcloud CLI) for Vertex AI authentication

> [!NOTE]
> The agent defaults to in-memory session and artifact services for local development.
> For production-consistent testing with durable persistence, see [Cloud Resource Configuration](#cloud-resource-configuration) below.

## Running Locally

### Quick Start (In-Memory Mode)

```bash
# Minimal setup for local development
cp .env.example .env  # Edit: GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION
gcloud auth application-default login

# Run server (uses in-memory sessions/memory/artifacts)
uv run server  # API-only (set SERVE_WEB_INTERFACE=TRUE for web UI)
LOG_LEVEL=DEBUG uv run server  # Debug mode

# Docker Compose (recommended - hot reloading)
docker compose up --build --watch
```

See [Docker Compose Workflow](./docker-compose-workflow.md) and [Environment Variables](./environment-variables.md).

### Configure Cloud Resources (Optional)

For production-consistent testing with cloud session/memory persistence and GCS artifacts, add deployed resource URIs to `.env` before running the server:

```bash
# From deployment logs (see CI/CD Infrastructure Setup in README)
AGENT_ENGINE=projects/.../reasoningEngines/...
ARTIFACT_SERVICE_URI=gs://...
```

See [Environment Variables](./environment-variables.md) for details.

## Testing Deployed Service

Proxy the deployed Cloud Run service to `http://localhost:8000` for local testing:

```bash
# Service name format: ${agent_name}-${workspace} (e.g., my-agent-default)
gcloud run services proxy <service-name> --project <project-id> --region <region> --port 8000

# Example
gcloud run services proxy my-agent-default --project my-project-id --region us-central1 --port 8000
curl http://localhost:8000/health

# Use a different port if 8000 is in use
gcloud run services proxy <service-name> --project <project-id> --region <region> --port 8080

# Stop proxy: Ctrl+C
```

If `SERVE_WEB_INTERFACE=TRUE` is set in the deployed service, open http://localhost:8000 in your browser to access the web UI.

Uses active gcloud account credentials (`gcloud auth login`).

See [Cloud Run proxy documentation](https://cloud.google.com/run/docs/authenticating/developers#proxy) for details.

## Development Workflow

### Feature Branch Development

```bash
# Create branch (feat/, fix/, docs/, refactor/, test/)
git checkout -b feat/your-feature-name

# Develop locally
uv run server  # Fast iteration
# Or: docker compose up --build --watch  # Matches production

# Quality checks before commit (100% coverage required)
uv run ruff format && uv run ruff check && uv run mypy
uv run pytest --cov --cov-report=term-missing

# Commit (conventional format: 50 char title, list body)
git add . && git commit -m "feat: add new tool"
```

### Pull Request and Deployment

```bash
# Push and create PR
git push origin feat/your-feature-name
gh pr create  # Follow PR format: What, Why, How, Tests

# After merge to main, monitor deployment
gh run list --workflow=ci-cd.yml --limit 5
gh run view --log
```

GitHub Actions automatically builds, tests, and deploys to Cloud Run. Check job summary for deployment details.

## Cloud Resource Configuration

After deploying infrastructure (see README), capture resource values for local development.

Get values from GitHub Actions logs (`gh run view <run-id>` or Actions tab UI) or GCP Console:

```bash
# Add to .env:
AGENT_ENGINE=projects/PROJECT_ID/locations/LOCATION/reasoningEngines/ID
ARTIFACT_SERVICE_URI=gs://BUCKET_NAME
```

See [Environment Variables](./environment-variables.md) for where to find each value.

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
  src/your_agent_name/
    agent.py              # LlmAgent configuration
    callbacks.py          # Agent callbacks
    prompt.py             # Agent prompts
    tools.py              # Custom tools
    server.py             # FastAPI development server
    utils/                # Utilities
      config.py           # Configuration and environment parsing
      observability.py    # OpenTelemetry setup
  tests/                  # Test suite
    conftest.py           # Shared fixtures
    test_*.py             # Unit and integration tests
  terraform/              # Infrastructure as code
    bootstrap/            # One-time CI/CD setup
    main/                 # Cloud Run deployment
  docs/                   # Documentation
  notebooks/              # Jupyter notebooks
  .env.example            # Environment template
  pyproject.toml          # Project configuration
  docker-compose.yml      # Local development
  Dockerfile              # Container image
  CLAUDE.md               # Project instructions
  README.md               # Main documentation
```

## Observability

OpenTelemetry exports traces to Cloud Trace and logs to Cloud Logging. Control log level: `LOG_LEVEL=DEBUG uv run server`

**View traces/logs:**
- [Cloud Trace](https://console.cloud.google.com/traces) | [Logs Explorer](https://console.cloud.google.com/logs)
- CLI: `gcloud logging tail "logName:projects/{PROJECT_ID}/logs/{AGENT_NAME}-otel-logs"`

See [Observability Guide](./observability.md) for details.
