# Google ADK on Bare Metal

## Project Overview

**Google ADK on Bare Metal** is a production-ready template designed for building and deploying AI agents using the Google Agent Development Kit (ADK) on self-hosted infrastructure. It removes cloud provider lock-in by providing a clean, performant, and observable foundation that runs on bare metal, VPS, or private clouds.

### Key Technologies
*   **Language:** Python 3.13+
*   **Framework:** Google ADK (`google-adk`)
*   **Model Interface:** LiteLLM (supports Google, OpenRouter, etc.)
*   **Server:** FastAPI
*   **Database:** PostgreSQL (via `asyncpg`)
*   **Observability:** OpenTelemetry (OTel) with Langfuse support
*   **Infrastructure:** Docker, Docker Compose

## Building and Running

### Prerequisites
*   Python 3.13+
*   [`uv`](https://github.com/astral-sh/uv) (Package Manager)
*   Docker & Docker Compose (for containerized deployment)

### Setup
1.  **Configure Environment:**
    Copy `.env.example` to `.env` and set the required variables:
    *   `AGENT_NAME`: Unique ID for the agent.
    *   `DATABASE_URL`: Postgres connection string.
    *   `OPENROUTER_API_KEY` / `GOOGLE_API_KEY`: LLM API keys.

2.  **Install Dependencies:**
    ```bash
    uv sync
    ```

### Execution Commands

| Task | Command | Description |
| :--- | :--- | :--- |
| **Run Locally** | `uv run python -m agent.server` | Starts the agent server on localhost:8080. |
| **Run (Script)**| `uv run server` | Alternative command using the project script entry point. |
| **Docker Run** | `docker compose up --build -d` | Builds and starts the agent in a Docker container. |
| **Test** | `uv run pytest` | Runs the test suite. |
| **Lint** | `uv run ruff check` | Runs linter checks. |
| **Format** | `uv run ruff format` | Formats code using Ruff. |
| **Type Check** | `uv run mypy .` | Runs static type checking. |

## Development Conventions

### Code Structure
*   **`src/agent/`**: Contains the core agent logic.
    *   `agent.py`: Defines the `root_agent` and ADK application configuration.
    *   `server.py`: FastAPI server entry point with OTel instrumentation.
    *   `prompt.py`: Manages agent prompts and instructions.
    *   `tools.py`: Helper tools for the agent.

*   **`tests/`**: Unit and integration tests.

### Code Quality
Before committing anything or creating a Pull Request, you **must** ensure all local checks pass. This includes running formatting, linting, type checks, and tests. The CI pipeline will run these same checks and fail if they are not satisfied:

1.  **Format Code:** `uv run ruff format`
2.  **Lint Code:** `uv run ruff check`
3.  **Type Check:** `uv run mypy .`
4.  **Run Tests:** `uv run pytest --cov=src`

**⚠️ CRITICAL WARNING:** If you modify code to fix an error reported by one of these tools (e.g., adding type hints for `mypy` or `# noqa` comments for `ruff check`), you **MUST** re-run the entire suite of checks starting from `ruff format`. Fixing an error for one tool frequently breaks the rules of another (especially formatting). Do not commit until all checks pass consecutively without any further file modifications.

Ensure all steps pass locally before staging and committing files to avoid CI failures.

### Testing Standards for AI Assistants
When asked to write or modify tests, you **MUST** adhere to the following strict guidelines derived from the ADK philosophy:

1.  **Real Code Over Mocks**:
    *   **Do not mock** internal logic (e.g., `LlmAgent`, `Prompt`, `Tool`). Use the real classes.
    *   **Only mock** external boundaries (e.g., `LiteLLM`, `asyncpg`, `Network APIs`).
    *   **Why?** This ensures we test the integration of components, not just isolated units.

2.  **Pytest Best Practices**:
    *   Use **fixtures** (`conftest.py`) for setup/teardown.
    *   Use **`@pytest.mark.parametrize`** for testing multiple inputs/outputs.
    *   Use **`tmp_path` fixture** for any file system operations.
    *   **Strict Mocking**: Always use `create_autospec(spec_set=True)` to ensure mocks match the actual API.

3.  **Test Coverage**:
    *   Every new feature **must** have a corresponding test.
    *   Tests must cover both the "Happy Path" (success) and "Edge Cases" (failure/errors).

### Deployment
*   **Containerization:** The `Dockerfile` provides a multi-stage build optimized for production.
*   **CI/CD:** GitHub Actions workflows (`.github/workflows/`) handle testing, linting, and publishing Docker images to GHCR.