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
Before creating a Pull Request, you **must** ensure all local checks pass. The CI pipeline will run these same checks:

1.  **Format Code:** `uv run ruff format`
2.  **Lint Code:** `uv run ruff check`
3.  **Type Check:** `uv run mypy .`
4.  **Run Tests:** `uv run pytest --cov=src`

Ensure all steps pass locally to avoid CI failures.

### Deployment
*   **Containerization:** The `Dockerfile` provides a multi-stage build optimized for production.
*   **CI/CD:** GitHub Actions workflows (`.github/workflows/`) handle testing, linting, and publishing Docker images to GHCR.
