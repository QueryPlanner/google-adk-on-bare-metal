## Open Services Agent Starter Pack

This repo is a **production-ready template** for building and deploying AI agents on your own infrastructure using **Google ADK**, **LiteLLM**, and **Postgres**.

**Key Features:**
- 🐳 **Self-Hosted Ready**: Docker & Compose setup included for private infrastructure.
- 🧩 **Extensible**: Structured for adding Tools and Sub-Agents easily.
- 💾 **Persistent**: Postgres-backed sessions out of the box.
- 🚀 **Modern Stack**: Python 3.13, `uv`, `fastapi`, `asyncpg`.

## Quickstart (Local Dev)

### Prerequisites

- Python **3.13+**
- [`uv`](https://github.com/astral-sh/uv)
- A Postgres connection string (Neon works great)
- An OpenRouter or Google API key

### 1) Configure env

Create `.env`:

- **`AGENT_NAME`**: a unique identifier for your agent (e.g. `my-awesome-agent`)
- **`GOOGLE_API_KEY`**: your AI Studio key
- **`OPENROUTER_API_KEY`**: your OpenRouter key (optional, for LiteLLM models)
- **`DATABASE_URL`**: a Postgres URL (can be the standard `postgresql://...` form)

Notes:
- We normalize Postgres URLs for SQLAlchemy+asyncpg automatically (e.g.
  `sslmode=require` becomes `ssl=require`).

### 2) Install deps

```bash
uv sync
```

### 3) Run the Agent Platform

```bash
uv run python -m agent_foundation.server
```

Then open `http://127.0.0.1:8080`.

## Deployment (Docker)

For production or clean local environments, use Docker.

👉 **[Read the Deployment Guide](docs/DEPLOYMENT.md)**

## Customization

- **Add an Agent**: Create a new folder in `src/` (if multi-agent) or edit `src/agent_foundation/`.
- **Tools**: Add functions to `src/agent_foundation/tools.py`
- **Main Logic**: Edit `src/agent_foundation/agent.py`

## Why not `adk web --session_service_uri ...`?

ADK’s CLI defaults to local SQLite session storage unless you pass
`--session_service_uri`. This repo provides a custom `server.py` so:

- you don’t have to remember a long CLI command
- sessions always go to Postgres (once `DATABASE_URL` is set)
- multiple agents are served automatically.

## Development & Observability

- [Development Guide](docs/development.md) - Local workflow and code quality
- [Observability Guide](docs/base-infra/observability.md) - Langfuse and Tracing setup
- [Environment Variables](docs/base-infra/environment-variables.md) - Full reference
