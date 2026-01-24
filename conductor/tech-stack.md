# Tech Stack - Open Services Agent Starter Pack

## Core Infrastructure
- **Language:** Python 3.13 (Managed by `uv`)
- **Agent Framework:** `google-adk` (version 1.21.0)
- **Model Interface:** `LiteLLM` (for multi-provider support)
- **Web Server:** FastAPI (via `agent_foundation.server`)
- **Database:** PostgreSQL (via `asyncpg`)

## Development Tooling
- **Package Manager:** `uv`
- **Linting & Formatting:** `ruff`
- **Static Analysis:** `mypy` (Strict mode)
- **Testing:** `pytest` (Asyncio support, 100% coverage goal)

## Observability & Deployment
- **Logging:** Standard Python Logging (Cloud Logging optional)
- **Tracing:** `OpenTelemetry` (OTLP, gRPC)
- **Containerization:** Docker & Docker Compose
- **Init System:** systemd (for bare-metal deployment)
