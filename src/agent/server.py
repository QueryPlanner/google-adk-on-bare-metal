"""FastAPI server module.

This module provides a FastAPI server for ADK agents with comprehensive observability
features using custom OpenTelemetry setup. Includes an optional ADK web interface for
interactive agent testing.
"""

import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from google.adk.cli.fast_api import get_fast_api_app
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from .artifact_storage import (
    ARTIFACT_STORAGE_ERROR_MESSAGE,
    ArtifactStorageError,
    prepare_artifact_storage,
)
from .health import DatabaseReadinessProbe, live
from .utils import (
    ObservabilityEnv,
    ServerEnv,
    configure_otel_resource,
    initialize_environment,
    setup_logging,
)

logger = logging.getLogger("agent.server")


def create_app(env: ServerEnv | None = None) -> FastAPI:
    """Create a configured ADK FastAPI application with durable artifacts."""
    server_env = env or initialize_environment(ServerEnv, print_config=False)
    observability_env = initialize_environment(ObservabilityEnv, print_config=False)

    configure_otel_resource(
        agent_name=server_env.agent_name,
        settings=observability_env,
    )
    GoogleADKInstrumentor().instrument()
    setup_logging(log_level=server_env.log_level)

    configured_agent_dir = (
        server_env.agent_dir or Path(__file__).resolve().parent.parent
    )
    artifact_storage = prepare_artifact_storage(configured_agent_dir)

    session_uri = server_env.session_uri
    if session_uri and session_uri.startswith("postgresql://"):
        session_uri = session_uri.replace("postgresql://", "postgresql+asyncpg://", 1)

    session_db_kwargs = {
        "pool_pre_ping": server_env.db_pool_pre_ping,
        "pool_recycle": server_env.db_pool_recycle,
        "pool_size": server_env.db_pool_size,
        "max_overflow": server_env.db_max_overflow,
        "pool_timeout": server_env.db_pool_timeout,
    }

    app = get_fast_api_app(
        agents_dir=str(artifact_storage.agents_dir),
        session_service_uri=session_uri,
        session_db_kwargs=session_db_kwargs,
        artifact_service_uri=artifact_storage.artifact_service_uri,
        # Memory service does not yet support Postgres scheme in ADK
        memory_service_uri=None,
        allow_origins=server_env.allow_origins_list,
        web=server_env.serve_web_interface,
        reload_agents=server_env.reload_agents,
    )
    readiness_probe = DatabaseReadinessProbe(
        database_url=server_env.database_url,
        attempt_timeout=server_env.db_readiness_probe_timeout,
    )

    async def ready() -> JSONResponse:
        """Report configured PostgreSQL connectivity readiness."""
        return await readiness_probe()

    app.get("/live")(live)
    app.get("/ready")(ready)
    return app


def main() -> None:
    """Run the FastAPI server.

    Starts the ADK agent server. Features include:
    - Environment variable loading and validation via Pydantic
    - Custom OpenTelemetry setup for resource attributes
    - Optional ADK web interface for interactive agent testing
    - Session and memory persistence
    - CORS configuration

    Environment Variables:
        AGENT_DIR: Path to agent source directory (default: auto-detect from __file__)
        AGENT_NAME: Unique service identifier (required)
        LOG_LEVEL: Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        SERVE_WEB_INTERFACE: Whether to serve the web interface (true/false)
        RELOAD_AGENTS: Whether to reload agents on file changes (true/false)
        AGENT_ENGINE: Agent Engine instance for session and memory
        DATABASE_URL: Postgres URL for session and memory
        DB_READINESS_PROBE_TIMEOUT: Maximum seconds for one HTTP database probe
        OPENROUTER_API_KEY: Key for LiteLLM/OpenRouter
        ALLOW_ORIGINS: JSON array string of allowed CORS origins
        HOST: Server host (default: 127.0.0.1, set to 0.0.0.0 for containers)
        PORT: Server port (default: 8080)
    """
    env = initialize_environment(ServerEnv, print_config=False)
    try:
        app = create_app(env)
    except ArtifactStorageError:
        logger.error(ARTIFACT_STORAGE_ERROR_MESSAGE)
        raise SystemExit(1) from None

    env.print_config()
    uvicorn.run(
        app,
        host=env.host,
        port=env.port,
    )

    return


if __name__ == "__main__":
    main()
