"""Environment configuration models for application settings.

This module provides Pydantic models for type-safe environment variable validation
and configuration management.
"""

import json
import os
import sys
from typing import Literal

from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)


def initialize_environment[T: BaseModel](
    model_class: type[T],
    override_dotenv: bool = True,
    print_config: bool = True,
) -> T:
    """Initialize and validate environment configuration.

    Factory function that handles the common initialization pattern: load environment
    variables, validate with Pydantic model, handle errors, and optionally print
    configuration.

    Args:
        model_class: Pydantic model class to validate environment with.
        override_dotenv: Whether to override existing environment variables.
            Defaults to True for consistency and predictability.
        print_config: Whether to call print_config() method if it exists.
            Defaults to True.

    Returns:
        Validated environment configuration instance.

    Raises:
        SystemExit: If validation fails.

    Examples:
        >>> # Simple case (most common)
        >>> env = initialize_environment(ServerEnv)
        >>>
        >>> # Skip printing configuration
        >>> env = initialize_environment(ServerEnv, print_config=False)
    """
    load_dotenv(override=override_dotenv)

    # Load and validate environment configuration
    try:
        env = model_class.model_validate(os.environ)
    except ValidationError as e:
        print("\n❌ Environment validation failed:\n")
        print(e)
        sys.exit(1)

    # Print configuration for user verification if method exists
    if print_config and hasattr(env, "print_config"):
        env.print_config()

    return env


class ServerEnv(BaseModel):
    """Environment configuration for local server development and deployment.

    Provides configuration for both local development and Cloud Run deployment,
    with sensible defaults for local development.

    Attributes:
        google_cloud_project: GCP project ID for authentication and observability.
        google_cloud_location: Vertex AI region (e.g., us-central1).
        agent_name: Unique agent identifier for resources and logs.
        log_level: Logging verbosity level.
        serve_web_interface: Whether to serve the ADK web interface.
        reload_agents: Whether to reload agents on file changes (local dev only).
        agent_engine: Agent Engine instance ID for session and memory persistence.
        artifact_service_uri: GCS bucket URI for artifact storage.
        allow_origins: JSON array string of allowed CORS origins.
        host: Server host (127.0.0.1 for local, 0.0.0.0 for containers).
        port: Server port.
        otel_capture_content: OpenTelemetry message content capture setting.
    """

    google_cloud_project: str | None = Field(
        default=None,
        alias="GOOGLE_CLOUD_PROJECT",
        description="GCP project ID for authentication and observability",
    )

    google_cloud_location: str = Field(
        default="us-central1",
        alias="GOOGLE_CLOUD_LOCATION",
        description="Vertex AI region (e.g., us-central1)",
    )

    agent_name: str = Field(
        ...,
        alias="AGENT_NAME",
        description="Unique agent identifier for resources and logs",
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Logging verbosity level",
    )

    serve_web_interface: bool = Field(
        default=False,
        alias="SERVE_WEB_INTERFACE",
        description="Whether to serve the ADK web interface",
    )

    reload_agents: bool = Field(
        default=False,
        alias="RELOAD_AGENTS",
        description="Whether to reload agents on file changes (local dev only)",
    )

    agent_engine: str | None = Field(
        default=None,
        alias="AGENT_ENGINE",
        description="Agent Engine instance ID for session and memory persistence",
    )
    
    database_url: str | None = Field(
        default=None,
        alias="DATABASE_URL",
        description="Database URL for session storage (e.g., postgresql://...)",
    )

    openrouter_api_key: str | None = Field(
        default=None,
        alias="OPENROUTER_API_KEY",
        description="OpenRouter API key for LiteLLM integration",
    )

    artifact_service_uri: str | None = Field(
        default=None,
        alias="ARTIFACT_SERVICE_URI",
        description="GCS bucket URI for artifact storage",
    )

    allow_origins: str = Field(
        default='["http://127.0.0.1", "http://127.0.0.1:8000"]',
        alias="ALLOW_ORIGINS",
        description="JSON array string of allowed CORS origins",
    )

    host: str = Field(
        default="127.0.0.1",
        alias="HOST",
        description="Server host (127.0.0.1 for local, 0.0.0.0 for containers)",
    )

    port: int = Field(
        default=8000,
        alias="PORT",
        description="Server port",
    )

    otel_capture_content: bool = Field(
        default=False,
        alias="OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
        description="OpenTelemetry message content capture setting",
    )

    model_config = ConfigDict(
        populate_by_name=True,  # Allow both field names and aliases
        extra="ignore",  # Ignore extra env vars (system vars, etc.)
    )

    def print_config(self) -> None:
        """Print server configuration for user verification."""
        print("\n\n✅ Environment variables loaded for server:\n")
        print(f"GOOGLE_CLOUD_PROJECT:  {self.google_cloud_project}")
        print(f"GOOGLE_CLOUD_LOCATION: {self.google_cloud_location}")
        print(f"AGENT_NAME:            {self.agent_name}")
        print(f"LOG_LEVEL:             {self.log_level}")
        print(f"SERVE_WEB_INTERFACE:   {self.serve_web_interface}")
        print(f"RELOAD_AGENTS:         {self.reload_agents}")
        print(f"AGENT_ENGINE:          {self.agent_engine}")
        print(f"DATABASE_URL:          {self.database_url}")
        print(f"OPENROUTER_API_KEY:    {'********' if self.openrouter_api_key else None}")
        print(f"ARTIFACT_SERVICE_URI:  {self.artifact_service_uri}")
        print(f"HOST:                  {self.host}")
        print(f"PORT:                  {self.port}")
        print(f"ALLOW_ORIGINS:         {self.allow_origins}")
        print(f"OTEL_CAPTURE_CONTENT:  {self.otel_capture_content}\n\n")

    @property
    def agent_engine_uri(self) -> str | None:
        """Agent Engine URI with protocol prefix."""
        return f"agentengine://{self.agent_engine}" if self.agent_engine else None
    
    @property
    def session_uri(self) -> str | None:
        """Session service URI (Database or Agent Engine)."""
        if self.database_url:
            return self.database_url
        return self.agent_engine_uri

    @property
    def allow_origins_list(self) -> list[str]:
        """Parse allow_origins JSON string to list.

        Returns:
            List of allowed origin strings.

        Raises:
            ValueError: If JSON parsing fails or result is not a list of strings.
        """
        try:
            origins = json.loads(self.allow_origins)
            if not isinstance(origins, list) or not all(
                isinstance(o, str) for o in origins
            ):
                msg = "ALLOW_ORIGINS must be a JSON array of strings"
                raise ValueError(msg)
            return origins
        except json.JSONDecodeError as e:
            msg = f"Failed to parse ALLOW_ORIGINS as JSON: {e}"
            raise ValueError(msg) from e
