"""Environment configuration models for application settings.

This module provides Pydantic models for type-safe environment variable validation
and configuration management.
"""

import json
import warnings
from typing import Any, Literal, Self, cast

from pydantic import (
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_LANGFUSE_BASE_URL = "https://cloud.langfuse.com"
_EXAMPLE_SECRET_VALUES = {
    "DATABASE_URL": frozenset(
        {
            "changethis",
            "postgresql://user:pass@host:port/dbname?ssl=require",
        }
    ),
    "OPENROUTER_API_KEY": frozenset(
        {
            "changethis",
            "your-openrouter-key-here",
        }
    ),
}


class SettingsConfigurationError(Exception):
    """Safe configuration failure whose message never contains input values."""


def _blank_optional_value(value: Any) -> Any:
    """Normalize a source-selected blank without falling through to lower sources."""
    raw_value = value.get_secret_value() if isinstance(value, SecretStr) else value
    if isinstance(raw_value, str) and not raw_value.strip():
        return None
    return value


def initialize_environment[T: BaseSettings](
    model_class: type[T],
    override_dotenv: bool | None = None,
    print_config: bool = True,
) -> T:
    """Initialize and validate environment configuration.

    Construct a settings model using its configured sources, handle validation errors,
    and optionally print a redacted configuration summary.

    Args:
        model_class: Pydantic settings class to construct.
        override_dotenv: Deprecated compatibility argument. ``False`` and ``None``
            use safe source precedence. ``True`` warns and is ignored because
            process environment variables must always win over dotenv.
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
    if override_dotenv is True:
        warnings.warn(
            "override_dotenv is deprecated and ignored; process environment "
            "variables always take priority over dotenv",
            DeprecationWarning,
            stacklevel=2,
        )

    try:
        env = model_class()
    except (SettingsConfigurationError, ValidationError) as e:
        print("\n❌ Environment validation failed:\n")
        print(e)
        raise SystemExit(1) from e

    # Print configuration for user verification if method exists
    if print_config and hasattr(env, "print_config"):
        env.print_config()

    return env


class RedactedBaseSettings(BaseSettings):
    """Base settings model whose structured validation errors omit input values."""

    model_config = SettingsConfigDict(
        populate_by_name=True,
        extra="ignore",
        hide_input_in_errors=True,
    )

    def __init__(self, **values: Any) -> None:
        """Validate settings and rebuild failures without their original inputs."""
        sanitized_error: ValidationError | None = None
        try:
            super().__init__(**values)
        except ValidationError as error:
            sanitized_error = ValidationError.from_exception_data(
                title=error.title,
                line_errors=cast(list[Any], error.errors(include_input=False)),
                hide_input=True,
            )

        if sanitized_error is not None:
            raise sanitized_error from None


def _check_example_secret(var_name: str, value: SecretStr | None) -> None:
    """Reject a documented example/default value without echoing it."""
    if value is None:
        return

    normalized_value = value.get_secret_value().strip().casefold()
    if normalized_value in _EXAMPLE_SECRET_VALUES[var_name]:
        msg = f"{var_name} must not use an example or default secret"
        raise SettingsConfigurationError(msg)


class AgentRuntimeEnv(RedactedBaseSettings):
    """Process-only settings consumed while constructing the ADK agent.

    The supported ADK loaders populate provider variables before importing the
    agent. Keeping this model process-only prevents a direct package import from
    searching for or mutating dotenv values on its own.
    """

    root_agent_model: str = Field(
        default="gemini-2.5-flash",
        alias="ROOT_AGENT_MODEL",
        description="Model used by the root ADK agent",
    )
    openrouter_api_key: SecretStr | None = Field(
        default=None,
        alias="OPENROUTER_API_KEY",
        description="OpenRouter API key for LiteLLM integration",
    )
    google_api_key: SecretStr | None = Field(
        default=None,
        alias="GOOGLE_API_KEY",
        description="Google AI Studio API key used by the native Gemini client",
    )

    model_config = SettingsConfigDict(
        env_file=None,
        populate_by_name=True,
        extra="ignore",
        hide_input_in_errors=True,
    )

    @field_validator(
        "openrouter_api_key",
        "google_api_key",
        mode="before",
    )
    @classmethod
    def _normalize_blank_provider_keys(cls, value: Any) -> Any:
        """Treat source-selected blank provider keys as unconfigured."""
        return _blank_optional_value(value)

    @model_validator(mode="after")
    def _reject_example_secrets(self) -> Self:
        """Reject known provider-key placeholders."""
        _check_example_secret("OPENROUTER_API_KEY", self.openrouter_api_key)
        return self


class ObservabilityEnv(RedactedBaseSettings):
    """Settings needed before ADK constructs the FastAPI application."""

    telemetry_namespace: str = Field(
        default="local",
        alias="TELEMETRY_NAMESPACE",
        description="OpenTelemetry service namespace",
    )
    service_revision: str = Field(
        default="local",
        alias="K_REVISION",
        description="Service version or deployment revision",
    )
    langfuse_public_key: SecretStr | None = Field(
        default=None,
        alias="LANGFUSE_PUBLIC_KEY",
        description="Langfuse public key",
    )
    langfuse_secret_key: SecretStr | None = Field(
        default=None,
        alias="LANGFUSE_SECRET_KEY",
        description="Langfuse secret key",
    )
    langfuse_base_url: str = Field(
        default=_DEFAULT_LANGFUSE_BASE_URL,
        alias="LANGFUSE_BASE_URL",
        description="Langfuse API base URL",
    )
    otel_exporter_otlp_endpoint: str | None = Field(
        default=None,
        alias="OTEL_EXPORTER_OTLP_ENDPOINT",
        description="Explicit OTLP exporter endpoint",
    )
    otel_exporter_otlp_protocol: str | None = Field(
        default=None,
        alias="OTEL_EXPORTER_OTLP_PROTOCOL",
        description="Explicit OTLP exporter protocol",
    )
    otel_exporter_otlp_headers: SecretStr | None = Field(
        default=None,
        alias="OTEL_EXPORTER_OTLP_HEADERS",
        description="Explicit OTLP exporter authentication headers",
    )
    otel_capture_message_content: bool = Field(
        default=False,
        alias="OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
        description="Whether GenAI instrumentation captures message content",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
        hide_input_in_errors=True,
    )

    @field_validator(
        "langfuse_public_key",
        "langfuse_secret_key",
        "otel_exporter_otlp_endpoint",
        "otel_exporter_otlp_protocol",
        "otel_exporter_otlp_headers",
        mode="before",
    )
    @classmethod
    def _normalize_blank_optional_fields(cls, value: Any) -> Any:
        """Treat source-selected blank optional exporter values as unconfigured."""
        return _blank_optional_value(value)

    @field_validator("langfuse_base_url", mode="before")
    @classmethod
    def _normalize_blank_langfuse_base_url(cls, value: Any) -> Any:
        """Use the local default when the selected Langfuse URL is blank."""
        return (
            _DEFAULT_LANGFUSE_BASE_URL
            if _blank_optional_value(value) is None
            else value
        )


class ServerEnv(RedactedBaseSettings):
    """Environment configuration for local server development and deployment.

    Provides configuration for local development and bare-metal deployment, with
    sensible defaults for local development.

    Attributes:
        agent_name: Unique agent identifier for resources and logs.
        log_level: Logging verbosity level.
        serve_web_interface: Whether to serve the ADK web interface.
        reload_agents: Whether to reload agents on file changes (local dev only).
        agent_engine: Agent Engine instance ID for session and memory persistence.
        database_url: Database URL used for persistent session storage.
        openrouter_api_key: OpenRouter key validated before server startup.
        allow_origins: JSON array string of allowed CORS origins.
        host: Server host (127.0.0.1 for local, 0.0.0.0 for containers).
        port: Server port.
        agent_dir: Directory containing ADK agent packages.
    """

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

    database_url: SecretStr | None = Field(
        default=None,
        alias="DATABASE_URL",
        description="Database URL for session storage (e.g., postgresql://...)",
    )

    db_pool_pre_ping: bool = Field(
        default=True,
        alias="DB_POOL_PRE_PING",
        description="Validate DB connections before use",
    )

    db_pool_recycle: int = Field(
        default=1800,
        alias="DB_POOL_RECYCLE",
        description="Recycle connections after this many seconds",
    )

    db_pool_size: int = Field(
        default=5,
        alias="DB_POOL_SIZE",
        description="Number of connections to keep open inside the connection pool",
    )

    db_max_overflow: int = Field(
        default=10,
        alias="DB_MAX_OVERFLOW",
        description="Number of connections to allow beyond pool_size",
    )

    db_pool_timeout: int = Field(
        default=30,
        alias="DB_POOL_TIMEOUT",
        description="Seconds to wait before giving up on getting a connection",
    )

    openrouter_api_key: SecretStr | None = Field(
        default=None,
        alias="OPENROUTER_API_KEY",
        description="OpenRouter API key for LiteLLM integration",
    )

    agent_dir: str | None = Field(
        default=None,
        alias="AGENT_DIR",
        description="Directory containing ADK agent packages",
    )

    allow_origins: str = Field(
        default='["http://127.0.0.1", "http://127.0.0.1:8080"]',
        alias="ALLOW_ORIGINS",
        description="JSON array string of allowed CORS origins",
    )

    host: str = Field(
        default="127.0.0.1",
        alias="HOST",
        description="Server host (127.0.0.1 for local, 0.0.0.0 for containers)",
    )

    port: int = Field(
        default=8080,
        alias="PORT",
        description="Server port",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,  # Allow both field names and aliases
        extra="ignore",  # Ignore extra env vars (system vars, etc.)
        hide_input_in_errors=True,
    )

    @field_validator(
        "database_url",
        "openrouter_api_key",
        mode="before",
    )
    @classmethod
    def _normalize_blank_server_secrets(cls, value: Any) -> Any:
        """Treat source-selected blank optional secrets as unconfigured."""
        return _blank_optional_value(value)

    @model_validator(mode="after")
    def _reject_example_secrets(self) -> Self:
        """Reject known example secrets before they reach a deployment."""
        _check_example_secret("DATABASE_URL", self.database_url)
        _check_example_secret("OPENROUTER_API_KEY", self.openrouter_api_key)
        return self

    def print_config(self) -> None:
        """Print a redacted server configuration for user verification."""
        print("\n\n✅ Environment variables loaded for server:\n")
        print(f"AGENT_NAME:            {self.agent_name}")
        print(f"LOG_LEVEL:             {self.log_level}")
        print(f"SERVE_WEB_INTERFACE:   {self.serve_web_interface}")
        print(f"RELOAD_AGENTS:         {self.reload_agents}")
        print(f"AGENT_ENGINE:          {self.agent_engine}")
        masked_database_url = str(self.database_url) if self.database_url else None
        print(f"DATABASE_URL:          {masked_database_url}")
        if self.database_url:
            print(f"DB_POOL_PRE_PING:      {self.db_pool_pre_ping}")
            print(f"DB_POOL_RECYCLE:       {self.db_pool_recycle}")
            print(f"DB_POOL_SIZE:          {self.db_pool_size}")
            print(f"DB_MAX_OVERFLOW:       {self.db_max_overflow}")
            print(f"DB_POOL_TIMEOUT:       {self.db_pool_timeout}")
        masked_key = str(self.openrouter_api_key) if self.openrouter_api_key else None
        print(f"OPENROUTER_API_KEY:    {masked_key}")
        print(f"AGENT_DIR:             {self.agent_dir}")
        print(f"HOST:                  {self.host}")
        print(f"PORT:                  {self.port}")
        print(f"ALLOW_ORIGINS:         {self.allow_origins}\n\n")

    @property
    def agent_engine_uri(self) -> str | None:
        """Agent Engine URI with protocol prefix."""
        return f"agentengine://{self.agent_engine}" if self.agent_engine else None

    @property
    def session_uri(self) -> str | None:
        """Session service URI (Database or Agent Engine)."""
        if self.database_url:
            database_url = self.database_url.get_secret_value()
            # asyncpg requires 'ssl=require' instead of 'sslmode=require'
            # Also removing channel_binding as it causes TypeError with current
            # sqlalchemy/asyncpg setup
            return database_url.replace("sslmode=require", "ssl=require").replace(
                "&channel_binding=require", ""
            )
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
