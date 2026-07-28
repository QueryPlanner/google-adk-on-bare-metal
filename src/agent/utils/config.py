"""Environment configuration models for application settings.

This module provides Pydantic models for type-safe environment variable validation
and configuration management.
"""

import ipaddress
import json
import os
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self, cast
from urllib.parse import (
    ParseResult,
    SplitResult,
    unquote,
    unquote_plus,
    urlparse,
    urlsplit,
    urlunsplit,
)

from pydantic import (
    Field,
    PrivateAttr,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, DotEnvSettingsSource, SettingsConfigDict

_DEFAULT_LANGFUSE_BASE_URL = "https://cloud.langfuse.com"
_DEFAULT_OTLP_TRACE_TIMEOUT_SECONDS = 2.0
_MAX_OTLP_CA_BYTES = 1024 * 1024
_MAX_GATEWAY_TOKEN_BYTES = 4 * 1024
_HTTP_HEADER_NAME = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_HTTP_HEADER_VALUE = re.compile(r"^[\x20-\x7e]*$")
_OTLP_HEADER_RAW_VALUE = re.compile(r"^[\x21\x23-\x2b\x2d-\x3a\x3c-\x5b\x5d-\x7e]*$")
_GATEWAY_BEARER_TOKEN = re.compile(r"^[0-9A-Za-z\-._~+/]+={0,}$")
_DNS_LABEL = re.compile(r"^(?!-)[0-9A-Za-z-]{1,63}(?<!-)$")
_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_SUPPORTED_OTLP_SETTINGS = frozenset(
    {
        "OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
        "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
        "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT",
    }
)
_OTLP_HTTP_CREDENTIAL_PROVIDER_SETTINGS = frozenset(
    {
        "OTEL_PYTHON_EXPORTER_OTLP_HTTP_CREDENTIAL_PROVIDER",
        "OTEL_PYTHON_EXPORTER_OTLP_HTTP_TRACES_CREDENTIAL_PROVIDER",
    }
)
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
_DATABASE_READINESS_MAX_TIMEOUT = 3600.0
_DATABASE_READINESS_MAX_INTERVAL = 60.0
_DATABASE_READINESS_MAX_PROBE_TIMEOUT = 2.0
_SHARED_ASYNCPG_QUERY_KEYS = frozenset(
    {
        "gsslib",
        "host",
        "krbsrvname",
        "passfile",
        "port",
        "target_session_attrs",
    }
)
_TARGETED_ASYNCPG_QUERY_KEYS = frozenset(
    {
        "channel_binding",
        "ssl",
        "sslmode",
    }
)


def _parse_postgresql_url(database_url: str) -> tuple[SplitResult, str]:
    """Parse a supported PostgreSQL URL without including its value in errors."""
    if not database_url or database_url != database_url.strip():
        msg = "DATABASE_URL must be a non-blank PostgreSQL URL"
        raise ValueError(msg)

    try:
        parsed = urlsplit(database_url)
        _ = parsed.port
    except ValueError:
        msg = "DATABASE_URL must be a valid PostgreSQL URL"
        raise ValueError(msg) from None

    scheme = parsed.scheme.casefold()
    supported_schemes = {"postgres", "postgresql", "postgresql+asyncpg"}
    scheme_separator = database_url.find(":")
    has_url_authority_marker = scheme_separator >= 0 and database_url[
        scheme_separator + 1 :
    ].startswith("//")
    if (
        scheme not in supported_schemes
        or not has_url_authority_marker
        or parsed.fragment
    ):
        msg = "DATABASE_URL must use a supported PostgreSQL URL"
        raise ValueError(msg)

    return parsed, scheme


def _normalize_postgresql_query(
    query: str,
    *,
    allow_host_option: bool,
    allow_port_option: bool,
    ssl_key: Literal["ssl", "sslmode"],
) -> str:
    """Return query options with identical direct and SQLAlchemy semantics."""
    query_parts = query.split("&") if query else []
    seen_keys: set[str] = set()
    has_ssl = False
    has_sslmode = False
    normalized_query_parts: list[str] = []

    for query_part in query_parts:
        raw_key, separator, raw_value = query_part.partition("=")
        key = unquote_plus(raw_key)

        if key not in _SHARED_ASYNCPG_QUERY_KEYS | _TARGETED_ASYNCPG_QUERY_KEYS:
            msg = "DATABASE_URL contains an unsupported PostgreSQL option"
            raise ValueError(msg)
        if key in seen_keys:
            msg = "DATABASE_URL contains an ambiguous PostgreSQL option"
            raise ValueError(msg)
        seen_keys.add(key)
        if not separator:
            msg = "DATABASE_URL contains an invalid PostgreSQL option"
            raise ValueError(msg)
        if (key == "host" and not allow_host_option) or (
            key == "port" and not allow_port_option
        ):
            msg = "DATABASE_URL contains an ambiguous PostgreSQL address option"
            raise ValueError(msg)

        if key not in _TARGETED_ASYNCPG_QUERY_KEYS:
            normalized_query_parts.append(query_part)
            continue

        if key == "ssl":
            has_ssl = True
            normalized_query_parts.append(
                query_part if ssl_key == "ssl" else f"sslmode={raw_value}"
            )
        elif key == "sslmode":
            has_sslmode = True
            normalized_query_parts.append(
                query_part if ssl_key == "sslmode" else f"ssl={raw_value}"
            )
        elif unquote_plus(raw_value) != "require":
            msg = "DATABASE_URL contains an unsupported channel_binding option"
            raise ValueError(msg)

    if has_ssl and has_sslmode:
        msg = "DATABASE_URL must not set both ssl and sslmode"
        raise ValueError(msg)

    return "&".join(normalized_query_parts)


def _rebuild_postgresql_url(
    parsed: SplitResult,
    *,
    query: str,
    scheme: str,
) -> str:
    """Rebuild a URL without collapsing an empty authority used for sockets."""
    if parsed.netloc:
        return urlunsplit((scheme, parsed.netloc, parsed.path, query, ""))

    normalized_url = f"{scheme}://{parsed.path}"
    return f"{normalized_url}?{query}" if query else normalized_url


def _normalize_asyncpg_database_url(database_url: str) -> str:
    """Normalize a supported PostgreSQL URL for a direct asyncpg connection."""
    parsed, scheme = _parse_postgresql_url(database_url)
    normalized_scheme = "postgresql" if scheme == "postgresql+asyncpg" else scheme
    return _rebuild_postgresql_url(
        parsed,
        query=_normalize_postgresql_query(
            parsed.query,
            allow_host_option=parsed.hostname is None,
            allow_port_option=parsed.port is None,
            ssl_key="sslmode",
        ),
        scheme=normalized_scheme,
    )


def _normalize_sqlalchemy_database_url(database_url: str) -> str:
    """Normalize a PostgreSQL URL for SQLAlchemy's asyncpg dialect."""
    parsed, scheme = _parse_postgresql_url(database_url)
    normalized_scheme = "postgresql" if scheme == "postgres" else scheme
    return _rebuild_postgresql_url(
        parsed,
        query=_normalize_postgresql_query(
            parsed.query,
            allow_host_option=parsed.hostname is None,
            allow_port_option=parsed.port is None,
            ssl_key="ssl",
        ),
        scheme=normalized_scheme,
    )


class SettingsConfigurationError(Exception):
    """Safe configuration failure whose message never contains input values."""


def _blank_optional_value(value: Any) -> Any:
    """Normalize a source-selected blank without falling through to lower sources."""
    raw_value = value.get_secret_value() if isinstance(value, SecretStr) else value
    if isinstance(raw_value, str) and not raw_value.strip():
        return None
    return value


def _reject_non_trace_otlp_settings(values: Mapping[Any, Any]) -> None:
    """Reject every unvalidated OTLP variable without echoing its key."""
    for key, value in values.items():
        normalized_key = str(key).upper()
        if normalized_key in _OTLP_HTTP_CREDENTIAL_PROVIDER_SETTINGS:
            raw_value = (
                value.get_secret_value() if isinstance(value, SecretStr) else value
            )
            if isinstance(raw_value, str) and not raw_value.strip():
                continue
            msg = "Custom OTLP HTTP credential providers are not supported"
            raise SettingsConfigurationError(msg)
        if (
            normalized_key.startswith("OTEL_EXPORTER_OTLP_")
            and normalized_key not in _SUPPORTED_OTLP_SETTINGS
        ):
            msg = (
                "Unsupported OTLP exporter setting; use only the trace endpoint, "
                "protocol, headers, certificate, and timeout settings"
            )
            raise SettingsConfigurationError(msg)


def _parse_safe_http_endpoint(endpoint: str) -> ParseResult:
    """Return a validated HTTP endpoint without including it in failures."""
    try:
        parsed = urlparse(endpoint)
        port = parsed.port
        decoded_endpoint = unquote(endpoint)
    except (UnicodeError, ValueError):
        msg = "OTLP endpoints must be valid HTTP(S) URLs"
        raise SettingsConfigurationError(msg) from None

    if (
        endpoint != endpoint.strip()
        or _PERCENT_ESCAPE.search(endpoint) is not None
        or any(
            character.isspace() or ord(character) < 32 or ord(character) == 127
            for character in decoded_endpoint
        )
        or "\\" in decoded_endpoint
        or parsed.scheme.casefold() not in {"http", "https"}
        or not parsed.hostname
        or not _is_valid_endpoint_hostname(parsed.hostname)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.params
        or parsed.query
        or parsed.fragment
        or port == 0
    ):
        msg = (
            "OTLP endpoints must be HTTP(S) URLs without credentials, "
            "queries, fragments, or control characters"
        )
        raise SettingsConfigurationError(msg)

    hostname = parsed.hostname
    if parsed.scheme.casefold() == "http" and not _is_loopback_host(hostname):
        msg = "Plaintext OTLP endpoints are restricted to loopback hosts"
        raise SettingsConfigurationError(msg)

    return parsed


def _is_valid_endpoint_hostname(hostname: str) -> bool:
    """Return whether a URL hostname is an IP literal or a valid DNS name."""
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        normalized_hostname = hostname.rstrip(".")
        return (
            bool(normalized_hostname)
            and not hostname.endswith("..")
            and len(normalized_hostname) <= 253
            and normalized_hostname.isascii()
            and all(
                _DNS_LABEL.fullmatch(label) is not None
                for label in normalized_hostname.split(".")
            )
        )


def _is_loopback_host(hostname: str) -> bool:
    """Return whether a hostname is explicitly local-only."""
    normalized_hostname = hostname.rstrip(".").casefold()
    if normalized_hostname == "localhost" or normalized_hostname.endswith(".localhost"):
        return True

    try:
        return ipaddress.ip_address(normalized_hostname).is_loopback
    except ValueError:
        return False


def _read_required_absolute_file(
    path: str,
    error_message: str,
    *,
    maximum_bytes: int,
) -> bytes:
    """Read one nonempty absolute file without disclosing its path."""
    required_path = Path(path)
    try:
        if not required_path.is_absolute() or not required_path.is_file():
            raise OSError
        with required_path.open("rb") as required_file:
            contents = required_file.read(maximum_bytes + 1)
        if not contents or len(contents) > maximum_bytes:
            raise OSError
    except OSError:
        raise SettingsConfigurationError(error_message) from None
    return contents


def _validate_otlp_certificate(certificate: str) -> None:
    """Require one readable absolute CA file without echoing its path."""
    _read_required_absolute_file(
        certificate,
        "The OTLP trace certificate must be a readable absolute file",
        maximum_bytes=_MAX_OTLP_CA_BYTES,
    )


def _load_gateway_bearer_token(token_file: str) -> SecretStr:
    """Load one strict bearer token from a Compose-mounted secret file."""
    raw_token = _read_required_absolute_file(
        token_file,
        "The trace gateway token must be a readable absolute file",
        maximum_bytes=_MAX_GATEWAY_TOKEN_BYTES,
    )
    try:
        token = raw_token.decode("ascii")
    except UnicodeDecodeError:
        token = ""
    if token.endswith("\r\n"):
        token = token[:-2]
    elif token.endswith("\n"):
        token = token[:-1]
    if _GATEWAY_BEARER_TOKEN.fullmatch(token) is None:
        msg = "The trace gateway token file contains an invalid bearer token"
        raise SettingsConfigurationError(msg)
    return SecretStr(token)


def _validate_otlp_headers(headers: str) -> None:
    """Validate OTLP header syntax without retaining or disclosing credentials."""
    seen_names: set[str] = set()
    for raw_header in headers.split(","):
        if not raw_header or "=" not in raw_header:
            msg = "OTLP trace headers must use comma-separated name=value pairs"
            raise SettingsConfigurationError(msg)

        raw_name, raw_value = raw_header.split("=", maxsplit=1)
        if (
            _PERCENT_ESCAPE.search(raw_name)
            or _PERCENT_ESCAPE.search(raw_value)
            or _OTLP_HEADER_RAW_VALUE.fullmatch(raw_value) is None
        ):
            msg = "OTLP trace headers contain invalid syntax or percent encoding"
            raise SettingsConfigurationError(msg)

        name = unquote(raw_name)
        value = unquote(raw_value)
        normalized_name = name.casefold()
        if (
            not name
            or _HTTP_HEADER_NAME.fullmatch(name) is None
            or _HTTP_HEADER_VALUE.fullmatch(value) is None
        ):
            msg = "OTLP trace headers contain an invalid name or value"
            raise SettingsConfigurationError(msg)
        if normalized_name in seen_names:
            msg = "OTLP trace headers must not contain duplicate names"
            raise SettingsConfigurationError(msg)
        seen_names.add(normalized_name)


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


class DatabaseReadinessEnv(RedactedBaseSettings):
    """Process-only settings for the container database readiness gate."""

    database_url: SecretStr | None = Field(
        default=None,
        alias="DATABASE_URL",
        description="PostgreSQL URL checked before the application starts",
    )
    db_ready_timeout: float = Field(
        default=60.0,
        alias="DB_READY_TIMEOUT",
        gt=0,
        le=_DATABASE_READINESS_MAX_TIMEOUT,
        description="Maximum total seconds to wait for database readiness",
    )
    retry_interval: float = Field(
        default=1.0,
        alias="DB_READY_RETRY_INTERVAL",
        gt=0,
        le=_DATABASE_READINESS_MAX_INTERVAL,
        description="Seconds between transient database readiness failures",
    )
    attempt_timeout: float = Field(
        default=5.0,
        alias="DB_READY_ATTEMPT_TIMEOUT",
        gt=0,
        le=_DATABASE_READINESS_MAX_INTERVAL,
        description="Maximum seconds for one database readiness attempt",
    )

    model_config = SettingsConfigDict(
        env_file=None,
        populate_by_name=True,
        extra="ignore",
        hide_input_in_errors=True,
    )

    @field_validator("database_url", mode="before")
    @classmethod
    def _normalize_blank_database_url(cls, value: Any) -> Any:
        """Treat a blank process value as an intentionally unconfigured database."""
        return _blank_optional_value(value)

    @model_validator(mode="after")
    def _reject_example_database_url(self) -> Self:
        """Reject the documented database placeholder before any connection attempt."""
        _check_example_secret("DATABASE_URL", self.database_url)
        return self


class ObservabilityEnv(RedactedBaseSettings):
    """Settings needed before ADK constructs the FastAPI application."""

    _gateway_bearer_token: SecretStr | None = PrivateAttr(default=None)

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
    langfuse_base_url: str | None = Field(
        default=None,
        alias="LANGFUSE_BASE_URL",
        description="Optional Langfuse API base URL override",
    )
    otel_exporter_otlp_traces_endpoint: str | None = Field(
        default=None,
        alias="OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        description="Explicit OTLP/HTTP trace endpoint",
    )
    otel_exporter_otlp_traces_protocol: str | None = Field(
        default=None,
        alias="OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
        description="Explicit OTLP trace protocol (http/protobuf only)",
    )
    otel_exporter_otlp_traces_headers: SecretStr | None = Field(
        default=None,
        alias="OTEL_EXPORTER_OTLP_TRACES_HEADERS",
        description="Explicit OTLP trace authentication headers",
    )
    otel_exporter_otlp_traces_certificate: str | None = Field(
        default=None,
        alias="OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE",
        description="Absolute CA certificate path for an explicit HTTPS collector",
    )
    otel_gateway_bearer_token_file: str | None = Field(
        default=None,
        alias="OTEL_GATEWAY_BEARER_TOKEN_FILE",
        description="Absolute bearer-token secret file for the local trace gateway",
    )
    otel_exporter_otlp_traces_timeout: float | None = Field(
        default=None,
        alias="OTEL_EXPORTER_OTLP_TRACES_TIMEOUT",
        gt=0,
        le=_DEFAULT_OTLP_TRACE_TIMEOUT_SECONDS,
        description="Bounded OTLP/HTTP request timeout in seconds",
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
        "otel_exporter_otlp_traces_endpoint",
        "otel_exporter_otlp_traces_protocol",
        "otel_exporter_otlp_traces_headers",
        "otel_exporter_otlp_traces_certificate",
        "otel_gateway_bearer_token_file",
        "otel_exporter_otlp_traces_timeout",
        mode="before",
    )
    @classmethod
    def _normalize_blank_optional_fields(cls, value: Any) -> Any:
        """Treat source-selected blank optional exporter values as unconfigured."""
        return _blank_optional_value(value)

    @field_validator("langfuse_base_url", mode="before")
    @classmethod
    def _normalize_blank_langfuse_base_url(cls, value: Any) -> Any:
        """Treat a selected blank Langfuse base URL as unconfigured."""
        return _blank_optional_value(value)

    def __init__(self, **values: Any) -> None:
        """Reject non-trace OTLP variables from every supported settings source."""
        if any(str(key).startswith("_") for key in values):
            msg = (
                "Per-instance observability settings source overrides are not supported"
            )
            raise SettingsConfigurationError(msg)
        _reject_non_trace_otlp_settings(values)
        _reject_non_trace_otlp_settings(os.environ)
        dotenv_values = DotEnvSettingsSource(type(self))()
        _reject_non_trace_otlp_settings(dotenv_values)
        super().__init__(**values)

    @model_validator(mode="after")
    def _validate_observability_mode(self) -> Self:
        """Validate one complete, safe, HTTP/protobuf trace export mode."""
        public_key_configured = self.langfuse_public_key is not None
        secret_key_configured = self.langfuse_secret_key is not None
        langfuse_setting_configured = (
            public_key_configured
            or secret_key_configured
            or self.langfuse_base_url is not None
        )

        explicit_endpoint = self.otel_exporter_otlp_traces_endpoint
        explicit_protocol = self.otel_exporter_otlp_traces_protocol
        explicit_headers = self.otel_exporter_otlp_traces_headers
        explicit_certificate = self.otel_exporter_otlp_traces_certificate
        gateway_token_file = self.otel_gateway_bearer_token_file
        explicit_timeout = self.otel_exporter_otlp_traces_timeout
        explicit_mode = any(
            value is not None
            for value in (
                explicit_endpoint,
                explicit_protocol,
                explicit_headers,
                explicit_certificate,
                gateway_token_file,
            )
        )
        if explicit_mode and langfuse_setting_configured:
            msg = "Langfuse and explicit OTLP trace settings are mutually exclusive"
            raise SettingsConfigurationError(msg)
        if langfuse_setting_configured and not (
            public_key_configured and secret_key_configured
        ):
            msg = "Langfuse export requires both public and secret keys"
            raise SettingsConfigurationError(msg)

        if explicit_endpoint is None and (
            explicit_protocol is not None
            or explicit_headers is not None
            or explicit_certificate is not None
            or gateway_token_file is not None
        ):
            msg = (
                "OTLP trace protocol, headers, certificate, and gateway token "
                "require a trace endpoint"
            )
            raise SettingsConfigurationError(msg)
        if explicit_headers is not None and gateway_token_file is not None:
            msg = (
                "Explicit OTLP headers and a trace gateway token are mutually exclusive"
            )
            raise SettingsConfigurationError(msg)
        if gateway_token_file is not None and explicit_certificate is None:
            msg = "A trace gateway token requires a custom CA certificate"
            raise SettingsConfigurationError(msg)
        if explicit_timeout is not None and not (
            explicit_endpoint is not None
            or (public_key_configured and secret_key_configured)
        ):
            msg = "OTLP trace timeout requires a complete remote export mode"
            raise SettingsConfigurationError(msg)

        if (
            explicit_protocol is not None
            and explicit_protocol.casefold() != "http/protobuf"
        ):
            msg = "Only the OTLP http/protobuf trace protocol is supported"
            raise SettingsConfigurationError(msg)
        if explicit_protocol is not None:
            self.otel_exporter_otlp_traces_protocol = explicit_protocol.casefold()

        if explicit_endpoint is not None:
            parsed_endpoint = _parse_safe_http_endpoint(explicit_endpoint)
            if (
                not parsed_endpoint.path.endswith("/v1/traces")
                or parsed_endpoint.path.count("/v1/traces") != 1
            ):
                msg = "The OTLP trace endpoint must end exactly once in /v1/traces"
                raise SettingsConfigurationError(msg)
            if (
                explicit_certificate is not None
                and parsed_endpoint.scheme.casefold() != "https"
            ):
                msg = "An OTLP trace certificate requires an HTTPS endpoint"
                raise SettingsConfigurationError(msg)
        if explicit_headers is not None:
            _validate_otlp_headers(explicit_headers.get_secret_value())
        if explicit_certificate is not None:
            _validate_otlp_certificate(explicit_certificate)
        if gateway_token_file is not None:
            self._gateway_bearer_token = _load_gateway_bearer_token(gateway_token_file)
        if public_key_configured and secret_key_configured:
            _parse_safe_http_endpoint(self.effective_langfuse_base_url)

        return self

    @property
    def effective_langfuse_base_url(self) -> str:
        """Return the configured Langfuse base or the hosted default."""
        return self.langfuse_base_url or _DEFAULT_LANGFUSE_BASE_URL

    @property
    def effective_otel_exporter_otlp_traces_timeout(self) -> float:
        """Return the bounded HTTP request timeout used for remote export."""
        return (
            self.otel_exporter_otlp_traces_timeout
            or _DEFAULT_OTLP_TRACE_TIMEOUT_SECONDS
        )

    @property
    def gateway_bearer_token(self) -> SecretStr | None:
        """Return the validated local gateway token without exposing it."""
        return self._gateway_bearer_token


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
        db_readiness_probe_timeout: Maximum duration of an HTTP readiness query.
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

    db_readiness_probe_timeout: float = Field(
        default=2.0,
        alias="DB_READINESS_PROBE_TIMEOUT",
        gt=0,
        le=_DATABASE_READINESS_MAX_PROBE_TIMEOUT,
        description="Maximum seconds for one HTTP database readiness probe",
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
            print(f"DB_READINESS_PROBE_TIMEOUT: {self.db_readiness_probe_timeout}")
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
            return _normalize_sqlalchemy_database_url(database_url)
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
