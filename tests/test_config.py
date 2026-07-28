"""Comprehensive unit tests for config module."""

import json
import os
import traceback
import warnings
from pathlib import Path
from typing import Any, cast

import pytest
from opentelemetry.util.re import parse_env_headers
from pydantic import SecretStr, TypeAdapter, ValidationError
from pydantic_settings import BaseSettings

from agent.utils.config import (
    AgentRuntimeEnv,
    ObservabilityEnv,
    ServerEnv,
    SettingsConfigurationError,
    initialize_environment,
)


@pytest.fixture(autouse=True)
def isolate_settings_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Prevent settings tests from reading the developer's repository dotenv."""
    monkeypatch.chdir(tmp_path)


class TestServerEnv:
    """Tests for ServerEnv model."""

    def test_valid_server_env_creation(self, valid_server_env: dict[str, str]) -> None:
        """Test creating ServerEnv with valid required fields."""
        env = ServerEnv.model_validate(valid_server_env)

        assert env.agent_name == "test-agent"
        assert isinstance(env, BaseSettings)

    def test_server_env_missing_required_field_raises_validation_error(self) -> None:
        """Test that missing required fields raise ValidationError."""
        data: dict[str, str] = {
            # Missing AGENT_NAME (the only required field)
        }

        with pytest.raises(ValidationError) as exc_info:
            ServerEnv.model_validate(data)

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("AGENT_NAME",) for error in errors)

    def test_server_env_optional_fields_use_defaults(
        self, valid_server_env: dict[str, str]
    ) -> None:
        """Test that optional fields use default values when not provided."""
        env = ServerEnv.model_validate(valid_server_env)

        # Check defaults
        assert env.log_level == "INFO"
        assert env.serve_web_interface is False
        assert env.reload_agents is False
        assert env.agent_engine is None
        assert env.database_url is None
        assert env.db_readiness_probe_timeout == 2
        assert env.openrouter_api_key is None
        assert env.agent_dir is None
        assert env.allow_origins == '["http://127.0.0.1", "http://127.0.0.1:8080"]'
        assert env.host == "127.0.0.1"
        assert env.port == 8080

    def test_server_env_optional_fields_with_values(
        self, valid_server_env: dict[str, str]
    ) -> None:
        """Test setting optional fields with actual values."""
        data = {
            **valid_server_env,
            "AGENT_NAME": "custom-agent",
            "LOG_LEVEL": "DEBUG",
            "SERVE_WEB_INTERFACE": "true",
            "RELOAD_AGENTS": "true",
            "AGENT_ENGINE": "test-engine-id",
            "DATABASE_URL": "postgresql://user:pass@localhost/db",
            "DB_READINESS_PROBE_TIMEOUT": "1.5",
            "OPENROUTER_API_KEY": "sk-or-v1-test",
            "AGENT_DIR": "/srv/agents",
            "ALLOW_ORIGINS": '["http://localhost:3000"]',
            "HOST": "0.0.0.0",  # noqa: S104
            "PORT": "9000",
        }

        env = ServerEnv.model_validate(data)

        assert env.agent_name == "custom-agent"
        assert env.log_level == "DEBUG"
        assert env.serve_web_interface is True
        assert env.reload_agents is True
        assert env.agent_engine == "test-engine-id"
        assert isinstance(env.database_url, SecretStr)
        assert (
            env.database_url.get_secret_value() == "postgresql://user:pass@localhost/db"
        )
        assert env.db_readiness_probe_timeout == 1.5
        assert isinstance(env.openrouter_api_key, SecretStr)
        assert env.openrouter_api_key.get_secret_value() == "sk-or-v1-test"
        assert env.agent_dir == "/srv/agents"
        assert env.allow_origins == '["http://localhost:3000"]'
        assert env.host == "0.0.0.0"  # noqa: S104
        assert env.port == 9000

    @pytest.mark.parametrize(
        "probe_timeout",
        [0, -1, 2.1, 3, float("inf"), float("nan")],
    )
    def test_server_env_rejects_readiness_probe_outside_http_budget(
        self,
        valid_server_env: dict[str, str],
        probe_timeout: float,
    ) -> None:
        """Keep the database attempt strictly inside the HTTP client timeout."""
        data = {
            **valid_server_env,
            "DB_READINESS_PROBE_TIMEOUT": probe_timeout,
        }

        with pytest.raises(ValidationError):
            ServerEnv.model_validate(data)

    def test_agent_engine_uri_property(self, valid_server_env: dict[str, str]) -> None:
        """Test that agent_engine_uri property is computed correctly."""
        # Without agent_engine
        env = ServerEnv.model_validate(valid_server_env)
        assert env.agent_engine_uri is None

        # With agent_engine
        data = {**valid_server_env, "AGENT_ENGINE": "test-engine-id"}
        env = ServerEnv.model_validate(data)
        assert env.agent_engine_uri == "agentengine://test-engine-id"

    def test_session_uri_property(self, valid_server_env: dict[str, str]) -> None:
        """Test that session_uri property is computed correctly."""
        # Case 1: Neither database_url nor agent_engine
        env = ServerEnv.model_validate(valid_server_env)
        assert env.session_uri is None

        # Case 2: Only agent_engine
        data = {**valid_server_env, "AGENT_ENGINE": "test-engine-id"}
        env = ServerEnv.model_validate(data)
        assert env.session_uri == "agentengine://test-engine-id"

        # Case 3: Only database_url (takes precedence)
        db_url = "postgresql://user:pass@localhost/db?sslmode=require&channel_binding=require"
        data = {**valid_server_env, "DATABASE_URL": db_url}
        env = ServerEnv.model_validate(data)
        # Should replace sslmode=require with ssl=require and remove
        # channel_binding=require
        # Note: it replaces &channel_binding=require with empty string
        expected = "postgresql://user:pass@localhost/db?ssl=require"
        assert env.session_uri == expected

        # Case 4: Both database_url and agent_engine (database_url takes precedence)
        data = {
            **valid_server_env,
            "DATABASE_URL": "postgresql://user:pass@localhost/db",
            "AGENT_ENGINE": "test-engine-id",
        }
        env = ServerEnv.model_validate(data)
        assert env.session_uri == "postgresql://user:pass@localhost/db"

    def test_allow_origins_list_property(
        self, valid_server_env: dict[str, str]
    ) -> None:
        """Test that allow_origins_list property parses JSON correctly."""
        data = {
            **valid_server_env,
            "ALLOW_ORIGINS": '["http://localhost:3000", "http://localhost:8080"]',
        }
        env = ServerEnv.model_validate(data)

        origins = env.allow_origins_list
        assert origins == ["http://localhost:3000", "http://localhost:8080"]

    def test_allow_origins_list_invalid_json_raises_error(
        self, valid_server_env: dict[str, str]
    ) -> None:
        """Test that invalid JSON in allow_origins raises ValueError."""
        data = {**valid_server_env, "ALLOW_ORIGINS": "not valid json"}
        env = ServerEnv.model_validate(data)

        with pytest.raises(ValueError, match="Failed to parse ALLOW_ORIGINS"):
            _ = env.allow_origins_list

    def test_allow_origins_list_not_array_raises_error(
        self, valid_server_env: dict[str, str]
    ) -> None:
        """Test that non-array JSON in allow_origins raises ValueError."""
        data = {**valid_server_env, "ALLOW_ORIGINS": '{"key": "value"}'}
        env = ServerEnv.model_validate(data)

        with pytest.raises(ValueError, match="must be a JSON array of strings"):
            _ = env.allow_origins_list

    def test_allow_origins_list_non_string_array_raises_error(
        self, valid_server_env: dict[str, str]
    ) -> None:
        """Test that array with non-strings raises ValueError."""
        data = {**valid_server_env, "ALLOW_ORIGINS": "[123, 456]"}
        env = ServerEnv.model_validate(data)

        with pytest.raises(ValueError, match="must be a JSON array of strings"):
            _ = env.allow_origins_list

    def test_server_env_print_config(
        self, valid_server_env: dict[str, str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test that print_config outputs expected information."""
        env = ServerEnv.model_validate(valid_server_env)
        env.print_config()

        captured = capsys.readouterr()
        output = captured.out

        # Check key information is printed
        assert "test-agent" in output
        assert "AGENT_NAME" in output
        assert "LOG_LEVEL" in output

    def test_server_env_print_config_with_db(
        self, valid_server_env: dict[str, str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test print_config masks secrets and outputs DB pool settings."""
        database_url = "postgresql://secret-user:secret-pass@localhost/private"
        openrouter_api_key = "sk-or-v1-secret-canary"
        data = {
            **valid_server_env,
            "DATABASE_URL": database_url,
            "OPENROUTER_API_KEY": openrouter_api_key,
        }
        env = ServerEnv.model_validate(data)
        env.print_config()

        captured = capsys.readouterr()
        output = captured.out

        assert "DB_POOL_PRE_PING" in output
        assert "DB_READINESS_PROBE_TIMEOUT" in output
        assert "DB_POOL_RECYCLE" in output
        assert "DB_POOL_SIZE" in output
        assert "DB_MAX_OVERFLOW" in output
        assert "DB_POOL_TIMEOUT" in output
        assert "DATABASE_URL" in output
        assert "OPENROUTER_API_KEY" in output
        assert "**********" in output
        assert database_url not in output
        assert openrouter_api_key not in output

    def test_server_env_ignores_extra_fields(
        self, valid_server_env: dict[str, str]
    ) -> None:
        """Test that extra environment variables are ignored."""
        data = {**valid_server_env, "EXTRA_VAR": "extra-value", "PATH": "/usr/bin"}

        env = ServerEnv.model_validate(data)
        assert env.agent_name == "test-agent"
        # Extra fields should not be included
        assert not hasattr(env, "EXTRA_VAR")
        assert not hasattr(env, "PATH")

    def test_secret_fields_are_redacted_from_model_output(
        self, valid_server_env: dict[str, str]
    ) -> None:
        """Test model representations and JSON never disclose raw secrets."""
        database_url = "postgresql://secret-user:secret-pass@localhost/private"
        openrouter_api_key = "sk-or-v1-secret-canary"
        env = ServerEnv.model_validate(
            {
                **valid_server_env,
                "DATABASE_URL": database_url,
                "OPENROUTER_API_KEY": openrouter_api_key,
            }
        )

        rendered_outputs = (
            repr(env),
            str(env.model_dump()),
            env.model_dump_json(),
        )

        for rendered_output in rendered_outputs:
            assert database_url not in rendered_output
            assert openrouter_api_key not in rendered_output
            assert "**********" in rendered_output

    @pytest.mark.parametrize(
        ("field_name", "placeholder"),
        [
            ("DATABASE_URL", "changethis"),
            (
                "DATABASE_URL",
                " POSTGRESQL://USER:PASS@HOST:PORT/DBNAME?SSL=REQUIRE ",
            ),
            ("OPENROUTER_API_KEY", "changethis"),
            ("OPENROUTER_API_KEY", " YOUR-OPENROUTER-KEY-HERE "),
        ],
    )
    def test_example_secrets_are_rejected_without_disclosure(
        self,
        valid_server_env: dict[str, str],
        field_name: str,
        placeholder: str,
    ) -> None:
        """Test known placeholders fail validation without echoing their values."""
        with pytest.raises(SettingsConfigurationError) as exc_info:
            ServerEnv.model_validate(
                {
                    **valid_server_env,
                    field_name: placeholder,
                }
            )

        error_output = str(exc_info.value)
        assert field_name in error_output
        assert placeholder.strip() not in error_output

    @pytest.mark.parametrize(
        "construction_method",
        [
            "constructor",
            "model_validate",
            "model_validate_json",
            "model_validate_strings",
            "type_adapter_validate_strings",
        ],
    )
    def test_placeholder_errors_never_embed_secret_inputs(
        self,
        valid_server_env: dict[str, str],
        construction_method: str,
    ) -> None:
        """Test all public validation paths propagate a safe placeholder failure."""
        rejected_secret = "changethis"  # noqa: S105
        unrelated_secret = "sk-or-v1-unrelated-secret-canary"  # noqa: S105
        data = {
            **valid_server_env,
            "DATABASE_URL": rejected_secret,
            "OPENROUTER_API_KEY": unrelated_secret,
        }

        with pytest.raises(SettingsConfigurationError) as exc_info:
            if construction_method == "constructor":
                ServerEnv(**cast(dict[str, Any], data))
            elif construction_method == "model_validate":
                ServerEnv.model_validate(data)
            elif construction_method == "model_validate_json":
                ServerEnv.model_validate_json(json.dumps(data))
            elif construction_method == "model_validate_strings":
                ServerEnv.model_validate_strings(data)
            else:
                TypeAdapter(ServerEnv).validate_strings(data)

        error = exc_info.value
        rendered_outputs = (
            str(error),
            repr(error),
            "".join(traceback.format_exception(error)),
        )
        for rendered_output in rendered_outputs:
            assert rejected_secret not in rendered_output
            assert unrelated_secret not in rendered_output

        assert "DATABASE_URL must not use an example or default secret" in str(error)
        assert error.__cause__ is None
        assert error.__context__ is None

    @pytest.mark.parametrize(
        "construction_method",
        [
            "constructor",
            "model_validate",
            "model_validate_json",
            "model_validate_strings",
            "type_adapter_validate_strings",
        ],
    )
    def test_validation_errors_never_disclose_unrelated_secrets(
        self,
        valid_server_env: dict[str, str],
        construction_method: str,
    ) -> None:
        """Test ordinary diagnostics never include unrelated secret fields."""
        database_url = "postgresql://secret-user:secret-pass@localhost/private"
        openrouter_key = "sk-or-v1-unrelated-secret-canary"
        data = {
            **valid_server_env,
            "DATABASE_URL": database_url,
            "OPENROUTER_API_KEY": openrouter_key,
            "PORT": "not-an-integer",
        }

        with pytest.raises(ValidationError) as exc_info:
            if construction_method == "constructor":
                ServerEnv(**cast(dict[str, Any], data))
            elif construction_method == "model_validate":
                ServerEnv.model_validate(data)
            elif construction_method == "model_validate_json":
                ServerEnv.model_validate_json(json.dumps(data))
            elif construction_method == "model_validate_strings":
                ServerEnv.model_validate_strings(data)
            else:
                TypeAdapter(ServerEnv).validate_strings(data)

        error = exc_info.value
        rendered_outputs = (
            str(error),
            repr(error),
            repr(error.errors()),
            repr(error.errors(include_input=True)),
            error.json(),
            "".join(traceback.format_exception(error)),
        )
        for rendered_output in rendered_outputs:
            assert database_url not in rendered_output
            assert openrouter_key not in rendered_output

        errors = error.errors(include_input=True)
        assert len(errors) == 1
        assert errors[0]["loc"] == ("PORT",)
        assert errors[0]["type"] == "int_parsing"

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("DATABASE_URL", "postgresql://user:pass@localhost/db"),
            ("OPENROUTER_API_KEY", "changethis-but-not-a-placeholder"),
        ],
    )
    def test_non_placeholder_secret_values_are_accepted(
        self,
        valid_server_env: dict[str, str],
        field_name: str,
        value: str,
    ) -> None:
        """Test legitimate values near placeholder patterns remain valid."""
        env = ServerEnv.model_validate(
            {
                **valid_server_env,
                field_name: value,
            }
        )

        secret_value = getattr(env, field_name.lower())
        assert isinstance(secret_value, SecretStr)
        assert secret_value.get_secret_value() == value


class TestAgentRuntimeEnv:
    """Tests for process-only agent construction settings."""

    def test_defaults_do_not_require_provider_keys(self) -> None:
        """Test native model defaults remain available without provider keys."""
        env = AgentRuntimeEnv()

        assert env.root_agent_model == "gemini-2.5-flash"
        assert env.openrouter_api_key is None
        assert env.google_api_key is None

    def test_process_values_are_typed_and_constructor_values_win(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test process inputs parse as secrets and constructor values rank first."""
        monkeypatch.setenv("ROOT_AGENT_MODEL", "process/model")
        monkeypatch.setenv("OPENROUTER_API_KEY", "process-openrouter-key")
        monkeypatch.setenv("GOOGLE_API_KEY", "process-google-key")

        env = AgentRuntimeEnv(ROOT_AGENT_MODEL="constructor/model")

        assert env.root_agent_model == "constructor/model"
        assert env.openrouter_api_key is not None
        assert env.openrouter_api_key.get_secret_value() == "process-openrouter-key"
        assert env.google_api_key is not None
        assert env.google_api_key.get_secret_value() == "process-google-key"

    def test_dotenv_is_ignored_without_mutating_process_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Test direct agent settings never consume or inject cwd dotenv values."""
        (tmp_path / ".env").write_text(
            "\n".join(
                [
                    "ROOT_AGENT_MODEL=dotenv/model",
                    "OPENROUTER_API_KEY=dotenv-secret-canary",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        original_environment = dict(os.environ)

        env = AgentRuntimeEnv()

        assert env.root_agent_model == "gemini-2.5-flash"
        assert env.openrouter_api_key is None
        assert dict(os.environ) == original_environment

    def test_empty_provider_keys_are_normalized_after_source_selection(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test deployment-injected empty strings remain false provider values."""
        monkeypatch.setenv("OPENROUTER_API_KEY", " ")
        monkeypatch.setenv("GOOGLE_API_KEY", "")

        env = AgentRuntimeEnv()

        assert env.openrouter_api_key is None
        assert env.google_api_key is None

    def test_openrouter_placeholder_is_rejected_without_disclosure(self) -> None:
        """Test runtime settings reject the documented provider placeholder."""
        placeholder = "your-openrouter-key-here"

        with pytest.raises(SettingsConfigurationError) as exc_info:
            AgentRuntimeEnv.model_validate({"OPENROUTER_API_KEY": placeholder})

        error = exc_info.value
        assert "OPENROUTER_API_KEY" in str(error)
        assert placeholder not in str(error)
        assert placeholder not in repr(error)
        assert placeholder not in "".join(traceback.format_exception(error))


class TestObservabilityEnv:
    """Tests for settings required before ADK agent loading."""

    def test_defaults(self) -> None:
        """Keep remote export and content capture disabled by default."""
        env = ObservabilityEnv()

        assert env.telemetry_namespace == "local"
        assert env.service_revision == "local"
        assert env.langfuse_public_key is None
        assert env.langfuse_secret_key is None
        assert env.langfuse_base_url is None
        assert env.effective_langfuse_base_url == "https://cloud.langfuse.com"
        assert env.otel_exporter_otlp_traces_endpoint is None
        assert env.otel_exporter_otlp_traces_protocol is None
        assert env.otel_exporter_otlp_traces_headers is None
        assert env.otel_exporter_otlp_traces_certificate is None
        assert env.otel_gateway_bearer_token_file is None
        assert env.gateway_bearer_token is None
        assert env.otel_exporter_otlp_traces_timeout is None
        assert env.effective_otel_exporter_otlp_traces_timeout == 2.0
        assert env.otel_capture_message_content is False

    def test_dotenv_langfuse_values_are_typed_without_environment_mutation(
        self,
        tmp_path: Path,
    ) -> None:
        """Read one complete Langfuse mode without mutating process settings."""
        public_key = "pk-lf-dotenv-canary"
        secret_key = "sk-lf-dotenv-canary"  # noqa: S105
        (tmp_path / ".env").write_text(
            "\n".join(
                [
                    "TELEMETRY_NAMESPACE=dotenv-namespace",
                    "K_REVISION=dotenv-revision",
                    f"LANGFUSE_PUBLIC_KEY={public_key}",
                    f"LANGFUSE_SECRET_KEY={secret_key}",
                    "LANGFUSE_BASE_URL=https://langfuse.example.test/",
                    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        original_environment = dict(os.environ)

        env = ObservabilityEnv()

        assert env.telemetry_namespace == "dotenv-namespace"
        assert env.service_revision == "dotenv-revision"
        assert env.langfuse_public_key is not None
        assert env.langfuse_public_key.get_secret_value() == public_key
        assert env.langfuse_secret_key is not None
        assert env.langfuse_secret_key.get_secret_value() == secret_key
        assert env.langfuse_base_url == "https://langfuse.example.test/"
        assert env.effective_langfuse_base_url == "https://langfuse.example.test/"
        assert env.otel_capture_message_content is True
        assert dict(os.environ) == original_environment

    def test_process_and_constructor_values_override_dotenv(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Test the complete source priority for observability settings."""
        (tmp_path / ".env").write_text(
            "TELEMETRY_NAMESPACE=dotenv\nK_REVISION=dotenv\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("TELEMETRY_NAMESPACE", "process")
        monkeypatch.setenv("K_REVISION", "process")

        env = ObservabilityEnv(TELEMETRY_NAMESPACE="constructor")

        assert env.telemetry_namespace == "constructor"
        assert env.service_revision == "process"

    def test_explicit_trace_values_are_normalized_and_redacted(
        self,
        tmp_path: Path,
    ) -> None:
        """Accept the one supported explicit HTTP/protobuf configuration."""
        header = "Authorization=Bearer%20header-secret-canary"
        certificate = tmp_path / "collector-ca.pem"
        certificate.write_text("synthetic-ca\n", encoding="utf-8")
        env = ObservabilityEnv.model_validate(
            {
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": (
                    "https://collector.example.test/otel/v1/traces"
                ),
                "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "HTTP/PROTOBUF",
                "OTEL_EXPORTER_OTLP_TRACES_HEADERS": header,
                "OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE": str(certificate),
                "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT": "1.5",
                "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true",
            }
        )

        assert env.otel_exporter_otlp_traces_endpoint == (
            "https://collector.example.test/otel/v1/traces"
        )
        assert env.otel_exporter_otlp_traces_protocol == "http/protobuf"
        assert env.otel_exporter_otlp_traces_headers is not None
        assert env.otel_exporter_otlp_traces_headers.get_secret_value() == header
        assert env.otel_exporter_otlp_traces_certificate == str(certificate)
        assert env.otel_exporter_otlp_traces_timeout == 1.5
        assert env.effective_otel_exporter_otlp_traces_timeout == 1.5
        assert header not in repr(env)
        assert env.otel_capture_message_content is True

    @pytest.mark.parametrize("line_ending", ["", "\n", "\r\n"])
    def test_gateway_token_file_is_loaded_and_redacted(
        self,
        line_ending: str,
        tmp_path: Path,
    ) -> None:
        """Load local receiver auth from a mounted secret without retaining a path."""
        certificate = tmp_path / "collector-ca.pem"
        certificate.write_text("synthetic-ca\n", encoding="utf-8")
        token = "gateway-token-secret-canary"  # noqa: S105
        token_file = tmp_path / "gateway-token"
        token_file.write_text(f"{token}{line_ending}", encoding="ascii")

        env = ObservabilityEnv.model_validate(
            {
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": (
                    "https://otel-collector:4318/v1/traces"
                ),
                "OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE": str(certificate),
                "OTEL_GATEWAY_BEARER_TOKEN_FILE": str(token_file),
            }
        )

        assert env.otel_gateway_bearer_token_file == str(token_file)
        assert env.gateway_bearer_token is not None
        assert env.gateway_bearer_token.get_secret_value() == token
        assert token not in repr(env)

    def test_blank_optional_values_remain_unconfigured(self) -> None:
        """Treat selected blank optional values as deliberate omissions."""
        env = ObservabilityEnv.model_validate(
            {
                "LANGFUSE_BASE_URL": " ",
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "",
                "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "\t",
                "OTEL_EXPORTER_OTLP_TRACES_HEADERS": " ",
                "OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE": "",
                "OTEL_GATEWAY_BEARER_TOKEN_FILE": "\t",
                "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT": "",
            }
        )

        assert env.langfuse_base_url is None
        assert env.otel_exporter_otlp_traces_endpoint is None
        assert env.otel_exporter_otlp_traces_protocol is None
        assert env.otel_exporter_otlp_traces_headers is None
        assert env.otel_exporter_otlp_traces_certificate is None
        assert env.otel_gateway_bearer_token_file is None
        assert env.gateway_bearer_token is None
        assert env.otel_exporter_otlp_traces_timeout is None

    @pytest.mark.parametrize(
        "settings",
        [
            {"LANGFUSE_PUBLIC_KEY": "pk-only-canary"},
            {"LANGFUSE_SECRET_KEY": "sk-only-canary"},
            {"LANGFUSE_BASE_URL": "https://langfuse-base-only.example"},
            {
                "LANGFUSE_PUBLIC_KEY": "pk-with-base-canary",
                "LANGFUSE_BASE_URL": "https://langfuse.example",
            },
        ],
    )
    def test_incomplete_langfuse_mode_is_rejected_without_disclosure(
        self,
        settings: dict[str, str],
    ) -> None:
        """Require atomic Langfuse credentials even when a base is supplied."""
        canaries = tuple(settings.values())

        with pytest.raises(SettingsConfigurationError) as exc_info:
            ObservabilityEnv.model_validate(settings)

        formatted_error = "".join(traceback.format_exception(exc_info.value))
        assert "requires both" in str(exc_info.value)
        assert all(canary not in formatted_error for canary in canaries)

    @pytest.mark.parametrize(
        "langfuse_settings",
        [
            {
                "LANGFUSE_PUBLIC_KEY": "pk-mixed-canary",
                "LANGFUSE_SECRET_KEY": "sk-mixed-canary",
            },
            {"LANGFUSE_BASE_URL": "https://mixed-base.example"},
        ],
    )
    def test_mixed_export_modes_are_rejected_without_disclosure(
        self,
        langfuse_settings: dict[str, str],
    ) -> None:
        """Never combine derived credentials with an explicit destination."""
        settings = {
            **langfuse_settings,
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": (
                "https://collector.example/v1/traces"
            ),
        }

        with pytest.raises(SettingsConfigurationError) as exc_info:
            ObservabilityEnv.model_validate(settings)

        formatted_error = "".join(traceback.format_exception(exc_info.value))
        assert "mutually exclusive" in str(exc_info.value)
        assert all(value not in formatted_error for value in settings.values())

    @pytest.mark.parametrize(
        ("option_name", "option_value"),
        [
            ("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf"),
            ("OTEL_EXPORTER_OTLP_TRACES_HEADERS", "x-test=value"),
            (
                "OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE",
                "/run/secrets/otel_gateway_ca",
            ),
            (
                "OTEL_GATEWAY_BEARER_TOKEN_FILE",
                "/run/secrets/otel_gateway_token",
            ),
        ],
    )
    def test_trace_options_without_endpoint_are_rejected(
        self,
        option_name: str,
        option_value: str,
    ) -> None:
        """Do not let partial explicit settings fall through to SDK defaults."""
        with pytest.raises(
            SettingsConfigurationError, match="require a trace endpoint"
        ):
            ObservabilityEnv.model_validate({option_name: option_value})

    def test_trace_timeout_without_remote_mode_is_rejected(self) -> None:
        """Do not materialize a timeout when no exporter can consume it."""
        with pytest.raises(
            SettingsConfigurationError,
            match="requires a complete remote export mode",
        ):
            ObservabilityEnv.model_validate({"OTEL_EXPORTER_OTLP_TRACES_TIMEOUT": "1"})

    @pytest.mark.parametrize(
        "timeout",
        [0, -1, 2.01, float("inf"), float("nan")],
    )
    def test_trace_timeout_must_fit_the_shutdown_budget(self, timeout: float) -> None:
        """Bound each HTTP attempt so one retry remains inside outer shutdown."""
        with pytest.raises(ValidationError):
            ObservabilityEnv.model_validate(
                {
                    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": (
                        "https://collector.example/v1/traces"
                    ),
                    "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT": timeout,
                }
            )

    @pytest.mark.parametrize("protocol", ["grpc", "http/json", "\thttp/protobuf"])
    def test_unsupported_protocol_is_rejected_without_echo(
        self,
        protocol: str,
    ) -> None:
        """Reject every transport except exact HTTP/protobuf."""
        with pytest.raises(SettingsConfigurationError) as exc_info:
            ObservabilityEnv.model_validate(
                {
                    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": (
                        "https://collector.example/v1/traces"
                    ),
                    "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": protocol,
                }
            )

        assert protocol not in str(exc_info.value)

    @pytest.mark.parametrize(
        "endpoint",
        [
            "collector.example/v1/traces",
            "ftp://collector.example/v1/traces",
            "https:///v1/traces",
            "https://user:password@collector.example/v1/traces",
            "https://collector.example/v1/traces;token=endpoint-canary",
            "https://collector.example/v1/traces?token=endpoint-canary",
            "https://collector.example/v1/traces#endpoint-canary",
            "https://collector.example:invalid/v1/traces",
            "https://collector.example:0/v1/traces",
            "http://collector.example/v1/traces",
            "https://collector.example/%0d%0a/v1/traces",
            "https://collector .example/v1/traces",
            "https://collector..example/v1/traces",
            "https://collector.example../v1/traces",
            "https://./v1/traces",
            "https://colléctor.example/v1/traces",
            (f"https://{'a' * 63}.{'b' * 63}.{'c' * 63}.{'d' * 63}/v1/traces"),
            "https://-collector.example/v1/traces",
            "https://collector-.example/v1/traces",
            "https://collector.example/%ZZ/v1/traces",
            "https://collector.example/%5C/v1/traces",
            " https://collector.example/v1/traces",
            "https://collector.example\\evil/v1/traces",
        ],
    )
    def test_unsafe_explicit_endpoint_is_rejected_without_disclosure(
        self,
        endpoint: str,
    ) -> None:
        """Reject ambiguous, credential-bearing, or remote plaintext URLs."""
        with pytest.raises(SettingsConfigurationError) as exc_info:
            ObservabilityEnv.model_validate(
                {"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": endpoint}
            )

        formatted_error = "".join(traceback.format_exception(exc_info.value))
        assert endpoint not in formatted_error
        assert "password" not in formatted_error
        assert "endpoint-canary" not in formatted_error

    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://collector.example",
            "https://collector.example/v1/traces/",
            "https://collector.example/v1/traces/v1/traces",
        ],
    )
    def test_explicit_endpoint_requires_one_exact_trace_suffix(
        self,
        endpoint: str,
    ) -> None:
        """Prevent missing or doubled HTTP signal paths."""
        with pytest.raises(SettingsConfigurationError, match="end exactly once"):
            ObservabilityEnv.model_validate(
                {"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": endpoint}
            )

    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://collector.example/v1/traces",
            "http://localhost:4318/v1/traces",
            "http://worker.localhost:4318/v1/traces",
            "http://127.0.0.1:4318/v1/traces",
            "http://[::1]:4318/v1/traces",
        ],
    )
    def test_safe_trace_endpoints_are_accepted(self, endpoint: str) -> None:
        """Allow TLS collectors and explicit loopback-only plaintext."""
        env = ObservabilityEnv.model_validate(
            {"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": endpoint}
        )

        assert env.otel_exporter_otlp_traces_endpoint == endpoint

    def test_trace_certificate_requires_https(self, tmp_path: Path) -> None:
        """Never apply custom CA trust to a plaintext collector."""
        certificate = tmp_path / "collector-ca.pem"
        certificate.write_text("synthetic-ca\n", encoding="utf-8")

        with pytest.raises(
            SettingsConfigurationError,
            match="certificate requires an HTTPS endpoint",
        ):
            ObservabilityEnv.model_validate(
                {
                    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": (
                        "http://127.0.0.1:4318/v1/traces"
                    ),
                    "OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE": str(certificate),
                }
            )

    def test_gateway_token_requires_a_custom_ca(self, tmp_path: Path) -> None:
        """Bind local bearer authentication to the chosen private-TLS transport."""
        token_file = tmp_path / "gateway-token"
        token_file.write_text("gateway-token", encoding="ascii")

        with pytest.raises(
            SettingsConfigurationError,
            match="requires a custom CA certificate",
        ):
            ObservabilityEnv.model_validate(
                {
                    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": (
                        "https://otel-collector:4318/v1/traces"
                    ),
                    "OTEL_GATEWAY_BEARER_TOKEN_FILE": str(token_file),
                }
            )

    def test_gateway_token_and_explicit_headers_are_mutually_exclusive(
        self,
        tmp_path: Path,
    ) -> None:
        """Prevent two authentication sources from competing."""
        certificate = tmp_path / "collector-ca.pem"
        certificate.write_text("synthetic-ca\n", encoding="utf-8")
        token_file = tmp_path / "gateway-token"
        token_file.write_text("gateway-token", encoding="ascii")

        with pytest.raises(
            SettingsConfigurationError,
            match="headers and a trace gateway token are mutually exclusive",
        ):
            ObservabilityEnv.model_validate(
                {
                    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": (
                        "https://otel-collector:4318/v1/traces"
                    ),
                    "OTEL_EXPORTER_OTLP_TRACES_HEADERS": "x-test=value",
                    "OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE": str(certificate),
                    "OTEL_GATEWAY_BEARER_TOKEN_FILE": str(token_file),
                }
            )

    @pytest.mark.parametrize(
        "certificate_kind",
        ["relative", "missing", "directory", "empty", "oversized"],
    )
    def test_trace_certificate_must_be_a_readable_absolute_file(
        self,
        certificate_kind: str,
        tmp_path: Path,
    ) -> None:
        """Fail before binding without disclosing an unusable CA path."""
        if certificate_kind == "relative":
            certificate = Path("relative-collector-ca.pem")
            certificate.write_text("synthetic-ca\n", encoding="utf-8")
        else:
            certificate = tmp_path / f"{certificate_kind}-collector-ca.pem"
            if certificate_kind == "directory":
                certificate.mkdir()
            elif certificate_kind == "empty":
                certificate.touch()
            elif certificate_kind == "oversized":
                certificate.write_bytes(b"x" * ((1024 * 1024) + 1))

        with pytest.raises(SettingsConfigurationError) as exc_info:
            ObservabilityEnv.model_validate(
                {
                    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": (
                        "https://collector.example/v1/traces"
                    ),
                    "OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE": str(certificate),
                }
            )

        formatted_error = "".join(traceback.format_exception(exc_info.value))
        assert "readable absolute file" in str(exc_info.value)
        assert str(certificate) not in formatted_error

    @pytest.mark.parametrize(
        "credential_case",
        [
            "relative",
            "missing",
            "directory",
            "empty",
            "non-ascii",
            "whitespace",
            "multiple-lines",
            "oversized",
        ],
    )
    def test_gateway_token_file_must_contain_one_strict_bearer_token(
        self,
        credential_case: str,
        tmp_path: Path,
    ) -> None:
        """Reject unusable or header-ambiguous local credentials without disclosure."""
        certificate = tmp_path / "collector-ca.pem"
        certificate.write_text("synthetic-ca\n", encoding="utf-8")
        if credential_case == "relative":
            token_file = Path("relative-gateway-token")
            token_file.write_text("gateway-token", encoding="ascii")
        else:
            token_file = tmp_path / f"{credential_case}-gateway-token"
            if credential_case == "directory":
                token_file.mkdir()
            elif credential_case == "empty":
                token_file.touch()
            elif credential_case == "non-ascii":
                token_file.write_text("tökén", encoding="utf-8")
            elif credential_case == "whitespace":
                token_file.write_text("gateway token\n", encoding="ascii")
            elif credential_case == "multiple-lines":
                token_file.write_text(
                    "gateway-token\nsecond-token\n",
                    encoding="ascii",
                )
            elif credential_case == "oversized":
                token_file.write_bytes(b"x" * ((4 * 1024) + 1))

        with pytest.raises(SettingsConfigurationError) as exc_info:
            ObservabilityEnv.model_validate(
                {
                    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": (
                        "https://otel-collector:4318/v1/traces"
                    ),
                    "OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE": str(certificate),
                    "OTEL_GATEWAY_BEARER_TOKEN_FILE": str(token_file),
                }
            )

        formatted_error = "".join(traceback.format_exception(exc_info.value))
        assert "trace gateway token" in str(exc_info.value)
        assert str(token_file) not in formatted_error

    @pytest.mark.parametrize(
        "headers",
        [
            "",
            "missing-equals",
            "x-test=value,",
            "=empty-name",
            "invalid name=value",
            "x-test=bad%ZZvalue",
            "x-test=canary%0D%0AX-Injected%3Ayes",
            "x-test=✓",
            'x-test=abc"def',
            "x-test=abc;def",
            "x-test=abc\\def",
            "x-test=abc def",
            "Authorization=one,authorization=two",
        ],
    )
    def test_malformed_headers_are_rejected_without_disclosure(
        self,
        headers: str,
    ) -> None:
        """Reject syntax ambiguity, CRLF, and case-insensitive duplicates."""
        settings = {
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": (
                "https://collector.example/v1/traces"
            ),
            "OTEL_EXPORTER_OTLP_TRACES_HEADERS": headers,
        }

        if not headers:
            env = ObservabilityEnv.model_validate(settings)
            assert env.otel_exporter_otlp_traces_headers is None
            return

        with pytest.raises(SettingsConfigurationError) as exc_info:
            ObservabilityEnv.model_validate(settings)

        assert headers not in "".join(traceback.format_exception(exc_info.value))

    def test_valid_encoded_headers_are_accepted(self) -> None:
        """Accept visible header values and encoded separators."""
        headers = (
            "Authorization=Bearer%20credential,"
            "x-routing=a%2Cb%3Dc,x-empty=,x!token=value,x-path=/v1/traces"
        )
        env = ObservabilityEnv.model_validate(
            {
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": (
                    "https://collector.example/v1/traces"
                ),
                "OTEL_EXPORTER_OTLP_TRACES_HEADERS": headers,
            }
        )

        assert env.otel_exporter_otlp_traces_headers is not None
        assert env.otel_exporter_otlp_traces_headers.get_secret_value() == headers
        assert parse_env_headers(headers, liberal=True) == {
            "authorization": "Bearer credential",
            "x-routing": "a,b=c",
            "x-empty": "",
            "x!token": "value",
            "x-path": "/v1/traces",
        }

    @pytest.mark.parametrize(
        ("source", "unsupported_name"),
        [
            ("constructor", "OTEL_EXPORTER_OTLP_ENDPOINT"),
            ("process", "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"),
            ("dotenv", "OTEL_EXPORTER_OTLP_TRACES_COMPRESSION"),
            (
                "constructor",
                "OTEL_PYTHON_EXPORTER_OTLP_HTTP_CREDENTIAL_PROVIDER",
            ),
            (
                "process",
                "OTEL_PYTHON_EXPORTER_OTLP_HTTP_TRACES_CREDENTIAL_PROVIDER",
            ),
            (
                "dotenv",
                "OTEL_PYTHON_EXPORTER_OTLP_HTTP_CREDENTIAL_PROVIDER",
            ),
        ],
    )
    def test_unvalidated_otlp_variables_are_rejected_from_every_source(
        self,
        source: str,
        unsupported_name: str,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Allowlist only the five validated trace exporter variables."""
        canary = "unsupported-setting-canary"
        constructor_values: dict[str, str] = {}
        if source == "constructor":
            constructor_values[unsupported_name] = canary
        elif source == "process":
            monkeypatch.setenv(unsupported_name, canary)
        else:
            (tmp_path / ".env").write_text(
                f"{unsupported_name}={canary}\n",
                encoding="utf-8",
            )

        with pytest.raises(SettingsConfigurationError) as exc_info:
            ObservabilityEnv.model_validate(constructor_values)

        formatted_error = "".join(traceback.format_exception(exc_info.value))
        assert unsupported_name not in formatted_error
        assert canary not in formatted_error

    def test_blank_otlp_http_credential_providers_are_inert(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Allow the gateway overlay to clear hostile provider hooks."""
        monkeypatch.setenv(
            "OTEL_PYTHON_EXPORTER_OTLP_HTTP_CREDENTIAL_PROVIDER",
            "",
        )
        monkeypatch.setenv(
            "OTEL_PYTHON_EXPORTER_OTLP_HTTP_TRACES_CREDENTIAL_PROVIDER",
            " ",
        )

        env = ObservabilityEnv()

        assert env.otel_exporter_otlp_traces_endpoint is None

    @pytest.mark.parametrize("_source_override", ["_env_file", "_secrets_dir"])
    def test_per_instance_settings_source_overrides_are_rejected(
        self,
        _source_override: str,
        tmp_path: Path,
    ) -> None:
        """Keep the OTLP allowlist on the documented settings sources."""
        source_path = tmp_path / "alternate-source"
        source_path.write_text(
            "OTEL_EXPORTER_OTLP_ENDPOINT=https://unsafe.example\n",
            encoding="utf-8",
        )

        with pytest.raises(
            SettingsConfigurationError,
            match="settings source overrides are not supported",
        ) as exc_info:
            ObservabilityEnv(**{_source_override: source_path})

        assert str(source_path) not in str(exc_info.value)


class TestInitializeEnvironment:
    """Tests for initialize_environment factory function."""

    def test_initialize_environment_success(
        self,
        valid_server_env: dict[str, str],
        set_environment: Any,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Test successful environment initialization."""
        monkeypatch.chdir(tmp_path)
        set_environment(valid_server_env)

        env = initialize_environment(ServerEnv, print_config=False)

        assert env.agent_name == "test-agent"

    def test_initialize_environment_validation_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Test that validation failure causes sys.exit."""
        monkeypatch.chdir(tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            initialize_environment(ServerEnv, print_config=False)

        assert exc_info.value.code == 1

    def test_initialize_environment_prints_config_by_default(
        self,
        valid_server_env: dict[str, str],
        set_environment: Any,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that print_config is called by default."""
        monkeypatch.chdir(tmp_path)
        set_environment(valid_server_env)

        initialize_environment(ServerEnv)

        output = capsys.readouterr().out
        assert "Environment variables loaded for server" in output
        assert "test-agent" in output

    def test_initialize_environment_skip_print_config(
        self,
        valid_server_env: dict[str, str],
        set_environment: Any,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that print_config can be skipped."""
        monkeypatch.chdir(tmp_path)
        set_environment(valid_server_env)

        initialize_environment(ServerEnv, print_config=False)

        assert capsys.readouterr().out == ""

    def test_legacy_false_override_slot_remains_compatible(
        self,
        valid_server_env: dict[str, str],
        set_environment: Any,
    ) -> None:
        """Test the old second positional argument remains safely accepted."""
        set_environment(valid_server_env)

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            env = initialize_environment(ServerEnv, False, False)

        assert env.agent_name == "test-agent"

    def test_legacy_true_override_warns_and_preserves_process_priority(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Test unsafe legacy override requests are warned about and ignored."""
        (tmp_path / ".env").write_text(
            "AGENT_NAME=dotenv-agent\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("AGENT_NAME", "process-agent")

        with pytest.warns(DeprecationWarning, match="deprecated and ignored"):
            env = initialize_environment(ServerEnv, True, False)

        assert env.agent_name == "process-agent"

    def test_initialize_environment_loads_dotenv(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Test settings load a real dotenv file from the current directory."""
        (tmp_path / ".env").write_text(
            "AGENT_NAME=dotenv-agent\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)

        env = initialize_environment(ServerEnv, print_config=False)

        assert env.agent_name == "dotenv-agent"

    def test_process_environment_overrides_dotenv(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Test VM/container environment values take priority over dotenv."""
        dotenv_database_url = "postgresql://dotenv:dotenv@localhost/dotenv"
        process_database_url = "postgresql://process:process@localhost/process"
        (tmp_path / ".env").write_text(
            "\n".join(
                [
                    "AGENT_NAME=dotenv-agent",
                    f"DATABASE_URL={dotenv_database_url}",
                    "OPENROUTER_API_KEY=dotenv-key",
                    "AGENT_DIR=/dotenv/agents",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("AGENT_NAME", "process-agent")
        monkeypatch.setenv("DATABASE_URL", process_database_url)
        monkeypatch.setenv("OPENROUTER_API_KEY", "process-key")
        monkeypatch.setenv("AGENT_DIR", "/process/agents")

        env = initialize_environment(ServerEnv, print_config=False)

        assert env.agent_name == "process-agent"
        assert env.database_url is not None
        assert env.database_url.get_secret_value() == process_database_url
        assert env.openrouter_api_key is not None
        assert env.openrouter_api_key.get_secret_value() == "process-key"
        assert env.agent_dir == "/process/agents"

    def test_blank_process_secrets_do_not_fall_through_to_dotenv(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Test selected blank process secrets become unset, not dotenv values."""
        (tmp_path / ".env").write_text(
            "\n".join(
                [
                    "AGENT_NAME=dotenv-agent",
                    "DATABASE_URL=postgresql://dotenv:dotenv@localhost/dotenv",
                    "OPENROUTER_API_KEY=dotenv-key",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("AGENT_NAME", "process-agent")
        monkeypatch.setenv("AGENT_ENGINE", "process-engine")
        monkeypatch.setenv("DATABASE_URL", "")
        monkeypatch.setenv("OPENROUTER_API_KEY", "")

        # The required alias is provided by the process environment above.
        env = ServerEnv()  # type: ignore[call-arg]

        assert env.database_url is None
        assert env.openrouter_api_key is None
        assert env.session_uri == "agentengine://process-engine"

    def test_constructor_values_override_environment_and_dotenv(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Test explicit constructor values have the highest settings priority."""
        (tmp_path / ".env").write_text(
            "AGENT_NAME=dotenv-agent\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("AGENT_NAME", "process-agent")
        monkeypatch.setenv("AGENT_DIR", "/process/agents")

        env = ServerEnv(
            AGENT_NAME="constructor-agent",
            AGENT_DIR="/constructor/agents",
        )

        assert env.agent_name == "constructor-agent"
        assert env.agent_dir == "/constructor/agents"

    def test_dotenv_loading_can_be_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Test callers can disable the configured dotenv source explicitly."""
        (tmp_path / ".env").write_text(
            "AGENT_NAME=dotenv-agent\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ValidationError):
            # Pydantic's generated signature omits this documented settings kwarg.
            ServerEnv(_env_file=None)  # type: ignore[call-arg]

    def test_missing_dotenv_uses_process_environment(
        self,
        valid_server_env: dict[str, str],
        set_environment: Any,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Test a missing optional dotenv file does not prevent startup."""
        monkeypatch.chdir(tmp_path)
        set_environment(valid_server_env)

        env = initialize_environment(ServerEnv, print_config=False)

        assert env.agent_name == "test-agent"

    def test_initialization_does_not_mutate_process_environment(
        self,
        valid_server_env: dict[str, str],
        set_environment: Any,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Test reading settings does not inject dotenv values into os.environ."""
        (tmp_path / ".env").write_text(
            "LOG_LEVEL=DEBUG\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        set_environment(valid_server_env)
        original_environment = dict(os.environ)

        env = initialize_environment(ServerEnv, print_config=False)

        assert env.log_level == "DEBUG"
        assert dict(os.environ) == original_environment

    def test_initialize_environment_prints_validation_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        """Test that validation errors are printed before exit."""
        monkeypatch.chdir(tmp_path)

        with pytest.raises(SystemExit):
            initialize_environment(ServerEnv, print_config=False)

        captured = capsys.readouterr()
        assert "Environment validation failed" in captured.out
        assert "AGENT_NAME" in captured.out

    def test_dotenv_placeholder_error_does_not_disclose_value(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        """Test startup rejects a dotenv placeholder without printing it."""
        placeholder = "YOUR-OPENROUTER-KEY-HERE"
        (tmp_path / ".env").write_text(
            f"AGENT_NAME=dotenv-agent\nOPENROUTER_API_KEY={placeholder}\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            initialize_environment(ServerEnv, print_config=False)

        captured = capsys.readouterr()
        assert exc_info.value.code == 1
        assert "OPENROUTER_API_KEY" in captured.out
        assert placeholder not in captured.out
        assert placeholder not in captured.err


class TestEdgeCases:
    """Tests for edge cases and field parsing."""

    def test_boolean_field_parsing(self, valid_server_env: dict[str, str]) -> None:
        """Test that boolean fields parse correctly from strings.

        Pydantic accepts multiple truthy/falsy string representations for bool fields.
        This test documents all accepted patterns.
        """
        # Test truthy values
        for truthy in ["true", "True", "TRUE"]:
            data = {**valid_server_env, "SERVE_WEB_INTERFACE": truthy}
            env = ServerEnv.model_validate(data)
            assert env.serve_web_interface is True, f"Failed for: {truthy}"

        # Test more truthy values
        for truthy in ["1", "yes", "Yes", "on", "On", "t", "y", "Y"]:
            data = {**valid_server_env, "RELOAD_AGENTS": truthy}
            env = ServerEnv.model_validate(data)
            assert env.reload_agents is True, f"Failed for: {truthy}"

        # Test falsy values
        for falsy in ["false", "False", "FALSE"]:
            data = {**valid_server_env, "SERVE_WEB_INTERFACE": falsy}
            env = ServerEnv.model_validate(data)
            assert env.serve_web_interface is False, f"Failed for: {falsy}"

        # Test more falsy values
        for falsy in ["0", "no", "No", "off", "Off", "f", "n", "N"]:
            data = {**valid_server_env, "RELOAD_AGENTS": falsy}
            env = ServerEnv.model_validate(data)
            assert env.reload_agents is False, f"Failed for: {falsy}"

    def test_boolean_field_invalid_values_raise_errors(
        self, valid_server_env: dict[str, str]
    ) -> None:
        """Test that invalid boolean values raise ValidationError.

        Documents what string values are NOT accepted for bool fields.
        """
        # Test invalid values that should raise ValidationError
        invalid_values = [
            "",  # Empty string
            "maybe",  # Invalid word
            "2",  # Invalid number (only "0" and "1" work)
            "yep",  # Similar to "yes" but not accepted
            "nope",  # Similar to "no" but not accepted
            "enabled",  # Descriptive but not accepted
            "disabled",  # Descriptive but not accepted
            "ok",  # Common but not accepted
            "sure",  # Informal affirmative
            "nah",  # Informal negative
        ]

        for invalid in invalid_values:
            data = {**valid_server_env, "SERVE_WEB_INTERFACE": invalid}
            with pytest.raises(ValidationError) as exc_info:
                ServerEnv.model_validate(data)

            # Verify the error is about bool parsing
            errors = exc_info.value.errors()
            assert any(error["type"] == "bool_parsing" for error in errors), (
                f"Expected bool_parsing error for: {invalid}"
            )

    def test_port_field_parsing(self, valid_server_env: dict[str, str]) -> None:
        """Test that port field parses integers from strings."""
        data = {**valid_server_env, "PORT": "9000"}

        env = ServerEnv.model_validate(data)
        assert env.port == 9000
        assert isinstance(env.port, int)
