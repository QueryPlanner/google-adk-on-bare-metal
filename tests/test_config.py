"""Comprehensive unit tests for config module."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from agent_foundation.utils.config import (
    ServerEnv,
    initialize_environment,
)


class TestServerEnv:
    """Tests for ServerEnv model."""

    def test_valid_server_env_creation(self, valid_server_env: dict[str, str]) -> None:
        """Test creating ServerEnv with valid required fields."""
        env = ServerEnv.model_validate(valid_server_env)

        assert env.google_cloud_project == "test-project"
        assert env.agent_name == "test-agent"

    def test_server_env_missing_required_field_raises_validation_error(self) -> None:
        """Test that missing required fields raise ValidationError."""
        data: dict[str, str] = {
            # Missing GOOGLE_CLOUD_PROJECT, AGENT_NAME,
            # and OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT
        }

        with pytest.raises(ValidationError) as exc_info:
            ServerEnv.model_validate(data)

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("GOOGLE_CLOUD_PROJECT",) for error in errors)
        assert any(error["loc"] == ("AGENT_NAME",) for error in errors)
        assert any(
            error["loc"] == ("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",)
            for error in errors
        )

    def test_server_env_optional_fields_use_defaults(
        self, valid_server_env: dict[str, str]
    ) -> None:
        """Test that optional fields use default values when not provided."""
        env = ServerEnv.model_validate(valid_server_env)

        # Check defaults
        assert env.google_cloud_location == "us-central1"
        assert env.log_level == "INFO"
        assert env.serve_web_interface is False
        assert env.reload_agents is False
        assert env.agent_engine is None
        assert env.artifact_service_uri is None
        assert env.allow_origins == '["http://127.0.0.1", "http://127.0.0.1:8000"]'
        assert env.host == "127.0.0.1"
        assert env.port == 8000

    def test_server_env_optional_fields_with_values(
        self, valid_server_env: dict[str, str]
    ) -> None:
        """Test setting optional fields with actual values."""
        data = {
            **valid_server_env,
            "GOOGLE_CLOUD_LOCATION": "us-west1",
            "AGENT_NAME": "custom-agent",
            "LOG_LEVEL": "DEBUG",
            "SERVE_WEB_INTERFACE": "true",
            "RELOAD_AGENTS": "true",
            "AGENT_ENGINE": "test-engine-id",
            "ARTIFACT_SERVICE_URI": "gs://test-bucket",
            "ALLOW_ORIGINS": '["http://localhost:3000"]',
            "HOST": "0.0.0.0",  # noqa: S104
            "PORT": "9000",
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "false",
        }

        env = ServerEnv.model_validate(data)

        assert env.google_cloud_location == "us-west1"
        assert env.agent_name == "custom-agent"
        assert env.log_level == "DEBUG"
        assert env.serve_web_interface is True
        assert env.reload_agents is True
        assert env.agent_engine == "test-engine-id"
        assert env.artifact_service_uri == "gs://test-bucket"
        assert env.allow_origins == '["http://localhost:3000"]'
        assert env.host == "0.0.0.0"  # noqa: S104
        assert env.port == 9000
        assert env.otel_capture_content is False

    def test_agent_engine_uri_property(self, valid_server_env: dict[str, str]) -> None:
        """Test that agent_engine_uri property is computed correctly."""
        # Without agent_engine
        env = ServerEnv.model_validate(valid_server_env)
        assert env.agent_engine_uri is None

        # With agent_engine
        data = {**valid_server_env, "AGENT_ENGINE": "test-engine-id"}
        env = ServerEnv.model_validate(data)
        assert env.agent_engine_uri == "agentengine://test-engine-id"

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
        assert "test-project" in output
        assert "test-agent" in output
        assert "GOOGLE_CLOUD_PROJECT" in output
        assert "AGENT_NAME" in output
        assert "LOG_LEVEL" in output

    def test_server_env_ignores_extra_fields(
        self, valid_server_env: dict[str, str]
    ) -> None:
        """Test that extra environment variables are ignored."""
        data = {**valid_server_env, "EXTRA_VAR": "extra-value", "PATH": "/usr/bin"}

        env = ServerEnv.model_validate(data)
        assert env.google_cloud_project == "test-project"
        # Extra fields should not be included
        assert not hasattr(env, "EXTRA_VAR")
        assert not hasattr(env, "PATH")


class TestInitializeEnvironment:
    """Tests for initialize_environment factory function."""

    def test_initialize_environment_success(
        self,
        valid_server_env: dict[str, str],
        set_environment: Any,
        mock_load_dotenv: MagicMock,
    ) -> None:
        """Test successful environment initialization."""
        # Set environment variables
        set_environment(valid_server_env)

        env = initialize_environment(ServerEnv, print_config=False)

        mock_load_dotenv.assert_called_once_with(override=True)
        assert env.google_cloud_project == "test-project"
        assert env.agent_name == "test-agent"

    def test_initialize_environment_validation_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_load_dotenv: MagicMock,
        mock_sys_exit: MagicMock,
    ) -> None:
        """Test that validation failure causes sys.exit."""
        # Set incomplete environment (missing required field)
        # Don't set GOOGLE_CLOUD_PROJECT
        monkeypatch.setenv("AGENT_NAME", "test-agent")

        with pytest.raises(SystemExit):
            initialize_environment(ServerEnv, print_config=False)

        mock_sys_exit.assert_called_once_with(1)

    def test_initialize_environment_prints_config_by_default(
        self,
        valid_server_env: dict[str, str],
        set_environment: Any,
        mock_load_dotenv: MagicMock,
        mock_print_config: Any,
    ) -> None:
        """Test that print_config is called by default."""
        set_environment(valid_server_env)

        with mock_print_config(ServerEnv) as mock_print:
            initialize_environment(ServerEnv)

        mock_print.assert_called_once()

    def test_initialize_environment_skip_print_config(
        self,
        valid_server_env: dict[str, str],
        set_environment: Any,
        mock_load_dotenv: MagicMock,
        mock_print_config: Any,
    ) -> None:
        """Test that print_config can be skipped."""
        set_environment(valid_server_env)

        with mock_print_config(ServerEnv) as mock_print:
            initialize_environment(ServerEnv, print_config=False)

        mock_print.assert_not_called()

    def test_initialize_environment_override_dotenv_false(
        self,
        valid_server_env: dict[str, str],
        set_environment: Any,
        mock_load_dotenv: MagicMock,
    ) -> None:
        """Test that override_dotenv can be set to False."""
        set_environment(valid_server_env)

        initialize_environment(ServerEnv, override_dotenv=False, print_config=False)

        mock_load_dotenv.assert_called_once_with(override=False)

    def test_initialize_environment_prints_validation_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        mock_load_dotenv: MagicMock,
        mock_sys_exit: MagicMock,
    ) -> None:
        """Test that validation errors are printed before exit."""
        # Set incomplete environment
        monkeypatch.setenv("AGENT_NAME", "test-agent")
        # Missing GOOGLE_CLOUD_PROJECT

        with pytest.raises(SystemExit):
            initialize_environment(ServerEnv, print_config=False)

        captured = capsys.readouterr()
        assert "Environment validation failed" in captured.out


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
