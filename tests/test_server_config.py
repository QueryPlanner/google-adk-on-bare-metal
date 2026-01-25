"""Tests for server configuration."""

import sys
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_dependencies() -> Generator[MagicMock]:
    """Mock external dependencies to prevent side effects during import."""
    with (
        patch("google.adk.cli.fast_api.get_fast_api_app") as mock_get_app,
        patch("agent.utils.initialize_environment") as mock_init_env,
        patch("agent.utils.configure_otel_resource"),
        patch("openinference.instrumentation.google_adk.GoogleADKInstrumentor"),
        patch("agent.utils.setup_logging"),
    ):
        # Setup basic env mock
        mock_env = MagicMock()
        mock_env.session_uri = "postgresql://user:pass@localhost/db"
        mock_env.allow_origins_list = ["*"]
        mock_env.serve_web_interface = True
        mock_env.reload_agents = False

        # Helper to support .host and .port access if needed
        mock_env.host = "127.0.0.1"
        mock_env.port = 8080

        # DB pool settings
        mock_env.db_pool_pre_ping = True
        mock_env.db_pool_recycle = 1800
        mock_env.db_pool_size = 5
        mock_env.db_max_overflow = 10
        mock_env.db_pool_timeout = 30

        mock_init_env.return_value = mock_env

        yield mock_get_app


def test_server_session_db_kwargs_configuration(mock_dependencies: MagicMock) -> None:
    """Verify session_db_kwargs is configured and passed to get_fast_api_app."""
    # Ensure agent.server is reloaded if it was already imported
    if "agent.server" in sys.modules:
        del sys.modules["agent.server"]

    import agent.server  # noqa: F401

    # expected kwargs
    expected_db_kwargs = {
        "pool_pre_ping": True,
        "pool_recycle": 1800,
        "pool_size": 5,
        "max_overflow": 10,
        "pool_timeout": 30,
    }

    # Verify the call
    mock_dependencies.assert_called_once()
    call_kwargs = mock_dependencies.call_args[1]

    assert "session_db_kwargs" in call_kwargs
    assert call_kwargs["session_db_kwargs"] == expected_db_kwargs
