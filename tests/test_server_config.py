"""Tests for server bootstrap configuration."""

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import create_autospec, patch

import pytest
from fastapi import FastAPI


@pytest.mark.parametrize("configured_agent_dir", [None, "/srv/test-agents"])
def test_server_bootstrap_uses_typed_settings_before_instrumentation(
    configured_agent_dir: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify server, database, agent-dir, and OTel settings are composed."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_NAME", "test-agent")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")
    monkeypatch.setenv("TELEMETRY_NAMESPACE", "test-namespace")
    if configured_agent_dir is not None:
        monkeypatch.setenv("AGENT_DIR", configured_agent_dir)

    mock_app = create_autospec(FastAPI, instance=True, spec_set=True)
    with (
        patch(
            "google.adk.cli.fast_api.get_fast_api_app",
            autospec=True,
            return_value=mock_app,
        ) as mock_get_app,
        patch(
            "openinference.instrumentation.google_adk.GoogleADKInstrumentor",
            autospec=True,
        ) as mock_instrumentor_class,
    ):

        def assert_otel_is_ready() -> None:
            resource_attributes = os.environ["OTEL_RESOURCE_ATTRIBUTES"]
            assert "service.name=test-agent" in resource_attributes
            assert "service.namespace=test-namespace" in resource_attributes

        mock_instrumentor_class.return_value.instrument.side_effect = (
            assert_otel_is_ready
        )
        sys.modules.pop("agent.server", None)
        server = importlib.import_module("agent.server")

    server_file = server.__file__
    assert server_file is not None
    expected_agent_dir = configured_agent_dir or str(
        Path(server_file).resolve().parent.parent
    )
    expected_db_kwargs = {
        "pool_pre_ping": True,
        "pool_recycle": 1800,
        "pool_size": 5,
        "max_overflow": 10,
        "pool_timeout": 30,
    }

    mock_instrumentor_class.return_value.instrument.assert_called_once_with()
    mock_get_app.assert_called_once()
    call_kwargs = mock_get_app.call_args.kwargs
    assert call_kwargs["agents_dir"] == expected_agent_dir
    assert call_kwargs["session_service_uri"] == (
        "postgresql+asyncpg://user:pass@localhost/db"
    )
    assert call_kwargs["session_db_kwargs"] == expected_db_kwargs
    assert expected_agent_dir == server.AGENT_DIR

    sys.modules.pop("agent.server", None)
