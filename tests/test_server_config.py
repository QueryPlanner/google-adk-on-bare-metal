"""Tests for server bootstrap configuration."""

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import create_autospec, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.adk.cli.fast_api import get_fast_api_app
from openinference.instrumentation.google_adk import GoogleADKInstrumentor


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
    mock_get_app = create_autospec(
        get_fast_api_app,
        spec_set=True,
        return_value=mock_app,
    )
    mock_instrumentor_class = create_autospec(
        GoogleADKInstrumentor,
        spec_set=True,
    )
    with (
        patch(
            "google.adk.cli.fast_api.get_fast_api_app",
            new=mock_get_app,
        ),
        patch(
            "openinference.instrumentation.google_adk.GoogleADKInstrumentor",
            new=mock_instrumentor_class,
        ),
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


def test_health_endpoint_reports_process_liveness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify the registered health route serves the container liveness contract."""
    monkeypatch.chdir(tmp_path)
    test_app = FastAPI()
    mock_get_app = create_autospec(
        get_fast_api_app,
        spec_set=True,
        return_value=test_app,
    )
    mock_instrumentor_class = create_autospec(
        GoogleADKInstrumentor,
        spec_set=True,
    )

    with (
        patch.dict(os.environ, {"AGENT_NAME": "health-test-agent"}, clear=True),
        patch(
            "google.adk.cli.fast_api.get_fast_api_app",
            new=mock_get_app,
        ),
        patch(
            "openinference.instrumentation.google_adk.GoogleADKInstrumentor",
            new=mock_instrumentor_class,
        ),
    ):
        sys.modules.pop("agent.server", None)
        try:
            server = importlib.import_module("agent.server")
            with TestClient(server.app) as client:
                response = client.get("/health")
        finally:
            sys.modules.pop("agent.server", None)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
