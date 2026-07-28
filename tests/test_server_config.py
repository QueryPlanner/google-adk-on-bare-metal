"""Tests for server bootstrap configuration."""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, create_autospec, patch

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.adk.cli.fast_api import get_fast_api_app
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from agent import server


def _agent_dir(
    *,
    configured: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    """Create either an explicit agent directory or the module-derived default."""
    if configured:
        agent_dir = tmp_path / "configured agents"
        agent_dir.mkdir()
        monkeypatch.setenv("AGENT_DIR", str(agent_dir))
        return agent_dir

    source_root = tmp_path / "default agents"
    (source_root / "agent").mkdir(parents=True)
    monkeypatch.setattr(
        server,
        "__file__",
        str(source_root / "agent" / "server.py"),
    )
    return source_root


def _mock_instrumentor() -> MagicMock:
    """Return a strict mock for the external ADK instrumentor."""
    return cast(
        MagicMock,
        create_autospec(
            GoogleADKInstrumentor,
            spec_set=True,
        ),
    )


@pytest.mark.parametrize("configured_agent_dir", [False, True])
def test_create_app_uses_typed_settings_and_explicit_artifact_uri(
    configured_agent_dir: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Compose settings before passing one resolved path and URI to ADK."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_NAME", "test-agent")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")
    monkeypatch.setenv("TELEMETRY_NAMESPACE", "test-namespace")
    agent_dir = _agent_dir(
        configured=configured_agent_dir,
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
    )
    test_app = FastAPI()
    mock_get_app = create_autospec(
        get_fast_api_app,
        spec_set=True,
        return_value=test_app,
    )
    mock_instrumentor_class = _mock_instrumentor()

    def assert_otel_is_ready() -> None:
        resource_attributes = os.environ["OTEL_RESOURCE_ATTRIBUTES"]
        assert "service.name=test-agent" in resource_attributes
        assert "service.namespace=test-namespace" in resource_attributes

    mock_instrumentor_class.return_value.instrument.side_effect = assert_otel_is_ready
    with (
        patch.object(server, "get_fast_api_app", new=mock_get_app),
        patch.object(
            server,
            "GoogleADKInstrumentor",
            new=mock_instrumentor_class,
        ),
    ):
        app = server.create_app()

    resolved_agent_dir = agent_dir.resolve()
    artifact_dir = resolved_agent_dir / ".adk" / "artifacts"
    expected_db_kwargs = {
        "pool_pre_ping": True,
        "pool_recycle": 1800,
        "pool_size": 5,
        "max_overflow": 10,
        "pool_timeout": 30,
    }

    assert app is test_app
    mock_instrumentor_class.return_value.instrument.assert_called_once_with()
    mock_get_app.assert_called_once()
    call_kwargs = mock_get_app.call_args.kwargs
    assert call_kwargs["agents_dir"] == str(resolved_agent_dir)
    assert call_kwargs["artifact_service_uri"] == artifact_dir.as_uri()
    assert call_kwargs["session_service_uri"] == (
        "postgresql+asyncpg://user:pass@localhost/db"
    )
    assert call_kwargs["session_db_kwargs"] == expected_db_kwargs
    assert artifact_dir.is_dir()
    assert list(artifact_dir.iterdir()) == []


def test_health_endpoint_reports_process_liveness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify the factory registers the container liveness contract."""
    agent_dir = tmp_path / "agents"
    agent_dir.mkdir()
    test_app = FastAPI()
    mock_get_app = create_autospec(
        get_fast_api_app,
        spec_set=True,
        return_value=test_app,
    )
    mock_instrumentor_class = _mock_instrumentor()

    with (
        patch.dict(
            os.environ,
            {
                "AGENT_DIR": str(agent_dir),
                "AGENT_NAME": "health-test-agent",
            },
            clear=True,
        ),
        patch.object(server, "get_fast_api_app", new=mock_get_app),
        patch.object(
            server,
            "GoogleADKInstrumentor",
            new=mock_instrumentor_class,
        ),
    ):
        app = server.create_app()
        with TestClient(app) as client:
            response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_import_has_no_application_or_storage_side_effect() -> None:
    """Require callers to opt into application and storage creation."""
    assert not hasattr(server, "app")


def test_main_runs_factory_application(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run uvicorn with the configured host and port after preparation."""
    agent_dir = tmp_path / "agents"
    agent_dir.mkdir()
    test_app = FastAPI()
    mock_get_app = create_autospec(
        get_fast_api_app,
        spec_set=True,
        return_value=test_app,
    )
    mock_instrumentor_class = _mock_instrumentor()
    mock_uvicorn_run = create_autospec(
        uvicorn.run,
        spec_set=True,
    )
    monkeypatch.setenv("AGENT_DIR", str(agent_dir))
    monkeypatch.setenv("AGENT_NAME", "main-test-agent")
    monkeypatch.setenv("HOST", "127.0.0.2")
    monkeypatch.setenv("PORT", "9090")

    with (
        patch.object(server, "get_fast_api_app", new=mock_get_app),
        patch.object(
            server,
            "GoogleADKInstrumentor",
            new=mock_instrumentor_class,
        ),
        patch.object(server.uvicorn, "run", new=mock_uvicorn_run),
    ):
        server.main()

    mock_uvicorn_run.assert_called_once_with(
        test_app,
        host="127.0.0.2",
        port=9090,
    )


def test_main_logs_stable_storage_failure_without_traceback(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exit one without exposing the underlying filesystem failure."""
    agent_dir = tmp_path / "agents"
    outside_dir = tmp_path / "outside"
    agent_dir.mkdir()
    outside_dir.mkdir()
    (agent_dir / ".adk").symlink_to(outside_dir, target_is_directory=True)
    mock_get_app = create_autospec(
        get_fast_api_app,
        spec_set=True,
    )
    mock_instrumentor_class = _mock_instrumentor()
    mock_uvicorn_run = create_autospec(
        uvicorn.run,
        spec_set=True,
    )
    monkeypatch.setenv("AGENT_DIR", str(agent_dir))
    monkeypatch.setenv("AGENT_NAME", "failure-test-agent")
    caplog.set_level(logging.ERROR, logger="agent.server")

    with (
        patch.object(server, "get_fast_api_app", new=mock_get_app),
        patch.object(
            server,
            "GoogleADKInstrumentor",
            new=mock_instrumentor_class,
        ),
        patch.object(server.uvicorn, "run", new=mock_uvicorn_run),
        pytest.raises(SystemExit) as exit_error,
    ):
        server.main()

    failure_records = [
        record for record in caplog.records if record.name == "agent.server"
    ]
    assert exit_error.value.code == 1
    assert exit_error.value.__cause__ is None
    assert exit_error.value.__suppress_context__
    assert [record.getMessage() for record in failure_records] == [
        "Artifact storage is unavailable."
    ]
    assert all(record.exc_info is None for record in failure_records)
    assert all(record.stack_info is None for record in failure_records)
    captured = capsys.readouterr()
    process_output = captured.out + captured.err
    for forbidden in (
        "AGENT_DIR",
        str(agent_dir),
        str(outside_dir),
        "ValueError",
        "Traceback",
    ):
        assert forbidden not in process_output
    mock_get_app.assert_not_called()
    mock_uvicorn_run.assert_not_called()


def test_main_sanitizes_post_creation_probe_failure(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Never expose a raw sync error or random probe after directory creation."""
    agent_dir = tmp_path / "agents"
    agent_dir.mkdir()
    raw_failure = OSError(
        f"sync failed at {agent_dir}/.artifact-storage-probe-secret/.probe-secret"
    )
    fsync_failure = create_autospec(
        os.fsync,
        spec_set=True,
        side_effect=raw_failure,
    )
    mock_get_app = create_autospec(
        get_fast_api_app,
        spec_set=True,
    )
    mock_instrumentor_class = _mock_instrumentor()
    mock_uvicorn_run = create_autospec(
        uvicorn.run,
        spec_set=True,
    )
    monkeypatch.setenv("AGENT_DIR", str(agent_dir))
    monkeypatch.setenv("AGENT_NAME", "probe-failure-test-agent")
    caplog.set_level(logging.ERROR, logger="agent.server")

    with (
        patch("agent.artifact_storage.os.fsync", new=fsync_failure),
        patch.object(server, "get_fast_api_app", new=mock_get_app),
        patch.object(
            server,
            "GoogleADKInstrumentor",
            new=mock_instrumentor_class,
        ),
        patch.object(server.uvicorn, "run", new=mock_uvicorn_run),
        pytest.raises(SystemExit) as exit_error,
    ):
        server.main()

    failure_records = [
        record for record in caplog.records if record.name == "agent.server"
    ]
    captured = capsys.readouterr()
    process_output = captured.out + captured.err
    assert exit_error.value.code == 1
    assert [record.getMessage() for record in failure_records] == [
        "Artifact storage is unavailable."
    ]
    for forbidden in (
        "AGENT_DIR",
        str(agent_dir),
        str(raw_failure),
        ".artifact-storage-probe-",
        ".probe-",
        "OSError",
        "Traceback",
    ):
        assert forbidden not in process_output
    assert list((agent_dir / ".adk" / "artifacts").iterdir()) == []
    fsync_failure.assert_called_once()
    mock_get_app.assert_not_called()
    mock_uvicorn_run.assert_not_called()


def test_module_entrypoint_sanitizes_real_startup_failure(tmp_path: Path) -> None:
    """Verify the supported process boundary without imported-module mocks."""
    missing_agent_dir = tmp_path / "missing-agent-dir"
    environment = {
        "ADK_DISABLE_LOAD_DOTENV": "true",
        "ADK_DISABLE_LOCAL_STORAGE": "true",
        "AGENT_DIR": str(missing_agent_dir),
        "AGENT_NAME": "module-failure-test-agent",
        "ALLOW_ORIGINS": "[]",
        "LANG": "C",
        "OTEL_SDK_DISABLED": "true",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONUNBUFFERED": "1",
        "RELOAD_AGENTS": "false",
        "SERVE_WEB_INTERFACE": "false",
    }

    result = subprocess.run(  # noqa: S603 - fixed interpreter and module
        [sys.executable, "-m", "agent.server"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    process_output = result.stdout + result.stderr
    message_lines = [
        line
        for line in process_output.splitlines()
        if "Artifact storage is unavailable." in line
    ]
    assert result.returncode == 1
    assert len(message_lines) == 1
    assert message_lines[0].endswith(
        "[ERROR] agent.server: Artifact storage is unavailable."
    )
    for forbidden in (
        "AGENT_DIR",
        str(missing_agent_dir),
        "FileNotFoundError",
        "OSError",
        "[Errno",
        "Traceback",
        ".artifact-storage-probe-",
        ".probe-",
    ):
        assert forbidden not in process_output
