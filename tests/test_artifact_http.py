"""Real HTTP integration coverage for persistent ADK artifacts."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
from fastapi.testclient import TestClient
from google.genai import types

_APP_NAME = "agent"
_USER_ID = "artifact-user"
_SESSION_ID = "artifact-session"
_ARTIFACT_NAME = "proof.txt"
_ARTIFACT_TEXT = "persistent artifact"
_ARTIFACTS_PATH = f"/apps/{_APP_NAME}/users/{_USER_ID}/sessions/{_SESSION_ID}/artifacts"


@pytest.fixture
def configured_artifact_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[tuple[ModuleType, Path]]:
    """Import the production server against one isolated artifact root."""
    agent_dir = tmp_path / "agent root"
    agent_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ADK_DISABLE_LOAD_DOTENV", "true")
    monkeypatch.setenv("ADK_DISABLE_LOCAL_STORAGE", "true")
    monkeypatch.setenv("AGENT_DIR", str(agent_dir))
    monkeypatch.setenv("AGENT_NAME", "artifact-http-test")
    monkeypatch.setenv("ALLOW_ORIGINS", "[]")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.setenv("RELOAD_AGENTS", "false")
    monkeypatch.setenv("SERVE_WEB_INTERFACE", "false")

    sys.modules.pop("agent.server", None)
    server = importlib.import_module("agent.server")
    try:
        yield server, agent_dir
    finally:
        sys.modules.pop("agent.server", None)


def test_artifact_survives_fresh_production_app(
    configured_artifact_server: tuple[ModuleType, Path],
) -> None:
    """Save through HTTP and load the exact version from a fresh app."""
    server, agent_dir = configured_artifact_server
    artifact = types.Part(text=_ARTIFACT_TEXT)
    request_body = {
        "filename": _ARTIFACT_NAME,
        "artifact": artifact.model_dump(
            mode="json",
            by_alias=True,
            exclude_none=True,
        ),
    }

    with TestClient(server.create_app()) as client:
        saved_response = client.post(_ARTIFACTS_PATH, json=request_body)

    assert saved_response.status_code == 200
    saved = saved_response.json()
    assert saved["version"] == 0

    with TestClient(server.create_app()) as client:
        loaded_response = client.get(
            f"{_ARTIFACTS_PATH}/{_ARTIFACT_NAME}/versions/{saved['version']}"
        )

    assert loaded_response.status_code == 200
    assert loaded_response.json() == {"text": _ARTIFACT_TEXT}
    assert (agent_dir / ".adk" / "artifacts").is_dir()


def test_missing_artifact_returns_exact_404(
    configured_artifact_server: tuple[ModuleType, Path],
) -> None:
    """Expose ADK's exact not-found contract through the production app."""
    server, _ = configured_artifact_server

    with TestClient(server.create_app()) as client:
        response = client.get(f"{_ARTIFACTS_PATH}/missing.txt/versions/0")

    assert response.status_code == 404
    assert response.json() == {"detail": "Artifact not found"}
