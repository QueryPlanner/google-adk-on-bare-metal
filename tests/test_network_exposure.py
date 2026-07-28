"""Single-VM network exposure and headless ADK security contracts."""

import subprocess
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]
from fastapi.testclient import TestClient
from google.adk.cli.fast_api import get_fast_api_app
from starlette.routing import Mount, Route, WebSocketRoute

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = REPOSITORY_ROOT / "compose.yaml"
ENV_EXAMPLE_PATH = REPOSITORY_ROOT / ".env.example"
SETUP_PATH = REPOSITORY_ROOT / "setup.sh"


def _environment_values(document: str) -> dict[str, str]:
    """Parse the non-secret example environment without source expansion."""
    values: dict[str, str] = {}
    for raw_line in document.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        assert separator == "=", f"Invalid environment line: {raw_line!r}"
        values[key] = value
    return values


def test_compose_defaults_to_loopback_and_headless_runtime() -> None:
    """Keep the host boundary closed while the container listens internally."""
    compose = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    agent = compose["services"]["agent"]

    assert agent["ports"] == [
        {
            "target": 8080,
            "published": "8080",
            "host_ip": "${AGENT_PUBLISH_HOST:-127.0.0.1}",
            "protocol": "tcp",
        }
    ]
    assert agent["environment"] == [
        "HOST=0.0.0.0",
        "PORT=8080",
        "AGENT_DIR=/app/src",
        "AGENT_NAME=local-dev",
        "LOG_LEVEL=INFO",
        "RELOAD_AGENTS=${RELOAD_AGENTS:-false}",
        "SERVE_WEB_INTERFACE=${SERVE_WEB_INTERFACE:-false}",
    ]


def test_example_environment_uses_safe_vm_defaults() -> None:
    """Protect both Compose and direct systemd deployments by default."""
    values = _environment_values(ENV_EXAMPLE_PATH.read_text(encoding="utf-8"))

    assert values["HOST"] == "127.0.0.1"
    assert values["PORT"] == "8080"
    assert values["AGENT_PUBLISH_HOST"] == "127.0.0.1"
    assert values["SERVE_WEB_INTERFACE"] == "false"
    assert values["RELOAD_AGENTS"] == "false"


def test_setup_firewall_does_not_open_the_adk_port() -> None:
    """Leave only SSH and operator-supplied proxy ingress open on fresh VMs."""
    document = SETUP_PATH.read_text(encoding="utf-8")
    allow_rules = [
        line.strip()
        for line in document.splitlines()
        if line.strip().startswith("ufw allow ")
    ]
    syntax = subprocess.run(  # noqa: S603 - fixed repository script
        ["/bin/bash", "-n", str(SETUP_PATH)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert syntax.returncode == 0, syntax.stderr
    assert allow_rules == [
        "ufw allow ssh",
        "ufw allow 80/tcp",
        "ufw allow 443/tcp",
    ]
    assert "ufw allow 8080" not in document


@pytest.mark.parametrize(
    ("server_version", "docker_exit", "expected_returncode"),
    (
        ("28.0.0", 0, 0),
        ("29.1.3", 0, 0),
        ("27.5.1", 0, 1),
        ("not-a-version", 0, 1),
        ("", 1, 1),
    ),
)
def test_setup_enforces_minimum_docker_server_version(
    tmp_path: Path,
    server_version: str,
    docker_exit: int,
    expected_returncode: int,
) -> None:
    """Fail closed when an existing Docker daemon lacks loopback hardening."""
    docker_path = tmp_path / "docker"
    docker_path.write_text(
        "#!/bin/bash\n"
        "set -eu\n"
        """[ "$*" = "version --format {{.Server.Version}}" ]\n"""
        """if [ "$FAKE_DOCKER_EXIT" -ne 0 ]; then\n"""
        """  exit "$FAKE_DOCKER_EXIT"\n"""
        "fi\n"
        "printf '%s\\n' \"$FAKE_DOCKER_SERVER_VERSION\"\n",
        encoding="utf-8",
    )
    docker_path.chmod(0o755)
    environment = {
        "FAKE_DOCKER_EXIT": str(docker_exit),
        "FAKE_DOCKER_SERVER_VERSION": server_version,
        "LANG": "C",
        "PATH": f"{tmp_path}:/usr/bin:/bin",
    }

    result = subprocess.run(  # noqa: S603 - fixed repository script
        ["/bin/bash", str(SETUP_PATH), "--verify-docker-version"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == expected_returncode
    if expected_returncode == 0:
        assert f"Docker Engine {server_version} meets" in result.stdout
    else:
        assert "ERROR" in result.stdout


def test_headless_adk_removes_builder_but_not_unauthenticated_api(
    tmp_path: Path,
) -> None:
    """Prove disabling the development UI is not application authentication."""
    app = get_fast_api_app(agents_dir=str(tmp_path), web=False)
    paths = {
        route.path
        for route in app.routes
        if isinstance(route, (Route, WebSocketRoute, Mount))
    }

    assert {"/", "/dev-ui", "/builder/save"}.isdisjoint(paths)
    assert {
        "/docs",
        "/run",
        "/run_live",
        "/run_sse",
        "/apps/{app_name}/users/{user_id}/sessions",
        ("/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts"),
    } <= paths

    client = TestClient(app)
    assert client.get("/docs").status_code == 200
    run_response = client.post("/run", json={})
    assert run_response.status_code == 422
    assert run_response.status_code not in {401, 403}
