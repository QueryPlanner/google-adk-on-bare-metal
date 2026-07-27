"""Real PostgreSQL integration coverage for the ADK session HTTP API."""

import asyncio
import json
import os
import re
import secrets
import subprocess
import sys
import textwrap
import uuid
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qsl, quote, unquote, urlsplit, urlunsplit

import asyncpg  # type: ignore[import-untyped]
import pytest

_ADMIN_URL_ENV = "TEST_POSTGRES_ADMIN_URL"
_DATABASE_NAME_PATTERN = re.compile(r"\Aadk_test_[0-9a-f]{32}\Z")
_ROLE_NAME_PATTERN = re.compile(r"\Aadk_role_[0-9a-f]{32}\Z")
_ROLE_PASSWORD_PATTERN = re.compile(r"\A[0-9a-f]{64}\Z")
_POSTGRESQL_URL_PATTERN = re.compile(
    r"""postgresql(?:\+asyncpg)?://[^\s"'<>]+""",
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
_RESULT_PREFIX = "POSTGRES_PROBE_RESULT="
_APP_NAME = "agent"
_USER_ID = "integration-user"
_SESSION_ID = "persistent-session"
_PERSISTED_VALUE = "real-postgres"

_CREATE_PHASE = """
import json

from fastapi.testclient import TestClient

from agent import server

base_path = "/apps/agent/users/integration-user/sessions"
payload = {
    "session_id": "persistent-session",
    "state": {"persisted": "real-postgres"},
}

with TestClient(server.app) as client:
    created = client.post(base_path, json=payload)
    duplicate = client.post(base_path, json=payload)

result = {
    "created_body": created.json(),
    "created_status": created.status_code,
    "duplicate_status": duplicate.status_code,
}
print("POSTGRES_PROBE_RESULT=" + json.dumps(result))
"""

_RESTART_PHASE = """
import json

from fastapi.testclient import TestClient

from agent import server

base_path = "/apps/agent/users/integration-user/sessions"
session_path = base_path + "/persistent-session"

with TestClient(server.app) as client:
    fetched = client.get(session_path)
    listed = client.get(base_path)
    deleted = client.delete(session_path)
    missing = client.get(session_path)

result = {
    "delete_status": deleted.status_code,
    "fetched_body": fetched.json(),
    "fetched_status": fetched.status_code,
    "list_body": listed.json(),
    "list_status": listed.status_code,
    "missing_status": missing.status_code,
}
print("POSTGRES_PROBE_RESULT=" + json.dumps(result))
"""


def _validate_database_name(database_name: str) -> str:
    """Return a database name only when it matches the destructive-test prefix."""
    if _DATABASE_NAME_PATTERN.fullmatch(database_name) is None:
        msg = "PostgreSQL test database name is invalid"
        raise ValueError(msg)
    return database_name


def _validate_role_name(role_name: str) -> str:
    """Return a role name only when it matches the destructive-test prefix."""
    if _ROLE_NAME_PATTERN.fullmatch(role_name) is None:
        msg = "PostgreSQL test role name is invalid"
        raise ValueError(msg)
    return role_name


def _validate_role_password(role_password: str) -> str:
    """Return only the generated fixed-alphabet role password."""
    if _ROLE_PASSWORD_PATTERN.fullmatch(role_password) is None:
        msg = "PostgreSQL test role password is invalid"
        raise ValueError(msg)
    return role_password


def _new_database_identity() -> tuple[str, str, str]:
    """Return validated names and a password for one isolated database owner."""
    identifier = uuid.uuid4().hex
    database_name = _validate_database_name(f"adk_test_{identifier}")
    role_name = _validate_role_name(f"adk_role_{identifier}")
    role_password = _validate_role_password(secrets.token_hex(32))
    if database_name.removeprefix("adk_test_") != role_name.removeprefix("adk_role_"):
        msg = "Generated PostgreSQL test identity is inconsistent"
        raise RuntimeError(msg)
    return database_name, role_name, role_password


def _database_url(
    admin_url: str,
    database_name: str,
    role_name: str,
    role_password: str,
) -> str:
    """Build the isolated owner's URL without retaining admin user information."""
    database_name = _validate_database_name(database_name)
    role_name = _validate_role_name(role_name)
    role_password = _validate_role_password(role_password)
    parsed = urlsplit(admin_url)
    if parsed.scheme != "postgresql" or parsed.hostname is None:
        msg = f"{_ADMIN_URL_ENV} must be a postgresql:// URL with a hostname"
        raise ValueError(msg)
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    port = f":{parsed.port}" if parsed.port is not None else ""
    netloc = f"{quote(role_name, safe='')}:{quote(role_password, safe='')}@{host}{port}"
    query_pairs = parse_qsl(
        parsed.query,
        keep_blank_values=True,
        strict_parsing=True,
    )
    unsupported_keys = sorted({key for key, _ in query_pairs})
    if unsupported_keys:
        msg = f"{_ADMIN_URL_ENV} has unsupported query parameter keys: " + ", ".join(
            unsupported_keys
        )
        raise ValueError(msg)
    return urlunsplit(
        (
            parsed.scheme,
            netloc,
            f"/{database_name}",
            "",
            "",
        )
    )


async def _close_connection(connection: asyncpg.Connection) -> None:
    """Close a completed connection without masking the operation outcome."""
    try:
        await connection.close(timeout=10)
    except Exception:
        with suppress(Exception):
            connection.terminate()


async def _create_role(
    admin_url: str,
    role_name: str,
    role_password: str,
) -> None:
    """Create one explicitly restricted login role for the test database."""
    role_name = _validate_role_name(role_name)
    role_password = _validate_role_password(role_password)
    connection = await asyncpg.connect(
        admin_url,
        timeout=10,
        command_timeout=10,
    )
    try:
        await connection.execute(
            f"""
            CREATE ROLE "{role_name}"
            WITH LOGIN
                 PASSWORD '{role_password}'
                 NOSUPERUSER
                 NOCREATEDB
                 NOCREATEROLE
                 NOINHERIT
                 NOREPLICATION
                 NOBYPASSRLS
                 CONNECTION LIMIT 5
            """  # noqa: S608
        )
    finally:
        await _close_connection(connection)


async def _create_database(
    admin_url: str,
    database_name: str,
    role_name: str,
) -> None:
    """Create one test database owned by its restricted role."""
    database_name = _validate_database_name(database_name)
    role_name = _validate_role_name(role_name)
    connection = await asyncpg.connect(
        admin_url,
        timeout=10,
        command_timeout=10,
    )
    try:
        await connection.execute(
            f"""
            CREATE DATABASE "{database_name}"
            OWNER "{role_name}"
            """  # noqa: S608
        )
    finally:
        await _close_connection(connection)


async def _role_is_restricted(admin_url: str, role_name: str) -> bool:
    """Verify the generated owner lacks cluster administration privileges."""
    role_name = _validate_role_name(role_name)
    connection = await asyncpg.connect(
        admin_url,
        timeout=10,
        command_timeout=10,
    )
    try:
        result = await connection.fetchval(
            """
            SELECT NOT rolsuper
               AND NOT rolcreatedb
               AND NOT rolcreaterole
               AND NOT rolinherit
               AND NOT rolreplication
               AND NOT rolbypassrls
            FROM pg_roles
            WHERE rolname = $1
            """,
            role_name,
        )
        return result is True
    finally:
        await _close_connection(connection)


async def _drop_database(admin_url: str, database_name: str) -> None:
    """Force-drop the database, terminating its clients within the timeout."""
    database_name = _validate_database_name(database_name)
    connection = await asyncpg.connect(
        admin_url,
        timeout=10,
        command_timeout=10,
    )
    try:
        # PostgreSQL 17 WITH (FORCE) terminates existing database connections.
        await connection.execute(
            f"""
            DROP DATABASE IF EXISTS "{database_name}"
            WITH (FORCE)
            """  # noqa: S608
        )
    finally:
        await _close_connection(connection)


async def _drop_role(admin_url: str, role_name: str) -> None:
    """Drop only the generated restricted database role."""
    role_name = _validate_role_name(role_name)
    connection = await asyncpg.connect(
        admin_url,
        timeout=10,
        command_timeout=10,
    )
    try:
        await connection.execute(
            f'DROP ROLE IF EXISTS "{role_name}"'  # noqa: S608
        )
    finally:
        await _close_connection(connection)


@pytest.fixture(scope="session")
def postgres_database_url() -> Iterator[str]:
    """Yield a least-privilege database URL and always remove its resources."""
    admin_url = os.environ.get(_ADMIN_URL_ENV)
    if not admin_url:
        if os.environ.get("GITHUB_ACTIONS", "").lower() == "true":
            pytest.fail(
                f"{_ADMIN_URL_ENV} is required in GitHub Actions",
                pytrace=False,
            )
        pytest.skip(f"Set {_ADMIN_URL_ENV} to run PostgreSQL integration tests")

    database_name, role_name, role_password = _new_database_identity()
    try:
        test_database_url = _database_url(
            admin_url,
            database_name,
            role_name,
            role_password,
        )
    except Exception:
        pytest.fail(
            "The PostgreSQL test admin URL is invalid",
            pytrace=False,
        )

    cleanup_failed = False
    try:
        try:
            asyncio.run(_create_role(admin_url, role_name, role_password))
            asyncio.run(_create_database(admin_url, database_name, role_name))
            if not asyncio.run(_role_is_restricted(admin_url, role_name)):
                msg = "Generated PostgreSQL test role is overprivileged"
                raise RuntimeError(msg)
        except Exception:
            pytest.fail(
                "Unable to provision isolated PostgreSQL test resources",
                pytrace=False,
            )

        yield test_database_url
    finally:
        try:
            asyncio.run(_drop_database(admin_url, database_name))
        except Exception:
            cleanup_failed = True
        try:
            asyncio.run(_drop_role(admin_url, role_name))
        except Exception:
            cleanup_failed = True
        if cleanup_failed:
            pytest.fail(
                "Unable to remove isolated PostgreSQL test resources",
                pytrace=False,
            )


def _phase_environment(phase_root: Path, database_url: str) -> dict[str, str]:
    """Build a minimal child environment without inherited credentials."""
    agent_dir = phase_root / "agents"
    home_dir = phase_root / "home"
    temp_dir = phase_root / "tmp"
    for directory in (agent_dir, home_dir, temp_dir):
        directory.mkdir(parents=True)

    return {
        "ADK_DISABLE_LOAD_DOTENV": "true",
        "ADK_DISABLE_LOCAL_STORAGE": "true",
        "AGENT_DIR": str(agent_dir),
        "AGENT_NAME": "postgres-integration-agent",
        "ALLOW_ORIGINS": "[]",
        "DATABASE_URL": database_url,
        "DB_MAX_OVERFLOW": "0",
        "DB_POOL_SIZE": "1",
        "DB_POOL_TIMEOUT": "5",
        "HOME": str(home_dir),
        "LOG_LEVEL": "WARNING",
        "OTEL_SDK_DISABLED": "true",
        "PATH": os.environ.get("PATH", ""),
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(_SOURCE_ROOT),
        "PYTHONUTF8": "1",
        "RELOAD_AGENTS": "false",
        "SERVE_WEB_INTERFACE": "false",
        "TMPDIR": str(temp_dir),
    }


def _bounded_diagnostics(output: str, database_url: str) -> str:
    """Structurally redact PostgreSQL credentials and bound failure output."""
    parsed = urlsplit(database_url)
    transformed_url = database_url.replace(
        "postgresql://",
        "postgresql+asyncpg://",
        1,
    )
    sensitive_values = {
        database_url,
        transformed_url,
        parsed.password or "",
        unquote(parsed.password or ""),
    }
    redacted = output
    for value in sorted(sensitive_values, key=len, reverse=True):
        if value:
            redacted = redacted.replace(value, "<redacted-database-value>")
    redacted = _POSTGRESQL_URL_PATTERN.sub(
        "<redacted-postgresql-url>",
        redacted,
    )
    return redacted[-4000:]


def _assert_local_storage_disabled(phase_root: Path) -> None:
    """Verify the ADK child created no local storage tree."""
    assert list(phase_root.rglob(".adk")) == []


def test_database_url_replaces_admin_user_information() -> None:
    """Give application children only the restricted database owner's login."""
    database_name = f"adk_test_{'0' * 32}"
    role_name = f"adk_role_{'0' * 32}"
    role_password = "a" * 64

    database_url = _database_url(
        "postgresql://cluster-admin:admin-canary@127.0.0.1:5432/postgres",
        database_name,
        role_name,
        role_password,
    )
    parsed = urlsplit(database_url)

    assert parsed.username == role_name
    assert parsed.password == role_password
    assert parsed.path == f"/{database_name}"
    assert parsed.query == ""
    assert "cluster-admin" not in database_url
    assert "admin-canary" not in database_url


@pytest.mark.parametrize(
    "override_key",
    [
        "channel_binding",
        "database",
        "dbname",
        "dsn",
        "gsslib",
        "host",
        "krbsrvname",
        "options",
        "passfile",
        "password",
        "port",
        "service",
        "servicefile",
        "server_settings",
        "ssl",
        "sslcert",
        "sslkey",
        "sslmode",
        "sslpassword",
        "sslrootcert",
        "target_session_attrs",
        "user",
        "username",
    ],
)
def test_database_url_rejects_query_identity_overrides(
    override_key: str,
) -> None:
    """Prevent admin query parameters from replacing the restricted owner."""
    database_name = f"adk_test_{'0' * 32}"
    role_name = f"adk_role_{'0' * 32}"

    with pytest.raises(ValueError) as error:
        _database_url(
            "postgresql://cluster-admin:admin-canary@127.0.0.1/postgres"
            f"?{override_key}=query-canary",
            database_name,
            role_name,
            "a" * 64,
        )

    assert override_key in str(error.value)
    assert "query-canary" not in str(error.value)


def test_phase_environment_disables_external_sources(tmp_path: Path) -> None:
    """Keep child processes independent from local files and provider secrets."""
    environment = _phase_environment(
        tmp_path / "phase",
        "postgresql://role:synthetic@127.0.0.1/database",
    )

    assert environment["ADK_DISABLE_LOAD_DOTENV"] == "true"
    assert environment["ADK_DISABLE_LOCAL_STORAGE"] == "true"
    assert environment["OTEL_SDK_DISABLED"] == "true"
    assert {
        "GOOGLE_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "OPENROUTER_API_KEY",
        "TEST_POSTGRES_ADMIN_URL",
    }.isdisjoint(environment)


def _run_phase(
    phase_root: Path,
    database_url: str,
    source: str,
) -> dict[str, Any]:
    """Run one fresh server process and return its structured result."""
    phase_root.mkdir()
    try:
        completed = subprocess.run(  # noqa: S603
            [sys.executable, "-c", textwrap.dedent(source)],
            cwd=phase_root,
            env=_phase_environment(phase_root, database_url),
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as error:
        timeout_stdout = (
            error.stdout.decode(errors="replace")
            if isinstance(error.stdout, bytes)
            else error.stdout or ""
        )
        timeout_stderr = (
            error.stderr.decode(errors="replace")
            if isinstance(error.stderr, bytes)
            else error.stderr or ""
        )
        stdout = _bounded_diagnostics(timeout_stdout, database_url)
        stderr = _bounded_diagnostics(timeout_stderr, database_url)
        pytest.fail(
            "PostgreSQL server subprocess timed out"
            f"\nstdout:\n{stdout}\nstderr:\n{stderr}",
            pytrace=False,
        )
    if completed.returncode != 0:
        stdout = _bounded_diagnostics(completed.stdout, database_url)
        stderr = _bounded_diagnostics(completed.stderr, database_url)
        pytest.fail(
            "PostgreSQL server subprocess failed"
            f"\nstdout:\n{stdout}\nstderr:\n{stderr}",
            pytrace=False,
        )

    result_line = next(
        (
            line
            for line in reversed(completed.stdout.splitlines())
            if line.startswith(_RESULT_PREFIX)
        ),
        None,
    )
    if result_line is None:
        pytest.fail(
            "PostgreSQL server subprocess returned no structured result",
            pytrace=False,
        )
    return cast(
        dict[str, Any],
        json.loads(result_line.removeprefix(_RESULT_PREFIX)),
    )


async def _read_persisted_row(database_url: str) -> tuple[int, str | None]:
    """Read the exact physical row count and state independently of ADK."""
    connection = await asyncpg.connect(
        database_url,
        timeout=10,
        command_timeout=10,
    )
    try:
        record = await connection.fetchrow(
            """
            SELECT count(*)::integer AS row_count,
                   max(state ->> 'persisted') AS persisted_value
            FROM sessions
            WHERE app_name = $1
              AND user_id = $2
              AND id = $3
            """,
            _APP_NAME,
            _USER_ID,
            _SESSION_ID,
        )
        if record is None:
            msg = "PostgreSQL aggregate query returned no record"
            raise RuntimeError(msg)
        return int(record["row_count"]), cast(str | None, record["persisted_value"])
    finally:
        await _close_connection(connection)


def _inspect_persisted_row(database_url: str) -> tuple[int, str | None]:
    """Run the independent database assertion without exposing its URL."""
    try:
        return asyncio.run(_read_persisted_row(database_url))
    except Exception:
        pytest.fail(
            "Unable to inspect the isolated PostgreSQL session row",
            pytrace=False,
        )


@pytest.mark.parametrize(
    ("diagnostic", "secret_canary"),
    [
        (
            "postgresql://role:encoded%40secret@127.0.0.1/database",
            "encoded%40secret",
        ),
        (
            "postgresql+asyncpg://role:encoded%40secret@127.0.0.1/database",
            "encoded%40secret",
        ),
        ("connection password was encoded@secret", "encoded@secret"),
        (
            "other=postgresql://other:other-secret@127.0.0.1/database",
            "other-secret",
        ),
    ],
)
def test_bounded_diagnostics_redacts_postgres_credentials(
    diagnostic: str,
    secret_canary: str,
) -> None:
    database_url = "postgresql://role:encoded%40secret@127.0.0.1/database"

    redacted = _bounded_diagnostics(diagnostic, database_url)

    assert secret_canary not in redacted
    assert len(redacted) <= 4000


def test_session_persists_across_server_processes(
    postgres_database_url: str,
    tmp_path: Path,
) -> None:
    """Prove the HTTP API persists and deletes a session in real PostgreSQL."""
    create_root = tmp_path / "create-process"
    create_result = _run_phase(
        create_root,
        postgres_database_url,
        _CREATE_PHASE,
    )

    _assert_local_storage_disabled(create_root)
    assert create_result["created_status"] == 200
    assert create_result["duplicate_status"] == 409
    assert create_result["created_body"]["id"] == _SESSION_ID
    assert create_result["created_body"]["state"] == {"persisted": _PERSISTED_VALUE}
    assert _inspect_persisted_row(postgres_database_url) == (1, _PERSISTED_VALUE)

    restart_root = tmp_path / "restart-process"
    restart_result = _run_phase(
        restart_root,
        postgres_database_url,
        _RESTART_PHASE,
    )

    _assert_local_storage_disabled(restart_root)
    assert restart_result["fetched_status"] == 200
    assert restart_result["fetched_body"]["id"] == _SESSION_ID
    assert restart_result["fetched_body"]["state"] == {"persisted": _PERSISTED_VALUE}
    assert restart_result["list_status"] == 200
    assert restart_result["list_body"] == [restart_result["fetched_body"]]
    assert restart_result["delete_status"] == 200
    assert restart_result["missing_status"] == 404
    assert _inspect_persisted_row(postgres_database_url) == (0, None)
