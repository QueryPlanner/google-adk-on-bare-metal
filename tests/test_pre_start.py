"""Database readiness and PID 1 entrypoint contract tests."""

from __future__ import annotations

import asyncio
import errno
import os
import runpy
import select
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, call, create_autospec
from urllib.parse import parse_qsl, urlsplit

import asyncpg  # type: ignore[import-untyped]
import pytest
from pydantic import ValidationError

from agent import pre_start
from agent.utils import DatabaseReadinessEnv, ServerEnv, SettingsConfigurationError

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
ENTRYPOINT_PATH = REPOSITORY_ROOT / "entrypoint.sh"
DATABASE_URL = (
    "postgresql://synthetic-user:encoded%40password@[2001:db8::1]:5432/"
    "agent%2Fdatabase?target_session_attrs=any"
)
READINESS_ENVIRONMENT_KEYS = (
    "DATABASE_URL",
    "DB_READY_TIMEOUT",
    "DB_READY_RETRY_INTERVAL",
    "DB_READY_ATTEMPT_TIMEOUT",
)


class _ExecRequested(BaseException):
    """Stop a unit test where a real process replacement would occur."""


def _mock_connection(
    *,
    query_result: int = 1,
) -> Any:
    """Return a strict asyncpg connection double."""
    connection = create_autospec(
        asyncpg.Connection,
        instance=True,
        spec_set=True,
    )
    connection.fetchval.return_value = query_result
    return connection


def _mock_connect(
    monkeypatch: pytest.MonkeyPatch,
    *,
    connection: Any | None = None,
    side_effect: Any = None,
) -> Any:
    """Replace only the asyncpg network boundary with a strict double."""
    connect = create_autospec(asyncpg.connect, spec_set=True)
    if side_effect is not None:
        connect.side_effect = side_effect
    else:
        connect.return_value = connection or _mock_connection()
    monkeypatch.setattr(pre_start.asyncpg, "connect", connect)
    return connect


def _mock_execvp(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace only the operating-system process boundary."""
    execvp = cast(MagicMock, create_autospec(os.execvp, spec_set=True))
    execvp.side_effect = _ExecRequested
    monkeypatch.setattr(pre_start.os, "execvp", execvp)
    return execvp


def _clear_readiness_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove only environment variables consumed by the readiness process."""
    for key in READINESS_ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)


def _subprocess_environment(
    tmp_path: Path,
    **overrides: str,
) -> dict[str, str]:
    """Build a minimal child environment without inherited credentials."""
    home_dir = tmp_path / "home"
    temp_dir = tmp_path / "tmp"
    home_dir.mkdir()
    temp_dir.mkdir()
    environment = {
        "HOME": str(home_dir),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(SOURCE_ROOT),
        "PYTHONUNBUFFERED": "1",
        "TMPDIR": str(temp_dir),
    }
    environment.update(overrides)
    return environment


@pytest.mark.parametrize(
    ("database_url", "expected"),
    [
        (
            "postgresql://user:encoded%40pass@[2001:db8::1]:5432/db%2Fname"
            "?target_session_attrs=any",
            "postgresql://user:encoded%40pass@[2001:db8::1]:5432/db%2Fname"
            "?target_session_attrs=any",
        ),
        (
            "postgres://user:pass@localhost/database",
            "postgres://user:pass@localhost/database",
        ),
        (
            "postgresql+asyncpg://user:pass@localhost/database",
            "postgresql://user:pass@localhost/database",
        ),
        (
            "postgresql:///database?host=%2Fvar%2Frun%2Fpostgresql"
            "&port=5432&target_session_attrs=any",
            "postgresql:///database?host=%2Fvar%2Frun%2Fpostgresql"
            "&port=5432&target_session_attrs=any",
        ),
        (
            "postgresql://user:encoded%40pass@/database"
            "?host=%2Fvar%2Frun%2Fpostgresql&port=5432",
            "postgresql://user:encoded%40pass@/database"
            "?host=%2Fvar%2Frun%2Fpostgresql&port=5432",
        ),
        (
            "postgresql://user:pass@localhost/database"
            "?port=5432&target_session_attrs=any",
            "postgresql://user:pass@localhost/database"
            "?port=5432&target_session_attrs=any",
        ),
    ],
)
def test_normalize_database_url_preserves_connection_identity(
    database_url: str,
    expected: str,
) -> None:
    """Normalize only the driver suffix, not connection identity details."""
    assert pre_start.normalize_database_url(database_url) == expected


def test_normalize_database_url_removes_required_channel_binding_only() -> None:
    database_url = (
        "postgresql://user:pass@localhost/database"
        "?target_session_attrs=any&channel_binding=require&krbsrvname=postgres"
    )

    normalized = pre_start.normalize_database_url(database_url)

    assert parse_qsl(urlsplit(normalized).query, keep_blank_values=True) == [
        ("target_session_attrs", "any"),
        ("krbsrvname", "postgres"),
    ]


def test_normalize_database_url_maps_ssl_for_direct_asyncpg() -> None:
    database_url = (
        "postgresql://user:pass@localhost/database?ssl=require&target_session_attrs=any"
    )

    normalized = pre_start.normalize_database_url(database_url)

    assert parse_qsl(urlsplit(normalized).query, keep_blank_values=True) == [
        ("sslmode", "require"),
        ("target_session_attrs", "any"),
    ]


def test_normalize_database_url_preserves_sslmode() -> None:
    database_url = (
        "postgresql://user:pass@localhost/database"
        "?sslmode=verify-full&target_session_attrs=any"
    )

    assert pre_start.normalize_database_url(database_url) == database_url


def test_server_database_url_preserves_sqlalchemy_ssl_parameter() -> None:
    database_url = (
        "postgresql://user:pass@localhost/database"
        "?ssl=require&channel_binding=require&target_session_attrs=any"
    )

    environment = ServerEnv.model_validate(
        {
            "AGENT_NAME": "test-agent",
            "DATABASE_URL": database_url,
        }
    )

    assert environment.session_uri == (
        "postgresql://user:pass@localhost/database?ssl=require&target_session_attrs=any"
    )


def test_server_database_url_normalizes_postgres_scheme_for_sqlalchemy() -> None:
    environment = ServerEnv.model_validate(
        {
            "AGENT_NAME": "test-agent",
            "DATABASE_URL": "postgres://user:pass@localhost/database",
        }
    )

    assert environment.session_uri == ("postgresql://user:pass@localhost/database")


@pytest.mark.parametrize(
    "database_url",
    [
        "   ",
        "mysql://synthetic-user:synthetic-password@localhost/database",
        "postgresql:/database",
        "postgresql://user:pass@[broken/database",
        "postgresql://user:pass@localhost/database#fragment",
        "postgresql://user:pass@localhost/database?channel_binding=prefer",
        "postgresql://user:pass@localhost/database?sslmode",
        "postgresql://user:pass@localhost/database?sslmode=require&sslmode=disable",
        "postgresql://user:pass@localhost/database?ssl=require&sslmode=require",
        "postgresql://user:pass@localhost/database?application_name=worker",
        (
            "postgresql://user:pass@localhost/database"
            "?host=%2Fvar%2Frun%2Fpostgresql&port=5432"
        ),
        "postgresql://user:pass@localhost:5432/database?port=6543",
        "postgresql://user:pass@localhost:not-a-port/database",
    ],
)
def test_normalize_database_url_rejects_unsupported_configuration_safely(
    database_url: str,
) -> None:
    """Reject ambiguous or unsupported inputs without echoing their values."""
    with pytest.raises(ValueError) as error:
        pre_start.normalize_database_url(database_url)

    rendered_error = str(error.value)
    assert database_url not in rendered_error
    assert "synthetic-password" not in rendered_error


def test_database_readiness_settings_use_bounded_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_readiness_environment(monkeypatch)

    environment = DatabaseReadinessEnv()

    assert environment.database_url is None
    assert environment.db_ready_timeout == 60
    assert environment.retry_interval == 1
    assert environment.attempt_timeout == 5


def test_database_readiness_settings_accept_documented_upper_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_readiness_environment(monkeypatch)
    environment = DatabaseReadinessEnv.model_validate(
        {
            "DATABASE_URL": "postgresql://user:pass@localhost/database",
            "DB_READY_TIMEOUT": 3600,
            "DB_READY_RETRY_INTERVAL": 60,
            "DB_READY_ATTEMPT_TIMEOUT": 60,
        }
    )

    assert environment.database_url is not None
    assert (
        environment.database_url.get_secret_value()
        == "postgresql://user:pass@localhost/database"
    )
    assert environment.db_ready_timeout == 3600
    assert environment.retry_interval == 60
    assert environment.attempt_timeout == 60


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("DB_READY_TIMEOUT", 0),
        ("DB_READY_TIMEOUT", 3600.1),
        ("DB_READY_RETRY_INTERVAL", 0),
        ("DB_READY_RETRY_INTERVAL", 60.1),
        ("DB_READY_ATTEMPT_TIMEOUT", 0),
        ("DB_READY_ATTEMPT_TIMEOUT", 60.1),
    ],
)
def test_database_readiness_settings_reject_out_of_bounds_values(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    value: float,
) -> None:
    _clear_readiness_environment(monkeypatch)

    with pytest.raises(ValidationError):
        DatabaseReadinessEnv.model_validate({field_name: value})


def test_database_readiness_settings_normalize_blank_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_readiness_environment(monkeypatch)

    environment = DatabaseReadinessEnv.model_validate({"DATABASE_URL": "   "})

    assert environment.database_url is None


def test_database_readiness_settings_do_not_read_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_readiness_environment(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "DATABASE_URL=postgresql://dotenv-user:dotenv-pass@localhost/database\n"
        "DB_READY_TIMEOUT=100\n",
        encoding="utf-8",
    )

    environment = DatabaseReadinessEnv()

    assert environment.database_url is None
    assert environment.db_ready_timeout == 60


@pytest.mark.parametrize(
    "database_url",
    [
        "changethis",
        "postgresql://user:pass@host:port/dbname?ssl=require",
    ],
)
def test_database_readiness_settings_reject_example_database_urls(
    monkeypatch: pytest.MonkeyPatch,
    database_url: str,
) -> None:
    _clear_readiness_environment(monkeypatch)

    with pytest.raises(SettingsConfigurationError) as error:
        DatabaseReadinessEnv.model_validate({"DATABASE_URL": database_url})

    assert database_url not in str(error.value)


def test_database_readiness_validation_errors_redact_all_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_readiness_environment(monkeypatch)
    database_url = (
        "postgresql://private-user:encoded%40password@localhost/private-database"
    )
    invalid_timeout = "not-a-number-secret-canary"

    with pytest.raises(ValidationError) as error:
        DatabaseReadinessEnv.model_validate(
            {
                "DATABASE_URL": database_url,
                "DB_READY_TIMEOUT": invalid_timeout,
            }
        )

    rendered_error = repr(error.value.errors(include_input=True))
    assert database_url not in rendered_error
    assert invalid_timeout not in rendered_error
    assert "private-user" not in rendered_error
    assert "encoded%40password" not in rendered_error


@pytest.mark.asyncio
async def test_check_database_runs_exact_probe_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _mock_connection()
    connect = _mock_connect(monkeypatch, connection=connection)

    await pre_start.check_database(DATABASE_URL, attempt_timeout=0.5)

    connect.assert_awaited_once_with(
        DATABASE_URL,
        timeout=0.5,
        command_timeout=0.5,
    )
    connection.fetchval.assert_awaited_once_with("SELECT 1", timeout=0.5)
    connection.close.assert_awaited_once_with(timeout=0.5)
    connection.terminate.assert_not_called()


@pytest.mark.asyncio
async def test_check_database_rejects_unexpected_probe_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _mock_connection(query_result=0)
    _mock_connect(monkeypatch, connection=connection)

    with pytest.raises(RuntimeError, match="readiness query"):
        await pre_start.check_database(DATABASE_URL, attempt_timeout=0.5)

    connection.close.assert_not_awaited()
    connection.terminate.assert_called_once_with()


@pytest.mark.asyncio
async def test_check_database_terminates_after_query_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _mock_connection()
    connection.fetchval.side_effect = asyncpg.PostgresConnectionError(
        "synthetic query failure"
    )
    _mock_connect(monkeypatch, connection=connection)

    with pytest.raises(
        asyncpg.PostgresConnectionError,
        match="synthetic query failure",
    ):
        await pre_start.check_database(DATABASE_URL, attempt_timeout=0.5)

    connection.close.assert_not_awaited()
    connection.terminate.assert_called_once_with()


@pytest.mark.asyncio
async def test_check_database_closes_when_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _mock_connection()
    connection.fetchval.side_effect = asyncio.CancelledError
    _mock_connect(monkeypatch, connection=connection)

    with pytest.raises(asyncio.CancelledError):
        await pre_start.check_database(DATABASE_URL, attempt_timeout=0.5)

    connection.close.assert_not_awaited()
    connection.terminate.assert_called_once_with()


@pytest.mark.asyncio
async def test_check_database_bounds_a_hung_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _mock_connection()

    async def never_return(*_args: Any, **_kwargs: Any) -> None:
        await asyncio.Event().wait()

    connection.fetchval.side_effect = never_return
    _mock_connect(monkeypatch, connection=connection)

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        await pre_start.check_database(DATABASE_URL, attempt_timeout=0.05)
    elapsed = time.monotonic() - started

    assert elapsed < 1
    connection.close.assert_not_awaited()
    connection.terminate.assert_called_once_with()


@pytest.mark.asyncio
async def test_check_database_bounds_a_hung_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _mock_connection()

    async def never_return(*_args: Any, **_kwargs: Any) -> None:
        await asyncio.Event().wait()

    connection.close.side_effect = never_return
    _mock_connect(monkeypatch, connection=connection)

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        await pre_start.check_database(DATABASE_URL, attempt_timeout=0.05)
    elapsed = time.monotonic() - started

    assert elapsed < 1
    connection.close.assert_awaited_once_with(timeout=0.05)
    connection.terminate.assert_called_once_with()


@pytest.mark.asyncio
async def test_check_database_terminates_when_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _mock_connection()
    connection.close.side_effect = OSError("synthetic close failure")
    _mock_connect(monkeypatch, connection=connection)

    with pytest.raises(OSError, match="synthetic close failure"):
        await pre_start.check_database(DATABASE_URL, attempt_timeout=0.5)

    connection.close.assert_awaited_once_with(timeout=0.5)
    connection.terminate.assert_called_once_with()


@pytest.mark.asyncio
async def test_check_database_terminates_when_close_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _mock_connection()
    connection.close.side_effect = asyncio.CancelledError
    _mock_connect(monkeypatch, connection=connection)

    with pytest.raises(asyncio.CancelledError):
        await pre_start.check_database(DATABASE_URL, attempt_timeout=0.5)

    connection.close.assert_awaited_once_with(timeout=0.5)
    connection.terminate.assert_called_once_with()


@pytest.mark.parametrize(
    ("invalid_field", "invalid_value"),
    [
        ("timeout", 0),
        ("timeout", float("inf")),
        ("retry_interval", 0),
        ("retry_interval", float("nan")),
        ("attempt_timeout", 0),
        ("attempt_timeout", float("inf")),
    ],
)
@pytest.mark.asyncio
async def test_wait_for_database_rejects_invalid_timing(
    monkeypatch: pytest.MonkeyPatch,
    invalid_field: str,
    invalid_value: float,
) -> None:
    connect = _mock_connect(monkeypatch)
    timing = {
        "timeout": 1.0,
        "retry_interval": 0.01,
        "attempt_timeout": 0.1,
    }
    timing[invalid_field] = invalid_value

    with pytest.raises(ValueError, match=invalid_field):
        await pre_start.wait_for_database(
            DATABASE_URL,
            timeout=timing["timeout"],
            retry_interval=timing["retry_interval"],
            attempt_timeout=timing["attempt_timeout"],
        )

    connect.assert_not_called()


@pytest.mark.parametrize(
    "transient_error",
    [
        ConnectionRefusedError("synthetic"),
        TimeoutError("synthetic"),
        OSError("synthetic aggregate network failure"),
        OSError(errno.ENETUNREACH, "synthetic"),
        asyncpg.PostgresConnectionError("synthetic"),
        asyncpg.CannotConnectNowError("synthetic"),
        asyncpg.AdminShutdownError("synthetic"),
        asyncpg.CrashShutdownError("synthetic"),
        asyncpg.TooManyConnectionsError("synthetic"),
    ],
)
@pytest.mark.asyncio
async def test_wait_for_database_retries_only_transient_failures(
    monkeypatch: pytest.MonkeyPatch,
    transient_error: BaseException,
) -> None:
    connection = _mock_connection()
    connect = _mock_connect(
        monkeypatch,
        side_effect=[transient_error, connection],
    )

    attempts = await pre_start.wait_for_database(
        DATABASE_URL,
        timeout=0.5,
        retry_interval=0.001,
        attempt_timeout=0.1,
    )

    assert attempts == 2
    assert connect.await_count == 2
    connection.fetchval.assert_awaited_once_with("SELECT 1", timeout=0.1)


@pytest.mark.asyncio
async def test_wait_for_database_exhaustion_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connect = _mock_connect(
        monkeypatch,
        side_effect=ConnectionRefusedError("synthetic transient failure"),
    )

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        await pre_start.wait_for_database(
            DATABASE_URL,
            timeout=0.03,
            retry_interval=0.01,
            attempt_timeout=0.01,
        )
    elapsed = time.monotonic() - started

    assert 2 <= connect.await_count < 20
    assert elapsed < 0.5


@pytest.mark.asyncio
async def test_wait_for_database_bounds_retry_sleep_by_total_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connect = _mock_connect(
        monkeypatch,
        side_effect=ConnectionRefusedError("synthetic transient failure"),
    )

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        await pre_start.wait_for_database(
            DATABASE_URL,
            timeout=0.02,
            retry_interval=0.5,
            attempt_timeout=0.01,
        )
    elapsed = time.monotonic() - started

    assert connect.await_count == 1
    assert elapsed < 0.25


@pytest.mark.parametrize(
    "permanent_error",
    [
        asyncpg.InvalidPasswordError("synthetic"),
        asyncpg.InvalidCatalogNameError("synthetic"),
        FileNotFoundError("synthetic"),
        PermissionError("synthetic"),
    ],
)
@pytest.mark.asyncio
async def test_wait_for_database_fails_fast_on_permanent_errors(
    monkeypatch: pytest.MonkeyPatch,
    permanent_error: Exception,
) -> None:
    connect = _mock_connect(monkeypatch, side_effect=permanent_error)

    with pytest.raises(type(permanent_error), match="synthetic"):
        await pre_start.wait_for_database(
            DATABASE_URL,
            timeout=0.5,
            retry_interval=0.01,
            attempt_timeout=0.1,
        )

    assert connect.await_count == 1


@pytest.mark.asyncio
async def test_wait_for_database_does_not_retry_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connect = _mock_connect(monkeypatch, side_effect=asyncio.CancelledError)

    with pytest.raises(asyncio.CancelledError):
        await pre_start.wait_for_database(
            DATABASE_URL,
            timeout=0.5,
            retry_interval=0.01,
            attempt_timeout=0.1,
        )

    assert connect.await_count == 1


@pytest.mark.parametrize("termination_signal", [signal.SIGTERM, signal.SIGINT])
def test_termination_signal_handler_uses_conventional_exit_status(
    termination_signal: signal.Signals,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        pre_start._exit_on_termination_signal(termination_signal, None)

    assert error.value.code == 128 + termination_signal
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_main_installs_and_restores_termination_signal_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_readiness_environment(monkeypatch)
    monkeypatch.setattr(
        pre_start.sys,
        "argv",
        ["agent.pre_start", "synthetic-command"],
    )
    execvp = _mock_execvp(monkeypatch)
    previous_handlers = {
        signal.SIGTERM: signal.SIG_DFL,
        signal.SIGINT: signal.SIG_IGN,
    }
    set_signal_handler = create_autospec(signal.signal, spec_set=True)

    def remember_previous_handler(
        termination_signal: signal.Signals,
        handler: Any,
    ) -> Any:
        if handler is pre_start._exit_on_termination_signal:
            return previous_handlers[termination_signal]
        return signal.SIG_DFL

    set_signal_handler.side_effect = remember_previous_handler
    monkeypatch.setattr(pre_start.signal, "signal", set_signal_handler)

    with pytest.raises(_ExecRequested):
        pre_start.main()

    execvp.assert_called_once_with("synthetic-command", ["synthetic-command"])
    assert set_signal_handler.call_count == 4
    set_signal_handler.assert_has_calls(
        [
            call(signal.SIGTERM, pre_start._exit_on_termination_signal),
            call(signal.SIGINT, pre_start._exit_on_termination_signal),
            call(signal.SIGTERM, signal.SIG_DFL),
            call(signal.SIGINT, signal.SIG_IGN),
        ],
        any_order=True,
    )


@pytest.mark.parametrize("database_url", [None, "   "])
def test_main_executes_command_without_configured_database(
    monkeypatch: pytest.MonkeyPatch,
    database_url: str | None,
) -> None:
    _clear_readiness_environment(monkeypatch)
    if database_url is not None:
        monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setattr(
        pre_start.sys,
        "argv",
        ["agent.pre_start", "synthetic-command", "--flag"],
    )
    execvp = _mock_execvp(monkeypatch)

    with pytest.raises(_ExecRequested):
        pre_start.main()

    execvp.assert_called_once_with(
        "synthetic-command",
        ["synthetic-command", "--flag"],
    )


def test_main_executes_command_only_after_database_is_ready(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_readiness_environment(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("DB_READY_TIMEOUT", "0.5")
    monkeypatch.setenv("DB_READY_RETRY_INTERVAL", "0.01")
    monkeypatch.setenv("DB_READY_ATTEMPT_TIMEOUT", "0.1")
    monkeypatch.setattr(
        pre_start.sys,
        "argv",
        ["agent.pre_start", "synthetic-command", "--flag"],
    )
    connection = _mock_connection()
    connect = _mock_connect(monkeypatch, connection=connection)
    execvp = _mock_execvp(monkeypatch)

    with pytest.raises(_ExecRequested):
        pre_start.main()

    assert connect.await_count == 1
    execvp.assert_called_once_with(
        "synthetic-command",
        ["synthetic-command", "--flag"],
    )
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert DATABASE_URL not in output
    assert "encoded%40password" not in output
    assert "encoded@password" not in output


@pytest.mark.parametrize(
    ("environment_key", "environment_value"),
    [
        (
            "DATABASE_URL",
            "mysql://private-user:encoded%40password@localhost/private",
        ),
        ("DB_READY_TIMEOUT", "not-a-number-secret-canary"),
    ],
)
def test_main_fails_closed_without_disclosing_configuration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    environment_key: str,
    environment_value: str,
) -> None:
    _clear_readiness_environment(monkeypatch)
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://private-user:encoded%40password@localhost/private",
    )
    monkeypatch.setenv(environment_key, environment_value)
    monkeypatch.setattr(
        pre_start.sys,
        "argv",
        ["agent.pre_start", "synthetic-command"],
    )
    execvp = _mock_execvp(monkeypatch)

    with pytest.raises(SystemExit) as error:
        pre_start.main()

    assert error.value.code != 0
    execvp.assert_not_called()
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert environment_value not in output
    assert "private-user" not in output
    assert "encoded%40password" not in output
    assert "encoded@password" not in output


def test_main_requires_a_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_readiness_environment(monkeypatch)
    monkeypatch.setattr(pre_start.sys, "argv", ["agent.pre_start"])
    execvp = _mock_execvp(monkeypatch)

    with pytest.raises(SystemExit) as error:
        pre_start.main()

    assert error.value.code != 0
    execvp.assert_not_called()


def test_main_reports_application_exec_failure_safely(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_readiness_environment(monkeypatch)
    monkeypatch.setattr(
        pre_start.sys,
        "argv",
        ["agent.pre_start", "private-command-canary"],
    )
    execvp = create_autospec(os.execvp, spec_set=True)
    execvp.side_effect = OSError("private-command-canary could not execute")
    monkeypatch.setattr(pre_start.os, "execvp", execvp)

    with pytest.raises(SystemExit) as error:
        pre_start.main()

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert error.value.code != 0
    execvp.assert_called_once_with(
        "private-command-canary",
        ["private-command-canary"],
    )
    assert "private-command-canary" not in output


def test_module_execution_invokes_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_readiness_environment(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["agent.pre_start", "synthetic-command"],
    )
    execvp = create_autospec(os.execvp, spec_set=True)
    execvp.side_effect = _ExecRequested
    monkeypatch.setattr(os, "execvp", execvp)
    monkeypatch.delitem(sys.modules, "agent.pre_start")

    with pytest.raises(_ExecRequested):
        runpy.run_module("agent.pre_start", run_name="__main__")

    execvp.assert_called_once_with(
        "synthetic-command",
        ["synthetic-command"],
    )


def test_entrypoint_is_valid_posix_shell() -> None:
    completed = subprocess.run(  # noqa: S603
        ["/bin/sh", "-n", str(ENTRYPOINT_PATH)],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert completed.returncode == 0, completed.stderr


def test_entrypoint_delegates_readiness_without_parsing_database_url() -> None:
    entrypoint = ENTRYPOINT_PATH.read_text(encoding="utf-8")

    assert 'exec "$python_bin" -m agent.pre_start "$@"' in entrypoint
    assert "DATABASE_URL" not in entrypoint
    assert "grep" not in entrypoint
    assert "sed" not in entrypoint
    assert "nc " not in entrypoint
    assert "while " not in entrypoint


def test_entrypoint_forwards_command_output_and_exit_code(tmp_path: Path) -> None:
    completed = subprocess.run(  # noqa: S603
        [
            "/bin/sh",
            str(ENTRYPOINT_PATH),
            sys.executable,
            "-c",
            "print('ENTRYPOINT_COMMAND_RAN'); raise SystemExit(23)",
        ],
        cwd=tmp_path,
        env=_subprocess_environment(tmp_path),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 23
    assert "ENTRYPOINT_COMMAND_RAN" in completed.stdout


def test_entrypoint_readiness_failure_blocks_command_and_redacts(
    tmp_path: Path,
) -> None:
    marker_path = tmp_path / "command-ran"
    database_url = "mysql://private-user:encoded%40password@localhost/private-database"
    completed = subprocess.run(  # noqa: S603
        [
            "/bin/sh",
            str(ENTRYPOINT_PATH),
            sys.executable,
            "-c",
            "from pathlib import Path; Path(__import__('sys').argv[1]).touch()",
            str(marker_path),
        ],
        cwd=tmp_path,
        env=_subprocess_environment(tmp_path, DATABASE_URL=database_url),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert not marker_path.exists()
    assert database_url not in output
    assert "private-user" not in output
    assert "encoded%40password" not in output
    assert "encoded@password" not in output


def test_entrypoint_term_during_readiness_exits_promptly(tmp_path: Path) -> None:
    marker_path = tmp_path / "command-ran"
    process = subprocess.Popen(  # noqa: S603
        [
            "/bin/sh",
            str(ENTRYPOINT_PATH),
            sys.executable,
            "-c",
            "from pathlib import Path; Path(__import__('sys').argv[1]).touch()",
            str(marker_path),
        ],
        cwd=tmp_path,
        env=_subprocess_environment(
            tmp_path,
            DATABASE_URL="postgresql://user:pass@127.0.0.1:1/database",
            DB_READY_TIMEOUT="30",
            DB_READY_RETRY_INTERVAL="0.1",
            DB_READY_ATTEMPT_TIMEOUT="0.2",
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        assert process.stdout is not None
        readiness_output: list[str] = []
        readiness_deadline = time.monotonic() + 5
        while time.monotonic() < readiness_deadline:
            readable, _, _ = select.select(
                [process.stdout],
                [],
                [],
                max(0, readiness_deadline - time.monotonic()),
            )
            if not readable:
                break
            line = process.stdout.readline()
            if not line:
                break
            readiness_output.append(line)
            if "Waiting for database readiness..." in line:
                break

        assert "Waiting for database readiness..." in "".join(readiness_output)
        assert process.poll() is None
        started = time.monotonic()
        process.terminate()
        process.communicate(timeout=5)
        elapsed = time.monotonic() - started
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)

    assert process.returncode == 128 + signal.SIGTERM
    assert elapsed < 3
    assert not marker_path.exists()
