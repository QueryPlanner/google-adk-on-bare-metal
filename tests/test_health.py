"""Process liveness and database readiness response contracts."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, cast
from unittest.mock import create_autospec

import asyncpg  # type: ignore[import-untyped]
import pytest
from pydantic import SecretStr

from agent import database
from agent.health import DatabaseReadinessProbe, live

DATABASE_URL = (
    "postgresql+asyncpg://private-user:encoded%40password@localhost/private-db"
    "?channel_binding=require&target_session_attrs=any"
)
NORMALIZED_DATABASE_URL = (
    "postgresql://private-user:encoded%40password@localhost/private-db"
    "?target_session_attrs=any"
)


def _connection(*, query_result: int = 1) -> Any:
    """Return a strict asyncpg connection boundary double."""
    connection = create_autospec(
        asyncpg.Connection,
        instance=True,
        spec_set=True,
    )
    connection.fetchval.return_value = query_result
    return connection


def _connect(
    monkeypatch: pytest.MonkeyPatch,
    *,
    connection: Any | None = None,
    side_effect: BaseException | None = None,
) -> Any:
    """Replace only the asyncpg connection boundary with a strict double."""
    connect = create_autospec(asyncpg.connect, spec_set=True)
    if side_effect is None:
        connect.return_value = connection or _connection()
    else:
        connect.side_effect = side_effect
    monkeypatch.setattr(database.asyncpg, "connect", connect)
    return connect


def _response_body(response: Any) -> dict[str, object]:
    """Decode one JSONResponse body for exact contract assertions."""
    return cast(dict[str, object], json.loads(response.body))


@pytest.mark.asyncio
async def test_live_reports_process_liveness() -> None:
    assert await live() == {"status": "alive"}


@pytest.mark.asyncio
async def test_readiness_without_database_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connect = _connect(monkeypatch)
    probe = DatabaseReadinessProbe(None, attempt_timeout=0.5)

    response = await probe()

    assert response.status_code == 200
    assert _response_body(response) == {
        "status": "ready",
        "checks": {"database": "not_configured"},
    }
    connect.assert_not_called()


@pytest.mark.asyncio
async def test_readiness_normalizes_checks_and_closes_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _connection()
    connect = _connect(monkeypatch, connection=connection)
    probe = DatabaseReadinessProbe(
        SecretStr(DATABASE_URL),
        attempt_timeout=0.5,
    )

    response = await probe()

    assert response.status_code == 200
    assert _response_body(response) == {
        "status": "ready",
        "checks": {"database": "healthy"},
    }
    connect.assert_awaited_once_with(
        NORMALIZED_DATABASE_URL,
        timeout=0.5,
        command_timeout=0.5,
    )
    connection.fetchval.assert_awaited_once_with("SELECT 1", timeout=0.5)
    connection.close.assert_awaited_once_with(timeout=0.5)
    connection.terminate.assert_not_called()
    assert isinstance(probe._database_url, SecretStr)
    assert DATABASE_URL not in repr(probe.__dict__)
    assert "private-user" not in repr(probe.__dict__)
    assert "encoded%40password" not in repr(probe.__dict__)


@pytest.mark.parametrize(
    ("failure_stage", "expected_termination"),
    [
        ("connection", False),
        ("query", True),
        ("unexpected_result", True),
        ("timeout", False),
        ("close", True),
        ("authentication", False),
    ],
)
@pytest.mark.asyncio
async def test_database_failures_map_to_one_secret_free_response(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure_stage: str,
    expected_termination: bool,
) -> None:
    failure_canary = "synthetic failure encoded%40password"
    connection = _connection(
        query_result=0 if failure_stage == "unexpected_result" else 1
    )
    connect_error: BaseException | None = None
    if failure_stage == "connection":
        connect_error = ConnectionError(failure_canary)
    elif failure_stage == "query":
        connection.fetchval.side_effect = OSError(failure_canary)
    elif failure_stage == "timeout":
        connect_error = TimeoutError(failure_canary)
    elif failure_stage == "close":
        connection.close.side_effect = OSError(failure_canary)
    elif failure_stage == "authentication":
        connect_error = asyncpg.InvalidPasswordError(failure_canary)

    connect = _connect(
        monkeypatch,
        connection=connection,
        side_effect=connect_error,
    )
    probe = DatabaseReadinessProbe(
        SecretStr(DATABASE_URL),
        attempt_timeout=0.5,
    )

    with caplog.at_level(logging.WARNING, logger="agent.health"):
        response = await probe()

    assert response.status_code == 503
    assert _response_body(response) == {
        "status": "not_ready",
        "checks": {"database": "unavailable"},
    }
    connect.assert_awaited_once_with(
        NORMALIZED_DATABASE_URL,
        timeout=0.5,
        command_timeout=0.5,
    )
    if expected_termination:
        connection.terminate.assert_called_once_with()
    else:
        connection.terminate.assert_not_called()

    records = [record for record in caplog.records if record.name == "agent.health"]
    assert len(records) == 1
    assert records[0].getMessage() == "Database readiness check failed."
    public_output = bytes(response.body).decode() + caplog.text
    assert DATABASE_URL not in public_output
    assert NORMALIZED_DATABASE_URL not in public_output
    assert "private-user" not in public_output
    assert "encoded%40password" not in public_output
    assert failure_canary not in public_output


@pytest.mark.asyncio
async def test_readiness_propagates_cancellation_without_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    connection = _connection()
    connection.fetchval.side_effect = asyncio.CancelledError
    connect = _connect(monkeypatch, connection=connection)
    probe = DatabaseReadinessProbe(
        SecretStr(DATABASE_URL),
        attempt_timeout=0.5,
    )

    with (
        caplog.at_level(logging.WARNING, logger="agent.health"),
        pytest.raises(asyncio.CancelledError),
    ):
        await probe()

    connect.assert_awaited_once()
    connection.terminate.assert_called_once_with()
    assert not [record for record in caplog.records if record.name == "agent.health"]


@pytest.mark.parametrize("attempt_timeout", [0, float("inf"), float("nan")])
def test_readiness_rejects_invalid_attempt_timeout(
    monkeypatch: pytest.MonkeyPatch,
    attempt_timeout: float,
) -> None:
    connect = _connect(monkeypatch)

    with pytest.raises(ValueError, match="attempt_timeout"):
        DatabaseReadinessProbe(None, attempt_timeout=attempt_timeout)

    connect.assert_not_called()
