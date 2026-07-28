"""Bounded PostgreSQL readiness gate and container process wrapper."""

from __future__ import annotations

import asyncio
import errno
import os
import signal
import socket
import sys
from types import FrameType
from typing import NoReturn

import asyncpg as asyncpg  # type: ignore[import-untyped]
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_delay,
    wait_fixed,
)

from .database import (
    _require_positive_finite,
    check_database,
    normalize_database_url,
)
from .utils import DatabaseReadinessEnv

_TRANSIENT_DATABASE_ERROR_TYPES = (
    ConnectionError,
    TimeoutError,
    socket.gaierror,
    asyncpg.PostgresConnectionError,
    asyncpg.CannotConnectNowError,
    asyncpg.AdminShutdownError,
    asyncpg.CrashShutdownError,
    asyncpg.TooManyConnectionsError,
)
_TRANSIENT_NETWORK_ERRNOS = frozenset(
    code
    for code in (
        errno.EADDRNOTAVAIL,
        errno.ECONNABORTED,
        errno.ECONNREFUSED,
        errno.ECONNRESET,
        getattr(errno, "EHOSTDOWN", None),
        errno.EHOSTUNREACH,
        errno.ENETDOWN,
        errno.ENETUNREACH,
        errno.ETIMEDOUT,
    )
    if code is not None
)


def _is_transient_database_error(error: BaseException) -> bool:
    """Return whether a failure can plausibly recover during VM startup."""
    if isinstance(error, _TRANSIENT_DATABASE_ERROR_TYPES):
        return True
    return type(error) is OSError and (
        error.errno is None or error.errno in _TRANSIENT_NETWORK_ERRNOS
    )


async def wait_for_database(
    database_url: str,
    *,
    timeout: float,
    retry_interval: float,
    attempt_timeout: float,
) -> int:
    """Wait for PostgreSQL readiness and return the number of attempts used."""
    _require_positive_finite("timeout", timeout)
    _require_positive_finite("retry_interval", retry_interval)
    _require_positive_finite("attempt_timeout", attempt_timeout)
    normalized_database_url = normalize_database_url(database_url)
    attempts = 0

    async with asyncio.timeout(timeout):
        async for attempt in AsyncRetrying(
            retry=retry_if_exception(_is_transient_database_error),
            wait=wait_fixed(retry_interval),
            stop=stop_after_delay(timeout),
            reraise=True,
        ):
            with attempt:
                attempts += 1
                await check_database(
                    normalized_database_url,
                    attempt_timeout=attempt_timeout,
                )

    return attempts


def _exit_on_termination_signal(
    signum: int,
    _frame: FrameType | None,
) -> NoReturn:
    """Exit promptly with the conventional status for a termination signal."""
    raise SystemExit(128 + signum)


def main() -> NoReturn:
    """Run readiness when configured, then replace this process with the command."""
    command = sys.argv[1:]
    if not command:
        print("No application command configured.", file=sys.stderr, flush=True)
        raise SystemExit(1)

    previous_sigterm = signal.signal(
        signal.SIGTERM,
        _exit_on_termination_signal,
    )
    try:
        previous_sigint = signal.signal(
            signal.SIGINT,
            _exit_on_termination_signal,
        )
        try:
            try:
                env = DatabaseReadinessEnv()
                if env.database_url is not None:
                    print("Waiting for database readiness...", flush=True)
                    asyncio.run(
                        wait_for_database(
                            env.database_url.get_secret_value(),
                            timeout=env.db_ready_timeout,
                            retry_interval=env.retry_interval,
                            attempt_timeout=env.attempt_timeout,
                        )
                    )
                    print("Database is ready.", flush=True)
            except Exception:
                print(
                    "Database readiness check failed.",
                    file=sys.stderr,
                    flush=True,
                )
                raise SystemExit(1) from None

            try:
                os.execvp(command[0], command)  # noqa: S606 - preserve container CMD
            except OSError:
                print(
                    "Application command could not start.",
                    file=sys.stderr,
                    flush=True,
                )
                raise SystemExit(1) from None
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
