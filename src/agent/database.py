"""Shared PostgreSQL connectivity checks."""

from __future__ import annotations

import asyncio
import math
from contextlib import suppress

import asyncpg  # type: ignore[import-untyped]

from .utils.config import _normalize_asyncpg_database_url


def _require_positive_finite(name: str, value: float) -> None:
    """Reject invalid timing without including the supplied value."""
    if not math.isfinite(value) or value <= 0:
        msg = f"{name} must be a positive finite number"
        raise ValueError(msg)


def normalize_database_url(database_url: str) -> str:
    """Normalize a supported PostgreSQL URL for direct asyncpg use."""
    return _normalize_asyncpg_database_url(database_url)


async def check_database(
    database_url: str,
    *,
    attempt_timeout: float,
) -> None:
    """Connect, run the readiness query, and close within one bounded attempt."""
    _require_positive_finite("attempt_timeout", attempt_timeout)

    connection: asyncpg.Connection | None = None
    try:
        async with asyncio.timeout(attempt_timeout):
            connection = await asyncpg.connect(
                database_url,
                timeout=attempt_timeout,
                command_timeout=attempt_timeout,
            )
            result = await connection.fetchval("SELECT 1", timeout=attempt_timeout)
            if result != 1:
                msg = "Database readiness query returned an unexpected result"
                raise RuntimeError(msg)
            await connection.close(timeout=attempt_timeout)
            connection = None
    except BaseException:
        if connection is not None:
            with suppress(Exception):
                connection.terminate()
        raise
