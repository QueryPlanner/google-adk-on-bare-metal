"""Truthful process liveness and database readiness probes."""

from __future__ import annotations

import logging

from fastapi.responses import JSONResponse
from pydantic import SecretStr

from .database import (
    _require_positive_finite,
    check_database,
    normalize_database_url,
)

logger = logging.getLogger(__name__)


async def live() -> dict[str, str]:
    """Report process liveness without checking external dependencies."""
    return {"status": "alive"}


class DatabaseReadinessProbe:
    """Report whether the configured PostgreSQL dependency is ready."""

    def __init__(
        self,
        database_url: SecretStr | None,
        *,
        attempt_timeout: float,
    ) -> None:
        _require_positive_finite("attempt_timeout", attempt_timeout)
        self._database_url = (
            SecretStr(normalize_database_url(database_url.get_secret_value()))
            if database_url is not None
            else None
        )
        self._attempt_timeout = attempt_timeout

    async def __call__(self) -> JSONResponse:
        """Run one bounded database check and return a stable public contract."""
        if self._database_url is None:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "ready",
                    "checks": {"database": "not_configured"},
                },
            )

        try:
            await check_database(
                self._database_url.get_secret_value(),
                attempt_timeout=self._attempt_timeout,
            )
        except Exception:
            logger.warning("Database readiness check failed.")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "checks": {"database": "unavailable"},
                },
            )

        return JSONResponse(
            status_code=200,
            content={
                "status": "ready",
                "checks": {"database": "healthy"},
            },
        )
