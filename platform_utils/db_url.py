"""Database URL normalization for SQLAlchemy AsyncEngine + asyncpg."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

ASYNC_POSTGRES_SCHEME_PREFIX = "postgresql+asyncpg://"
SYNC_POSTGRES_SCHEME_PREFIX = "postgresql://"


def to_asyncpg_sqlalchemy_url(url: str) -> str:
    """Convert common Postgres URLs into asyncpg-compatible SQLAlchemy URLs.

    Why:
    - ADK's `DatabaseSessionService` uses SQLAlchemy `create_async_engine()`.
    - `asyncpg` doesn't accept libpq-style query params like `sslmode=require`.

    Behavior:
    - `postgresql://...` -> `postgresql+asyncpg://...`
    - `?sslmode=require` -> `?ssl=require` (asyncpg supports `ssl=...`)
    - drops `channel_binding` (asyncpg doesn't accept it)
    """
    if url.startswith(SYNC_POSTGRES_SCHEME_PREFIX):
        async_url = ASYNC_POSTGRES_SCHEME_PREFIX + url.removeprefix(SYNC_POSTGRES_SCHEME_PREFIX)
    else:
        async_url = url

    parts = urlsplit(async_url)
    query_items = dict(parse_qsl(parts.query, keep_blank_values=True))

    sslmode = query_items.pop("sslmode", None)
    if sslmode and "ssl" not in query_items:
        query_items["ssl"] = sslmode

    query_items.pop("channel_binding", None)

    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(query_items, doseq=True),
            parts.fragment,
        )
    )
