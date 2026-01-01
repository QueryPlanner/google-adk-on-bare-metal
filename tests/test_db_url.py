from platform_utils.db_url import to_asyncpg_sqlalchemy_url


def test_to_asyncpg_sqlalchemy_url_converts_scheme() -> None:
    url = "postgresql://user:pass@host:5432/db"
    assert to_asyncpg_sqlalchemy_url(url).startswith("postgresql+asyncpg://")


def test_to_asyncpg_sqlalchemy_url_translates_sslmode_to_ssl() -> None:
    url = "postgresql://user:pass@host/db?sslmode=require"
    converted = to_asyncpg_sqlalchemy_url(url)
    assert "sslmode=" not in converted
    assert "ssl=require" in converted


def test_to_asyncpg_sqlalchemy_url_drops_channel_binding() -> None:
    url = "postgresql://user:pass@host/db?sslmode=require&channel_binding=require"
    converted = to_asyncpg_sqlalchemy_url(url)
    assert "channel_binding=" not in converted
