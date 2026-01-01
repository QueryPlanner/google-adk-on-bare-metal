"""Environment variable helpers."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_standard_env_files(*, repo_root: Path, agent_dir: Path) -> None:
    """Load `.env` files in a predictable order (non-destructive).

    - Repo root `.env` supports shared settings.
    - Agent dir `.env` supports agent-specific overrides.
    """
    load_dotenv(dotenv_path=repo_root / ".env", override=False)
    load_dotenv(dotenv_path=agent_dir / ".env", override=False)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} is not set")
    return value
