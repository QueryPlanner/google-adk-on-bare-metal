"""FastAPI server entrypoint for the Agent Platform.
This server hosts all agents found in the 'agents/' directory.

Run:
  - `python -m server`
"""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app

from platform_utils.db_url import to_asyncpg_sqlalchemy_url
from platform_utils.env import load_standard_env_files, require_env


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def _agents_dir() -> str:
  """Directory that ADK scans for agent folders."""
  repo_root = Path(__file__).resolve().parent
  default_agents_dir = repo_root / "agents"
  return os.getenv("AGENTS_DIR", str(default_agents_dir))


def create_app() -> FastAPI:
  # Load .env from root or specific override
  repo_root = Path(__file__).resolve().parent

  # Check for .env in root
  load_standard_env_files(repo_root=repo_root, agent_dir=Path("."))

  session_service_uri = to_asyncpg_sqlalchemy_url(require_env("DATABASE_URL"))

  return get_fast_api_app(
      agents_dir=_agents_dir(),
      session_service_uri=session_service_uri,
      web=True,
  )


app = create_app()


@app.get("/health")
async def health() -> dict[str, str]:
  return {"status": "ok"}


def main() -> None:
  host = os.getenv("HOST", DEFAULT_HOST)
  port = int(os.getenv("PORT", str(DEFAULT_PORT)))
  uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
  main()