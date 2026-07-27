"""Subprocess contracts for lazy package and ADK loading behavior."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, cast

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
_RESULT_PREFIX = "PROBE_RESULT="


def _probe_environment(tmp_path: Path) -> dict[str, str]:
    """Build a minimal subprocess environment with no inherited credentials."""
    return {
        "ADK_DISABLE_LOAD_DOTENV": "true",
        "HOME": str(tmp_path),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(_SRC_ROOT),
    }


def _run_probe(tmp_path: Path, source: str) -> dict[str, Any]:
    """Run an isolated interpreter and parse its structured result."""
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=tmp_path,
        env=_probe_environment(tmp_path),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result_line = next(
        (
            line
            for line in reversed(completed.stdout.splitlines())
            if line.startswith(_RESULT_PREFIX)
        ),
        None,
    )
    assert result_line is not None, completed.stderr
    return cast(
        dict[str, Any],
        json.loads(result_line.removeprefix(_RESULT_PREFIX)),
    )


def test_config_import_has_no_agent_or_dotenv_side_effects(tmp_path: Path) -> None:
    """Test importing settings neither builds the agent nor mutates environment."""
    (tmp_path / ".env").write_text(
        "IMPORT_DOTENV_SENTINEL=dotenv-secret-canary\n",
        encoding="utf-8",
    )

    result = _run_probe(
        tmp_path,
        """
        import json
        import os
        import sys

        before = dict(os.environ)
        import agent.utils.config  # noqa: F401
        after = dict(os.environ)

        result = {
            "agent_loaded": "agent.agent" in sys.modules,
            "dotenv_value": os.getenv("IMPORT_DOTENV_SENTINEL"),
            "environment_unchanged": before == after,
            "mem0_loaded": "mem0" in sys.modules,
            "qdrant_loaded": any(
                name.startswith("qdrant_client") for name in sys.modules
            ),
        }
        print("PROBE_RESULT=" + json.dumps(result))
        """,
    )

    assert result == {
        "agent_loaded": False,
        "dotenv_value": None,
        "environment_unchanged": True,
        "mem0_loaded": False,
        "qdrant_loaded": False,
    }


def test_lazy_app_import_is_process_only_and_cached(tmp_path: Path) -> None:
    """Test the compatibility export ignores cwd dotenv and caches the ADK app."""
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "ROOT_AGENT_MODEL=dotenv/provider-model",
                "OPENROUTER_API_KEY=your-openrouter-key-here",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = _run_probe(
        tmp_path,
        """
        import json
        import os

        import agent
        from agent import app

        model = app.root_agent.model
        model_name = model if isinstance(model, str) else model.model
        result = {
            "app_name": app.name,
            "cached": agent.app is app,
            "dotenv_key_loaded": "OPENROUTER_API_KEY" in os.environ,
            "model_name": model_name,
        }
        print("PROBE_RESULT=" + json.dumps(result))
        """,
    )

    assert result == {
        "app_name": "agent",
        "cached": True,
        "dotenv_key_loaded": False,
        "model_name": "gemini-2.5-flash",
    }


def test_empty_google_key_keeps_openrouter_only_routing(tmp_path: Path) -> None:
    """Test an injected empty Google key does not disable OpenRouter routing."""
    result = _run_probe(
        tmp_path,
        """
        import json
        import os
        from unittest.mock import create_autospec

        from mem0 import Memory

        memory_factory = create_autospec(
            Memory.from_config,
            spec_set=True,
            return_value=object(),
        )
        Memory.from_config = memory_factory
        os.environ["ROOT_AGENT_MODEL"] = "gemini-2.5-flash"
        os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-routing-test"
        os.environ["GOOGLE_API_KEY"] = ""

        from agent import app

        model = app.root_agent.model
        result = {
            "mem0_factory_calls": memory_factory.call_count,
            "model_name": model if isinstance(model, str) else model.model,
        }
        print("PROBE_RESULT=" + json.dumps(result))
        """,
    )

    assert result == {
        "mem0_factory_calls": 1,
        "model_name": "openrouter/google/gemini-2.5-flash",
    }


def test_installed_adk_loader_resolves_lazy_package_export(tmp_path: Path) -> None:
    """Test the pinned ADK loader loads dotenv while preserving process values."""
    (tmp_path / "agent").symlink_to(_SRC_ROOT / "agent", target_is_directory=True)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "ADK_LOADER_SENTINEL=dotenv-loaded",
                "ROOT_AGENT_MODEL=openrouter/dotenv/model",
                "",
            ]
        ),
        encoding="utf-8",
    )
    result = _run_probe(
        tmp_path,
        f"""
        import json
        import os

        from google.adk.apps import App
        from google.adk.cli.utils.agent_loader import AgentLoader

        os.environ.pop("ADK_DISABLE_LOAD_DOTENV")
        os.environ["ROOT_AGENT_MODEL"] = "openrouter/process/model"
        loaded = AgentLoader({str(tmp_path)!r}).load_agent("agent")
        model = loaded.root_agent.model
        result = {{
            "app_name": loaded.name,
            "dotenv_loaded": os.getenv("ADK_LOADER_SENTINEL"),
            "is_app": isinstance(loaded, App),
            "model_name": model if isinstance(model, str) else model.model,
        }}
        print("PROBE_RESULT=" + json.dumps(result))
        """,
    )

    assert result == {
        "app_name": "agent",
        "dotenv_loaded": "dotenv-loaded",
        "is_app": True,
        "model_name": "openrouter/process/model",
    }
