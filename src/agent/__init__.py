"""Agent implementation public package interface.

The ADK app is loaded lazily so importing utility submodules does not construct the
agent or load its runtime environment.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.adk.apps import App

    app: App

__all__ = ["app"]


def __getattr__(name: str) -> Any:
    """Load the public ADK app only when callers explicitly request it."""
    if name != "app":
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    from .agent import app as agent_app

    globals()["app"] = agent_app
    return agent_app
