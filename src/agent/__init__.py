"""Agent implementation public package interface.

The ADK app is loaded lazily so importing utility submodules does not construct the
agent or load its runtime environment.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

    from google.adk.apps import App

    agent: ModuleType
    app: App

__all__ = ["agent", "app"]


def __getattr__(name: str) -> Any:
    """Load the ADK module and app only when callers explicitly request them."""
    if name not in __all__:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    from importlib import import_module

    agent_module = import_module(".agent", __name__)
    globals()["agent"] = agent_module
    globals()["app"] = agent_module.app
    return globals()[name]
