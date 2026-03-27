"""ADK LlmAgent configuration."""

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.plugins.global_instruction_plugin import GlobalInstructionPlugin
from google.adk.plugins.logging_plugin import LoggingPlugin

from .callbacks import (
    LoggingCallbacks,
    add_memories_to_context,
    add_session_to_memory,
)
from .mem0_integration import is_mem0_enabled, save_memory, search_memory
from .prompt import (
    return_description_root,
    return_global_instruction,
    return_instruction_root,
)
from .tools import example_tool

logger = logging.getLogger(__name__)

logging_callbacks = LoggingCallbacks()


def _find_and_load_dotenv() -> None:
    """Load a nearby ``.env`` so ``ROOT_AGENT_MODEL`` is set before we read it.

    The ADK agent loader calls ``load_dotenv_for_agent`` before importing this
    module, but other import paths (tests, tooling) may import ``agent`` first.
    Loading here avoids defaulting to native Gemini without ``GOOGLE_API_KEY``.
    """
    here = Path(__file__).resolve().parent
    for directory in (here, *here.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            break


def _normalize_model_for_openrouter(model_name: str) -> str:
    """Map common IDs to OpenRouter/LiteLLM form when routing via OpenRouter only.

    Examples:
        ``gemini-2.5-flash`` → ``openrouter/google/gemini-2.5-flash``
        ``google/gemini-2.0-flash-001`` → ``openrouter/google/gemini-2.0-flash-001``
        ``openrouter/openai/gpt-oss-120b`` → unchanged
    """
    normalized = model_name.strip()
    lower = normalized.lower()
    if lower.startswith("openrouter/"):
        return normalized
    if "/" in normalized:
        return f"openrouter/{normalized}"
    if normalized.startswith("gemini-"):
        return f"openrouter/google/{normalized}"
    return normalized


_find_and_load_dotenv()

# Determine model configuration
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

model_name = os.getenv("ROOT_AGENT_MODEL", "gemini-2.5-flash")
model: Any = model_name

use_litellm = False

# OpenRouter-only: never use native Gemini (requires GOOGLE_API_KEY).
if openrouter_api_key and not google_api_key:
    model_name = _normalize_model_for_openrouter(model_name)
    use_litellm = True
elif model_name.lower().startswith("openrouter/") or "/" in model_name:
    use_litellm = True

if use_litellm:
    try:
        from google.adk.models import LiteLlm

        litellm_kwargs: dict[str, Any] = {}
        if model_name.lower().startswith("openrouter/") and openrouter_api_key:
            litellm_kwargs["api_key"] = openrouter_api_key

        logger.info("Using LiteLlm for model: %s", model_name)
        model = LiteLlm(model=model_name, **litellm_kwargs)
    except ImportError:
        logger.warning(
            "LiteLlm not available, falling back to string model name. "
            "OpenRouter models may not work."
        )

# Build the list of tools, optionally including mem0 tools
agent_tools: list[Any] = [example_tool]

# Conditionally add mem0 tools if mem0 is configured
if is_mem0_enabled():
    logger.info("mem0 is enabled, adding memory tools")
    agent_tools.extend([save_memory, search_memory])
else:
    logger.info("mem0 is not configured, memory tools disabled")

# Build before_model_callback with optional memory injection
before_model_callbacks: list[Any] = [logging_callbacks.before_model]
if is_mem0_enabled():
    before_model_callbacks.append(add_memories_to_context)

root_agent = LlmAgent(
    name="root_agent",
    description=return_description_root(),
    before_agent_callback=logging_callbacks.before_agent,
    after_agent_callback=[logging_callbacks.after_agent, add_session_to_memory],
    model=model,
    instruction=return_instruction_root(),
    tools=agent_tools,
    before_model_callback=before_model_callbacks,
    after_model_callback=logging_callbacks.after_model,
    before_tool_callback=logging_callbacks.before_tool,
    after_tool_callback=logging_callbacks.after_tool,
)

# Optional App configs explicitly set to None for template documentation
app = App(
    name="agent",
    root_agent=root_agent,
    plugins=[
        GlobalInstructionPlugin(return_global_instruction),
        LoggingPlugin(),
    ],
    events_compaction_config=None,
    context_cache_config=None,
    resumability_config=None,
)
