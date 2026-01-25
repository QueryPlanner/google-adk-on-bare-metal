"""ADK LlmAgent configuration."""

import logging
import os
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.plugins.global_instruction_plugin import GlobalInstructionPlugin
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.tools.preload_memory_tool import PreloadMemoryTool

from .callbacks import LoggingCallbacks, add_session_to_memory
from .prompt import (
    return_description_root,
    return_global_instruction,
    return_instruction_root,
)
from .tools import example_tool

logger = logging.getLogger(__name__)

logging_callbacks = LoggingCallbacks()

# Determine model configuration
model_name = os.getenv("ROOT_AGENT_MODEL", "gemini-2.5-flash")
model: Any = model_name

# Explicitly use LiteLlm for OpenRouter or other provider-prefixed models
# that might not be auto-detected by ADK's registry.
if model_name.lower().startswith("openrouter/") or "/" in model_name:
    try:
        from google.adk.models import LiteLlm

        logger.info(f"Using LiteLlm for model: {model_name}")
        model = LiteLlm(model=model_name)
    except ImportError:
        logger.warning(
            "LiteLlm not available, falling back to string model name. "
            "OpenRouter models may not work."
        )

root_agent = LlmAgent(
    name="root_agent",
    description=return_description_root(),
    before_agent_callback=logging_callbacks.before_agent,
    after_agent_callback=[logging_callbacks.after_agent, add_session_to_memory],
    model=model,
    instruction=return_instruction_root(),
    tools=[PreloadMemoryTool(), example_tool],
    before_model_callback=logging_callbacks.before_model,
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
