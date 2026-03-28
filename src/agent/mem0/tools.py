"""Mem0 tool functions for ADK agent integration.

This module provides tool functions that can be used by the LLM agent
to interact with the mem0 memory system.
"""

import logging
from typing import Any

from google.adk.tools import ToolContext

from .manager import get_mem0_manager

logger = logging.getLogger(__name__)


def save_memory(
    tool_context: ToolContext,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tool to save a memory to mem0.

    This tool allows the agent to store important information from
    conversations for future reference.

    Args:
        tool_context: ADK ToolContext with access to session state.
        content: The memory content to store.
        metadata: Optional metadata to attach to the memory.

    Returns:
        A dictionary with the result of the save operation.
    """
    logger.info(f"save_memory tool called with content length: {len(content)}")

    # Get user_id from session state if available
    user_id = tool_context.state.get("user_id") if tool_context.state else None

    manager = get_mem0_manager()
    return manager.save_memory(content, user_id=user_id, metadata=metadata)


def search_memory(
    tool_context: ToolContext,
    query: str,
    limit: int = 10,
) -> dict[str, Any]:
    """Tool to search for relevant memories in mem0.

    This tool allows the agent to retrieve relevant memories based on
    a search query to provide contextually appropriate responses.

    Args:
        tool_context: ADK ToolContext with access to session state.
        query: The search query to find relevant memories.
        limit: Maximum number of memories to return (default: 10).

    Returns:
        A dictionary with the search results containing relevant memories.
    """
    logger.info(f"search_memory tool called with query: {query[:50]}...")

    # Get user_id from session state if available
    user_id = tool_context.state.get("user_id") if tool_context.state else None

    manager = get_mem0_manager()
    return manager.search_memory(query, user_id=user_id, limit=limit)
