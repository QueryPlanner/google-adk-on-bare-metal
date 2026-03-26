"""Mem0 memory integration for persistent conversation memory.

This module provides tools and utilities for integrating mem0ai memory system
with the Google ADK agent, enabling persistent memory across conversations.
Uses LiteLLM/OpenRouter for LLM operations and local FastEmbed for embeddings.
"""

import logging
import os
from typing import Any

from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)

# Global mem0 client instance (initialized lazily)
_mem0_client: Any = None
_mem0_enabled: bool | None = None


def is_mem0_enabled() -> bool:
    """Check if mem0 is configured and available.

    Returns:
        True if mem0 is configured with an LLM API key and the client can
        be initialized.
    """
    global _mem0_enabled

    if _mem0_enabled is not None:
        return _mem0_enabled

    # Check for MEM0_LLM_API_KEY or fall back to OPENROUTER_API_KEY
    llm_api_key = os.getenv("MEM0_LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not llm_api_key:
        logger.debug(
            "Neither MEM0_LLM_API_KEY nor OPENROUTER_API_KEY set, "
            "mem0 integration disabled"
        )
        _mem0_enabled = False
        return False

    try:
        get_mem0_client()
        _mem0_enabled = True
        return True
    except Exception as e:
        logger.warning(f"Failed to initialize mem0 client: {e}")
        _mem0_enabled = False
        return False


def get_mem0_client() -> Any:
    """Get or create the mem0 client instance.

    Configures mem0 with:
    - LiteLLM for LLM operations (OpenRouter or other providers)
    - FastEmbed for local embeddings (no API key needed)
    - Qdrant for local vector storage

    Returns:
        The mem0 client instance.

    Raises:
        ImportError: If mem0ai is not installed.
        ValueError: If no LLM API key is configured.
    """
    global _mem0_client

    if _mem0_client is not None:
        return _mem0_client

    logger.debug("Initializing mem0 client...")

    # Step 1: Fetch API key
    llm_api_key = os.getenv("MEM0_LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not llm_api_key:
        raise ValueError(
            "MEM0_LLM_API_KEY or OPENROUTER_API_KEY environment variable is required"
        )
    api_key_source = (
        "MEM0_LLM_API_KEY" if os.getenv("MEM0_LLM_API_KEY") else "OPENROUTER_API_KEY"
    )
    logger.debug(f"API key source: {api_key_source}")

    # Step 2: Resolve LLM model
    llm_model = os.getenv("MEM0_LLM_MODEL", "openrouter/google/gemini-2.0-flash-001")
    logger.debug(f"LLM model: {llm_model}")

    # Step 3: Fetch additional config values
    llm_temperature = float(os.getenv("MEM0_LLM_TEMPERATURE", "0.1"))
    llm_max_tokens = int(os.getenv("MEM0_LLM_MAX_TOKENS", "1000"))
    collection_name = os.getenv("MEM0_COLLECTION_NAME", "agent_memories")
    qdrant_host = os.getenv("MEM0_QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("MEM0_QDRANT_PORT", "6333"))
    logger.debug(
        f"Qdrant config: {qdrant_host}:{qdrant_port}, collection: {collection_name}"
    )

    try:
        from mem0 import Memory  # type: ignore[import-untyped]

        # Step 4: Build configuration
        config: dict[str, Any] = {
            "version": "v1.1",
            "llm": {
                "provider": "litellm",
                "config": {
                    "model": llm_model,
                    "api_key": llm_api_key,
                    "temperature": llm_temperature,
                    "max_tokens": llm_max_tokens,
                },
            },
            "embedder": {
                "provider": "fastembed",
                "config": {
                    "model": "thenlper/gte-large",
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": collection_name,
                    "host": qdrant_host,
                    "port": qdrant_port,
                },
            },
        }
        logger.debug("Configuration built, creating Memory instance...")

        # Step 5: Initialize client
        _mem0_client = Memory(config)
        logger.info(f"mem0 client initialized successfully with model: {llm_model}")
        return _mem0_client

    except ImportError as e:
        raise ImportError(
            "mem0ai is not installed. Install it with: pip install mem0ai"
        ) from e


class Mem0Manager:
    """Manager class for mem0 memory operations.

    This class provides a high-level interface for storing and retrieving
    memories using the mem0ai library.

    Attributes:
        client: The mem0 client instance.
        user_id: Default user ID for memory operations.
    """

    def __init__(self, user_id: str | None = None) -> None:
        """Initialize the Mem0Manager.

        Args:
            user_id: Optional default user ID for memory operations.
                If not provided, uses MEM0_USER_ID env var or defaults to "default".
        """
        self._client: Any = None
        self._user_id: str = user_id or str(os.getenv("MEM0_USER_ID", "default"))

    @property
    def client(self) -> Any:
        """Get the mem0 client, initializing if needed.

        Returns:
            The mem0 client instance.
        """
        if self._client is None:
            self._client = get_mem0_client()
        return self._client

    @property
    def user_id(self) -> str:
        """Get the default user ID.

        Returns:
            The configured user ID.
        """
        return self._user_id

    def save_memory(
        self,
        content: str,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Save a memory to mem0.

        Args:
            content: The memory content to store.
            user_id: Optional user ID (uses default if not provided).
            metadata: Optional metadata to attach to the memory.

        Returns:
            A dictionary with the result of the save operation.
        """
        if not is_mem0_enabled():
            return {
                "status": "disabled",
                "message": "mem0 is not configured or unavailable",
            }

        try:
            result = self.client.add(
                content,
                user_id=user_id or self._user_id,
                metadata=metadata,
            )
            logger.debug(f"Saved memory: {result}")
            return {
                "status": "success",
                "message": "Memory saved successfully",
                "memory_id": result.get("id") if isinstance(result, dict) else None,
            }
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
            return {
                "status": "error",
                "message": f"Failed to save memory: {e}",
            }

    def search_memory(
        self,
        query: str,
        user_id: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Search for relevant memories in mem0.

        Args:
            query: The search query.
            user_id: Optional user ID (uses default if not provided).
            limit: Maximum number of memories to return.

        Returns:
            A dictionary with the search results.
        """
        if not is_mem0_enabled():
            return {
                "status": "disabled",
                "message": "mem0 is not configured or unavailable",
                "memories": [],
            }

        try:
            results = self.client.search(
                query,
                user_id=user_id or self._user_id,
                limit=limit,
            )
            logger.debug(f"Search results: {results}")
            return {
                "status": "success",
                "memories": results if isinstance(results, list) else [],
            }
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return {
                "status": "error",
                "message": f"Failed to search memories: {e}",
                "memories": [],
            }

    def get_all_memories(self, user_id: str | None = None) -> dict[str, Any]:
        """Get all memories for a user.

        Args:
            user_id: Optional user ID (uses default if not provided).

        Returns:
            A dictionary with all memories.
        """
        if not is_mem0_enabled():
            return {
                "status": "disabled",
                "message": "mem0 is not configured or unavailable",
                "memories": [],
            }

        try:
            results = self.client.get_all(user_id=user_id or self._user_id)
            logger.debug(
                f"Retrieved {len(results) if isinstance(results, list) else 0} memories"
            )
            return {
                "status": "success",
                "memories": results if isinstance(results, list) else [],
            }
        except Exception as e:
            logger.error(f"Failed to get memories: {e}")
            return {
                "status": "error",
                "message": f"Failed to get memories: {e}",
                "memories": [],
            }


# Global manager instance
_mem0_manager: Mem0Manager | None = None


def get_mem0_manager() -> Mem0Manager:
    """Get or create the global Mem0Manager instance.

    Returns:
        The Mem0Manager instance.
    """
    global _mem0_manager
    if _mem0_manager is None:
        _mem0_manager = Mem0Manager()
    return _mem0_manager


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
