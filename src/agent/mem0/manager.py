"""Mem0 manager class for high-level memory operations.

This module provides the Mem0Manager class which wraps the mem0 client
with a higher-level interface for storing and retrieving memories.
"""

import logging
import os
from typing import Any

from .client import get_mem0_client, is_mem0_enabled

logger = logging.getLogger(__name__)

# Global manager instance
_mem0_manager: "Mem0Manager | None" = None


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

            # Mem0 v1.x returns {"results": [...]} for add operations
            memory_id = None
            if isinstance(result, dict):
                if "results" in result and result["results"]:
                    # Extract ID from the first result
                    memory_id = result["results"][0].get("id")
                elif "id" in result:
                    # Fallback for older response format
                    memory_id = result.get("id")

            return {
                "status": "success",
                "message": "Memory saved successfully",
                "memory_id": memory_id,
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
            result = self.client.search(
                query,
                user_id=user_id or self._user_id,
                limit=limit,
            )
            logger.debug(f"Search results: {result}")

            # Mem0 v1.x returns {"results": [...]} for search operations
            if isinstance(result, dict) and "results" in result:
                memories = result["results"]
            elif isinstance(result, list):
                # Fallback for older response format
                memories = result
            else:
                memories = []

            return {
                "status": "success",
                "memories": memories,
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
            result = self.client.get_all(user_id=user_id or self._user_id)
            logger.debug(f"Get all results: {result}")

            # Mem0 v1.x returns {"results": [...]} for get_all operations
            if isinstance(result, dict) and "results" in result:
                memories = result["results"]
            elif isinstance(result, list):
                # Fallback for older response format
                memories = result
            else:
                memories = []

            logger.debug(f"Retrieved {len(memories)} memories")
            return {
                "status": "success",
                "memories": memories,
            }
        except Exception as e:
            logger.error(f"Failed to get memories: {e}")
            return {
                "status": "error",
                "message": f"Failed to get memories: {e}",
                "memories": [],
            }


def get_mem0_manager() -> Mem0Manager:
    """Get or create the global Mem0Manager instance.

    Returns:
        The Mem0Manager instance.
    """
    global _mem0_manager
    if _mem0_manager is None:
        _mem0_manager = Mem0Manager()
    return _mem0_manager
