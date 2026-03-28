"""Mem0 memory integration for persistent conversation memory.

This package provides tools and utilities for integrating mem0ai memory system
with the Google ADK agent, enabling persistent memory across conversations.
Uses LiteLLM/OpenRouter for LLM operations and local FastEmbed for embeddings.
"""

from .client import get_mem0_client, is_mem0_enabled
from .manager import Mem0Manager, get_mem0_manager
from .tools import save_memory, search_memory

__all__ = [
    "get_mem0_client",
    "get_mem0_manager",
    "is_mem0_enabled",
    "Mem0Manager",
    "save_memory",
    "search_memory",
]
