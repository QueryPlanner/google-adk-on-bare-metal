"""Mem0 client initialization and configuration.

This module provides client setup, configuration building, and lifecycle
management for the mem0 memory system.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Global mem0 client instance (initialized lazily)
_mem0_client: Any = None
_mem0_enabled: bool | None = None

_DEFAULT_EMBEDDER_MODEL = "BAAI/bge-small-en-v1.5"
_DEFAULT_QDRANT_PATH = "./data/qdrant"
_FASTEMBED_MODEL_DIMS = {
    "BAAI/bge-small-en": 384,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en": 768,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "mixedbread-ai/mxbai-embed-large-v1": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "snowflake/snowflake-arctic-embed-xs": 384,
    "snowflake/snowflake-arctic-embed-s": 384,
    "snowflake/snowflake-arctic-embed-m": 768,
    "snowflake/snowflake-arctic-embed-l": 1024,
    "thenlper/gte-large": 1024,
}


def _resolve_embedder_dimensions(
    embedder_model: str, embedder_dims_override: str | None
) -> int:
    """Resolve embedding dimensions for the configured local embedder.

    Mem0's Qdrant config defaults to 1536 dimensions unless we override it.
    For FastEmbed models we must pass the correct value explicitly, otherwise
    the collection schema is created with the wrong size.
    """
    if embedder_dims_override:
        return int(embedder_dims_override)

    if embedder_model in _FASTEMBED_MODEL_DIMS:
        return _FASTEMBED_MODEL_DIMS[embedder_model]

    msg = (
        f"Unknown embedding dimensions for model '{embedder_model}'. "
        "Set MEM0_EMBEDDER_DIMS explicitly."
    )
    raise ValueError(msg)


def _validate_local_collection_dimensions(
    qdrant_path: str, collection_name: str, expected_dims: int
) -> None:
    """Fail early when an embedded local collection uses the wrong dimensions."""
    metadata_path = Path(qdrant_path) / "meta.json"
    if not metadata_path.is_file():
        return

    metadata = json.loads(metadata_path.read_text())
    collections = metadata.get("collections", {})
    collection_metadata = collections.get(collection_name)
    if not collection_metadata:
        return

    actual_dims = collection_metadata.get("vectors", {}).get("size")
    if actual_dims == expected_dims:
        return

    msg = (
        f"Mem0 collection '{collection_name}' at '{qdrant_path}' uses "
        f"{actual_dims} dimensions, but embedder requires {expected_dims}. "
        "Delete the local Qdrant data or choose a different collection name."
    )
    raise ValueError(msg)


def _build_mem0_config(
    llm_api_key: str,
    llm_model: str,
    llm_temperature: float,
    llm_max_tokens: int,
    embedder_model: str,
    embedder_dims: int,
    collection_name: str,
    qdrant_path: str | None,
    qdrant_host: str | None,
    qdrant_port: int | None,
) -> dict[str, Any]:
    """Build the mem0 configuration dictionary.

    Supports two modes:
    - Embedded on-disk mode (default): Uses local file path for persistence
    - Remote server mode: Connects to a separate Qdrant server

    Keeping this in one place makes it easier to support multiple mem0
    constructor styles without duplicating the same nested config.
    """
    # Build vector store config based on mode
    if qdrant_host and qdrant_port:
        # Remote server mode - connect to external Qdrant
        vector_store_config: dict[str, Any] = {
            "collection_name": collection_name,
            "embedding_model_dims": embedder_dims,
            "host": qdrant_host,
            "port": qdrant_port,
        }
    else:
        # Embedded on-disk mode (default) - no separate service needed
        vector_store_config = {
            "collection_name": collection_name,
            "embedding_model_dims": embedder_dims,
            "path": qdrant_path or _DEFAULT_QDRANT_PATH,
            "on_disk": True,
        }

    return {
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
                "model": embedder_model,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": vector_store_config,
        },
    }


def _create_mem0_memory_client(memory_class: Any, config: dict[str, Any]) -> Any:
    """Create a mem0 Memory client across old and new mem0 versions.

    Newer mem0 releases expect `Memory.from_config(config)` or a typed
    `MemoryConfig`, while older releases accepted `Memory(config)` directly.
    Try the modern API first and only fall back when needed.
    """
    from_config = getattr(memory_class, "from_config", None)
    if callable(from_config):
        return from_config(config)

    return memory_class(config)


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
    embedder_model = os.getenv("MEM0_EMBEDDER_MODEL", _DEFAULT_EMBEDDER_MODEL)
    embedder_dims = _resolve_embedder_dimensions(
        embedder_model=embedder_model,
        embedder_dims_override=os.getenv("MEM0_EMBEDDER_DIMS"),
    )
    collection_name = os.getenv("MEM0_COLLECTION_NAME", "agent_memories")

    # Qdrant configuration - support both embedded and server modes
    # Embedded mode (default): uses on-disk storage at qdrant_path
    # Server mode: connects to external Qdrant at qdrant_host:qdrant_port
    qdrant_path = os.getenv("MEM0_QDRANT_PATH", _DEFAULT_QDRANT_PATH)
    qdrant_host = os.getenv("MEM0_QDRANT_HOST", None)
    qdrant_port = (
        int(os.getenv("MEM0_QDRANT_PORT", "0"))
        if os.getenv("MEM0_QDRANT_PORT")
        else None
    )

    if qdrant_host and qdrant_port:
        logger.debug(
            "Qdrant server mode: %s:%s, collection: %s, embedder: %s",
            qdrant_host,
            qdrant_port,
            collection_name,
            embedder_model,
        )
    else:
        logger.debug(
            "Qdrant embedded mode: path=%s, collection: %s, embedder: %s",
            qdrant_path,
            collection_name,
            embedder_model,
        )
        _validate_local_collection_dimensions(
            qdrant_path=qdrant_path,
            collection_name=collection_name,
            expected_dims=embedder_dims,
        )

    try:
        from mem0 import Memory  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "mem0ai is not installed. Install it with: pip install mem0ai"
        ) from e

    try:
        # Step 4: Build configuration
        config = _build_mem0_config(
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            embedder_model=embedder_model,
            embedder_dims=embedder_dims,
            collection_name=collection_name,
            qdrant_path=qdrant_path,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
        )
        logger.debug("Configuration built, creating Memory instance...")

        # Step 5: Initialize client
        _mem0_client = _create_mem0_memory_client(Memory, config)
        logger.info(f"mem0 client initialized successfully with model: {llm_model}")
        return _mem0_client

    except ImportError as e:
        raise ImportError(
            "mem0 dependencies are incomplete. Install the required extras, "
            "for example: pip install mem0ai fastembed"
        ) from e
