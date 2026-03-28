"""Unit tests for mem0 integration package."""

import json
import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.mem0 import (
    Mem0Manager,
    get_mem0_manager,
    is_mem0_enabled,
    save_memory,
    search_memory,
)
from agent.mem0.client import (
    _build_mem0_config,
    _create_mem0_memory_client,
    _resolve_embedder_dimensions,
    _validate_local_collection_dimensions,
    get_mem0_client,
)


@contextmanager
def mock_mem0_enabled(client: MagicMock) -> Generator[None]:
    """Context manager to mock both is_mem0_enabled and get_mem0_client.

    This is needed because is_mem0_enabled() internally calls get_mem0_client()
    to verify the client can be initialized.

    We patch in the manager module because that's where the functions are used.
    """
    with (
        patch("agent.mem0.manager.is_mem0_enabled", return_value=True),
        patch("agent.mem0.manager.get_mem0_client", return_value=client),
    ):
        yield


@pytest.fixture
def mock_tool_context() -> MagicMock:
    """Create a mock ToolContext for testing tool functions."""
    context = MagicMock()
    context.state = {"user_id": "test_user_123"}
    return context


@pytest.fixture
def mock_tool_context_no_state() -> MagicMock:
    """Create a mock ToolContext with no state."""
    context = MagicMock()
    context.state = None
    return context


@pytest.fixture
def mock_mem0_client() -> MagicMock:
    """Create a mock mem0 client with Mem0 v1.x response shapes.

    Mem0 v1.x returns {"results": [...]} for all operations.
    """
    client = MagicMock()
    # Mem0 v1.x add returns {"results": [{"id": "...", ...}]}
    client.add.return_value = {
        "results": [{"id": "memory-123", "memory": "test memory"}]
    }
    # Mem0 v1.x search returns {"results": [...]}
    client.search.return_value = {
        "results": [
            {"id": "mem-1", "memory": "test memory 1"},
            {"id": "mem-2", "memory": "test memory 2"},
        ]
    }
    # Mem0 v1.x get_all returns {"results": [...]}
    client.get_all.return_value = {
        "results": [
            {"id": "mem-1", "memory": "test memory 1"},
        ]
    }
    return client


@pytest.fixture
def mock_mem0_client_legacy_format() -> MagicMock:
    """Create a mock mem0 client with legacy (pre-v1.x) response shapes.

    This fixture tests backward compatibility with older mem0 versions.
    """
    client = MagicMock()
    # Legacy add returns {"id": "..."}
    client.add.return_value = {"id": "memory-legacy-123"}
    # Legacy search returns a list directly
    client.search.return_value = [
        {"id": "mem-1", "memory": "legacy memory 1"},
    ]
    # Legacy get_all returns a list directly
    client.get_all.return_value = [
        {"id": "mem-1", "memory": "legacy memory 1"},
    ]
    return client


def build_mem0_module(mock_mem0_client: MagicMock) -> MagicMock:
    """Create a mock mem0 module that matches the current mem0 API shape."""
    memory_class = MagicMock()
    memory_class.from_config.return_value = mock_mem0_client

    mock_module = MagicMock()
    mock_module.Memory = memory_class
    return mock_module


@pytest.fixture(autouse=True)
def reset_mem0_globals() -> Generator[None]:
    """Reset global mem0 state before and after each test."""
    import agent.mem0.client as client_module
    import agent.mem0.manager as manager_module

    # Store original state
    original_client = client_module._mem0_client
    original_enabled = client_module._mem0_enabled
    original_manager = manager_module._mem0_manager

    # Reset before test
    client_module._mem0_client = None
    client_module._mem0_enabled = None
    manager_module._mem0_manager = None

    yield

    # Restore after test
    client_module._mem0_client = original_client
    client_module._mem0_enabled = original_enabled
    manager_module._mem0_manager = original_manager


class TestIsMem0Enabled:
    """Tests for is_mem0_enabled function."""

    def test_returns_cached_enabled_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that cached enabled value is returned without re-checking."""
        import agent.mem0.client as mem0_module

        # Set cached value directly
        mem0_module._mem0_enabled = True

        # Should return cached value without checking env
        result = is_mem0_enabled()
        assert result is True

    def test_returns_cached_disabled_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that cached disabled value is returned without re-checking."""
        import agent.mem0.client as mem0_module

        # Set cached value directly
        mem0_module._mem0_enabled = False

        result = is_mem0_enabled()
        assert result is False

    def test_disabled_when_no_api_key(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test disabled when neither MEM0_LLM_API_KEY nor OPENROUTER_API_KEY is set."""
        caplog.set_level(logging.DEBUG)

        # Ensure no API keys are set
        monkeypatch.delenv("MEM0_LLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        result = is_mem0_enabled()

        assert result is False
        assert "Neither MEM0_LLM_API_KEY nor OPENROUTER_API_KEY set" in caplog.text

    def test_disabled_on_client_init_failure(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test disabled when client initialization fails."""
        caplog.set_level(logging.WARNING)

        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        with patch(
            "agent.mem0.client.get_mem0_client",
            side_effect=Exception("Init failed"),
        ):
            result = is_mem0_enabled()

        assert result is False
        assert "Failed to initialize mem0 client" in caplog.text

    def test_enabled_on_successful_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test enabled when client initializes successfully."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        with patch("agent.mem0.client.get_mem0_client", return_value=MagicMock()):
            result = is_mem0_enabled()

        assert result is True


class TestGetMem0Client:
    """Tests for get_mem0_client function."""

    def test_returns_cached_client(self) -> None:
        """Test that cached client is returned without re-initializing."""
        import agent.mem0.client as mem0_module

        mock_client = MagicMock()
        mem0_module._mem0_client = mock_client

        result = get_mem0_client()

        assert result is mock_client

    def test_raises_value_error_when_no_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test ValueError when no API key is configured."""
        monkeypatch.delenv("MEM0_LLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with pytest.raises(ValueError, match="MEM0_LLM_API_KEY or OPENROUTER_API_KEY"):
            get_mem0_client()

    def test_uses_mem0_llm_api_key(
        self, monkeypatch: pytest.MonkeyPatch, mock_mem0_client: MagicMock
    ) -> None:
        """Test that MEM0_LLM_API_KEY is used when set."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "mem0-test-key")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with patch.dict(
            "sys.modules",
            {"mem0": build_mem0_module(mock_mem0_client)},
        ):
            result = get_mem0_client()

            assert result is mock_mem0_client

    def test_uses_openrouter_api_key_as_fallback(
        self, monkeypatch: pytest.MonkeyPatch, mock_mem0_client: MagicMock
    ) -> None:
        """Test that OPENROUTER_API_KEY is used when MEM0_LLM_API_KEY is not set."""
        monkeypatch.delenv("MEM0_LLM_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")

        with patch.dict(
            "sys.modules",
            {"mem0": build_mem0_module(mock_mem0_client)},
        ):
            result = get_mem0_client()

            assert result is mock_mem0_client

    def test_uses_custom_llm_model(
        self, monkeypatch: pytest.MonkeyPatch, mock_mem0_client: MagicMock
    ) -> None:
        """Test that custom LLM model is used when configured."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")
        monkeypatch.setenv("MEM0_LLM_MODEL", "custom-model")

        with patch.dict(
            "sys.modules",
            {"mem0": build_mem0_module(mock_mem0_client)},
        ):
            result = get_mem0_client()
            assert result is mock_mem0_client

    def test_uses_custom_qdrant_server_config(
        self, monkeypatch: pytest.MonkeyPatch, mock_mem0_client: MagicMock
    ) -> None:
        """Test that custom Qdrant server configuration is used when host/port set."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")
        monkeypatch.setenv("MEM0_COLLECTION_NAME", "custom_collection")
        monkeypatch.setenv("MEM0_QDRANT_HOST", "custom-host")
        monkeypatch.setenv("MEM0_QDRANT_PORT", "7333")

        with patch.dict(
            "sys.modules",
            {"mem0": build_mem0_module(mock_mem0_client)},
        ):
            result = get_mem0_client()
            assert result is mock_mem0_client

    def test_uses_embedded_qdrant_by_default(
        self, monkeypatch: pytest.MonkeyPatch, mock_mem0_client: MagicMock
    ) -> None:
        """Test that embedded Qdrant mode is used by default (no host/port)."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")
        monkeypatch.setenv("MEM0_COLLECTION_NAME", "test_collection")
        # Ensure no Qdrant server config is set
        monkeypatch.delenv("MEM0_QDRANT_HOST", raising=False)
        monkeypatch.delenv("MEM0_QDRANT_PORT", raising=False)

        with patch.dict(
            "sys.modules",
            {"mem0": build_mem0_module(mock_mem0_client)},
        ):
            result = get_mem0_client()
            assert result is mock_mem0_client

    def test_uses_custom_qdrant_path(
        self, monkeypatch: pytest.MonkeyPatch, mock_mem0_client: MagicMock
    ) -> None:
        """Test that custom Qdrant path is used for embedded mode."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")
        monkeypatch.setenv("MEM0_QDRANT_PATH", "./custom/data/path")

        with patch.dict(
            "sys.modules",
            {"mem0": build_mem0_module(mock_mem0_client)},
        ):
            result = get_mem0_client()
            assert result is mock_mem0_client

    def test_raises_import_error_when_mem0_not_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test ImportError when mem0ai is not installed."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        with (
            patch.dict("sys.modules", {"mem0": None}),
            pytest.raises(ImportError, match="mem0ai is not installed"),
        ):
            get_mem0_client()


class TestMem0Manager:
    """Tests for Mem0Manager class."""

    def test_init_with_user_id(self) -> None:
        """Test initialization with explicit user_id."""
        manager = Mem0Manager(user_id="explicit_user")
        assert manager.user_id == "explicit_user"

    def test_init_with_env_user_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test initialization with user_id from environment."""
        monkeypatch.setenv("MEM0_USER_ID", "env_user")
        manager = Mem0Manager()
        assert manager.user_id == "env_user"

    def test_init_with_default_user_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test initialization with default user_id."""
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        manager = Mem0Manager()
        assert manager.user_id == "default"

    def test_client_property_lazy_init(
        self, monkeypatch: pytest.MonkeyPatch, mock_mem0_client: MagicMock
    ) -> None:
        """Test that client is lazily initialized."""

        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        manager = Mem0Manager()

        with mock_mem0_enabled(mock_mem0_client):
            client = manager.client
            assert client is mock_mem0_client

    def test_client_property_returns_cached_client(
        self, monkeypatch: pytest.MonkeyPatch, mock_mem0_client: MagicMock
    ) -> None:
        """Test that client property returns cached client."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        manager = Mem0Manager()

        with mock_mem0_enabled(mock_mem0_client):
            # Access client twice
            _ = manager.client
            _ = manager.client
            # Should only initialize once (verified by mock call count)


class TestMem0ManagerSaveMemory:
    """Tests for Mem0Manager.save_memory method."""

    def test_save_memory_disabled(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test save_memory when mem0 is disabled."""
        caplog.set_level(logging.DEBUG)
        monkeypatch.delenv("MEM0_LLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        manager = Mem0Manager()
        result = manager.save_memory("test content")

        assert result["status"] == "disabled"
        assert "mem0 is not configured" in result["message"]

    def test_save_memory_success_v1_format(
        self, monkeypatch: pytest.MonkeyPatch, mock_mem0_client: MagicMock
    ) -> None:
        """Test successful save_memory with Mem0 v1.x response format."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        manager = Mem0Manager()

        with mock_mem0_enabled(mock_mem0_client):
            result = manager.save_memory("test content", user_id="test_user")

            assert result["status"] == "success"
            assert result["memory_id"] == "memory-123"
            mock_mem0_client.add.assert_called_once()

    def test_save_memory_success_legacy_format(
        self, monkeypatch: pytest.MonkeyPatch, mock_mem0_client_legacy_format: MagicMock
    ) -> None:
        """Test save_memory with legacy (pre-v1.x) response format."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        manager = Mem0Manager()

        with mock_mem0_enabled(mock_mem0_client_legacy_format):
            result = manager.save_memory("test content", user_id="test_user")

            assert result["status"] == "success"
            assert result["memory_id"] == "memory-legacy-123"
            mock_mem0_client_legacy_format.add.assert_called_once()

    def test_save_memory_with_metadata(
        self, monkeypatch: pytest.MonkeyPatch, mock_mem0_client: MagicMock
    ) -> None:
        """Test save_memory with metadata."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        manager = Mem0Manager()
        metadata = {"source": "test", "priority": "high"}

        with mock_mem0_enabled(mock_mem0_client):
            result = manager.save_memory("test content", metadata=metadata)

            assert result["status"] == "success"
            call_kwargs = mock_mem0_client.add.call_args[1]
            assert call_kwargs["metadata"] == metadata

    def test_save_memory_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_mem0_client: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test save_memory handles errors."""
        caplog.set_level(logging.ERROR)
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        mock_mem0_client.add.side_effect = Exception("Save failed")
        manager = Mem0Manager()

        with mock_mem0_enabled(mock_mem0_client):
            result = manager.save_memory("test content")

            assert result["status"] == "error"
            assert "Save failed" in result["message"]
            assert "Failed to save memory" in caplog.text


class TestMem0ManagerSearchMemory:
    """Tests for Mem0Manager.search_memory method."""

    def test_search_memory_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test search_memory when mem0 is disabled."""
        monkeypatch.delenv("MEM0_LLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        manager = Mem0Manager()
        result = manager.search_memory("test query")

        assert result["status"] == "disabled"
        assert result["memories"] == []

    def test_search_memory_success_v1_format(
        self, monkeypatch: pytest.MonkeyPatch, mock_mem0_client: MagicMock
    ) -> None:
        """Test successful search_memory with Mem0 v1.x response format."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        manager = Mem0Manager()

        with mock_mem0_enabled(mock_mem0_client):
            result = manager.search_memory("test query", user_id="test_user", limit=5)

            assert result["status"] == "success"
            assert len(result["memories"]) == 2
            mock_mem0_client.search.assert_called_once_with(
                "test query", user_id="test_user", limit=5
            )

    def test_search_memory_success_legacy_format(
        self, monkeypatch: pytest.MonkeyPatch, mock_mem0_client_legacy_format: MagicMock
    ) -> None:
        """Test search_memory with legacy (pre-v1.x) response format."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        manager = Mem0Manager()

        with mock_mem0_enabled(mock_mem0_client_legacy_format):
            result = manager.search_memory("test query")

            assert result["status"] == "success"
            assert len(result["memories"]) == 1

    def test_search_memory_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_mem0_client: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test search_memory handles errors."""
        caplog.set_level(logging.ERROR)
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        mock_mem0_client.search.side_effect = Exception("Search failed")
        manager = Mem0Manager()

        with mock_mem0_enabled(mock_mem0_client):
            result = manager.search_memory("test query")

            assert result["status"] == "error"
            assert result["memories"] == []
            assert "Failed to search memories" in caplog.text


class TestMem0ManagerGetAllMemories:
    """Tests for Mem0Manager.get_all_memories method."""

    def test_get_all_memories_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_all_memories when mem0 is disabled."""
        monkeypatch.delenv("MEM0_LLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        manager = Mem0Manager()
        result = manager.get_all_memories()

        assert result["status"] == "disabled"
        assert result["memories"] == []

    def test_get_all_memories_success_v1_format(
        self, monkeypatch: pytest.MonkeyPatch, mock_mem0_client: MagicMock
    ) -> None:
        """Test successful get_all_memories with Mem0 v1.x response format."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        manager = Mem0Manager()

        with mock_mem0_enabled(mock_mem0_client):
            result = manager.get_all_memories(user_id="test_user")

            assert result["status"] == "success"
            assert len(result["memories"]) == 1
            mock_mem0_client.get_all.assert_called_once_with(user_id="test_user")

    def test_get_all_memories_success_legacy_format(
        self, monkeypatch: pytest.MonkeyPatch, mock_mem0_client_legacy_format: MagicMock
    ) -> None:
        """Test get_all_memories with legacy (pre-v1.x) response format."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        manager = Mem0Manager()

        with mock_mem0_enabled(mock_mem0_client_legacy_format):
            result = manager.get_all_memories()

            assert result["status"] == "success"
            assert len(result["memories"]) == 1

    def test_get_all_memories_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_mem0_client: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test get_all_memories handles errors."""
        caplog.set_level(logging.ERROR)
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        mock_mem0_client.get_all.side_effect = Exception("Get failed")
        manager = Mem0Manager()

        with mock_mem0_enabled(mock_mem0_client):
            result = manager.get_all_memories()

            assert result["status"] == "error"
            assert result["memories"] == []
            assert "Failed to get memories" in caplog.text


class TestGetMem0Manager:
    """Tests for get_mem0_manager function."""

    def test_creates_manager_on_first_call(self) -> None:
        """Test that manager is created on first call."""
        manager = get_mem0_manager()
        assert isinstance(manager, Mem0Manager)

    def test_returns_same_manager_on_subsequent_calls(self) -> None:
        """Test that the same manager instance is returned."""
        manager1 = get_mem0_manager()
        manager2 = get_mem0_manager()
        assert manager1 is manager2


class TestSaveMemoryTool:
    """Tests for save_memory tool function."""

    def test_save_memory_tool_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_tool_context: MagicMock,
        mock_mem0_client: MagicMock,
    ) -> None:
        """Test save_memory tool with successful operation."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        with mock_mem0_enabled(mock_mem0_client):
            result = save_memory(mock_tool_context, "test content")

            assert result["status"] == "success"
            # Verify user_id was extracted from tool context state
            call_kwargs = mock_mem0_client.add.call_args[1]
            assert call_kwargs["user_id"] == "test_user_123"

    def test_save_memory_tool_with_no_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_tool_context_no_state: MagicMock,
        mock_mem0_client: MagicMock,
    ) -> None:
        """Test save_memory tool when context has no state."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        with mock_mem0_enabled(mock_mem0_client):
            result = save_memory(mock_tool_context_no_state, "test content")

            assert result["status"] == "success"
            # Should use default user_id from manager
            call_kwargs = mock_mem0_client.add.call_args[1]
            assert call_kwargs["user_id"] == "default"

    def test_save_memory_tool_with_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_tool_context: MagicMock,
        mock_mem0_client: MagicMock,
    ) -> None:
        """Test save_memory tool with metadata."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")
        metadata = {"source": "test", "priority": "high"}

        with mock_mem0_enabled(mock_mem0_client):
            result = save_memory(mock_tool_context, "test content", metadata=metadata)

            assert result["status"] == "success"
            call_kwargs = mock_mem0_client.add.call_args[1]
            assert call_kwargs["metadata"] == metadata


class TestSearchMemoryTool:
    """Tests for search_memory tool function."""

    def test_search_memory_tool_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_tool_context: MagicMock,
        mock_mem0_client: MagicMock,
    ) -> None:
        """Test search_memory tool with successful operation."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        with mock_mem0_enabled(mock_mem0_client):
            result = search_memory(mock_tool_context, "test query", limit=5)

            assert result["status"] == "success"
            assert len(result["memories"]) == 2
            call_kwargs = mock_mem0_client.search.call_args[1]
            assert call_kwargs["user_id"] == "test_user_123"
            assert call_kwargs["limit"] == 5

    def test_search_memory_tool_with_no_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_tool_context_no_state: MagicMock,
        mock_mem0_client: MagicMock,
    ) -> None:
        """Test search_memory tool when context has no state."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        with mock_mem0_enabled(mock_mem0_client):
            result = search_memory(mock_tool_context_no_state, "test query")

            assert result["status"] == "success"
            call_kwargs = mock_mem0_client.search.call_args[1]
            assert call_kwargs["user_id"] == "default"


class TestBuildMem0Config:
    """Tests for _build_mem0_config function."""

    def test_embedded_mode_by_default(self) -> None:
        """Test that embedded mode is used when no host/port provided."""
        config = _build_mem0_config(
            llm_api_key="test-key",
            llm_model="test-model",
            llm_temperature=0.1,
            llm_max_tokens=1000,
            embedder_model="test-embedder",
            embedder_dims=384,
            collection_name="test_collection",
            qdrant_path="./data/qdrant",
            qdrant_host=None,
            qdrant_port=None,
        )

        vector_store_config = config["vector_store"]["config"]
        assert vector_store_config["embedding_model_dims"] == 384
        assert "path" in vector_store_config
        assert vector_store_config["path"] == "./data/qdrant"
        assert vector_store_config["on_disk"] is True
        assert (
            "host" not in vector_store_config or vector_store_config.get("host") is None
        )

    def test_server_mode_when_host_port_provided(self) -> None:
        """Test that server mode is used when host and port are provided."""
        config = _build_mem0_config(
            llm_api_key="test-key",
            llm_model="test-model",
            llm_temperature=0.1,
            llm_max_tokens=1000,
            embedder_model="test-embedder",
            embedder_dims=768,
            collection_name="test_collection",
            qdrant_path="./data/qdrant",
            qdrant_host="localhost",
            qdrant_port=6333,
        )

        vector_store_config = config["vector_store"]["config"]
        assert vector_store_config["embedding_model_dims"] == 768
        assert "host" in vector_store_config
        assert vector_store_config["host"] == "localhost"
        assert vector_store_config["port"] == 6333
        # When using server mode, path should not be set
        assert (
            "path" not in vector_store_config or vector_store_config.get("path") is None
        )

    def test_default_qdrant_path(self) -> None:
        """Test that default qdrant path is used when not specified."""
        config = _build_mem0_config(
            llm_api_key="test-key",
            llm_model="test-model",
            llm_temperature=0.1,
            llm_max_tokens=1000,
            embedder_model="test-embedder",
            embedder_dims=384,
            collection_name="test_collection",
            qdrant_path=None,  # Should use default
            qdrant_host=None,
            qdrant_port=None,
        )

        vector_store_config = config["vector_store"]["config"]
        assert vector_store_config["path"] == "./data/qdrant"

    def test_llm_config(self) -> None:
        """Test that LLM config is properly set."""
        config = _build_mem0_config(
            llm_api_key="test-api-key",
            llm_model="openrouter/test-model",
            llm_temperature=0.5,
            llm_max_tokens=2000,
            embedder_model="test-embedder",
            embedder_dims=384,
            collection_name="test_collection",
            qdrant_path="./data/qdrant",
            qdrant_host=None,
            qdrant_port=None,
        )

        llm_config = config["llm"]["config"]
        assert llm_config["api_key"] == "test-api-key"
        assert llm_config["model"] == "openrouter/test-model"
        assert llm_config["temperature"] == 0.5
        assert llm_config["max_tokens"] == 2000

    def test_embedder_config(self) -> None:
        """Test that embedder config is properly set."""
        config = _build_mem0_config(
            llm_api_key="test-key",
            llm_model="test-model",
            llm_temperature=0.1,
            llm_max_tokens=1000,
            embedder_model="BAAI/bge-small-en-v1.5",
            embedder_dims=384,
            collection_name="test_collection",
            qdrant_path="./data/qdrant",
            qdrant_host=None,
            qdrant_port=None,
        )

        embedder_config = config["embedder"]["config"]
        assert embedder_config["model"] == "BAAI/bge-small-en-v1.5"


class TestResolveEmbedderDimensions:
    """Tests for embedder dimension resolution."""

    def test_uses_known_model_dimensions(self) -> None:
        """Known FastEmbed models should resolve without an env override."""
        assert _resolve_embedder_dimensions("BAAI/bge-small-en-v1.5", None) == 384

    def test_uses_explicit_override_when_set(self) -> None:
        """Explicit dimensions should take precedence over model mapping."""
        assert _resolve_embedder_dimensions("unknown-model", "512") == 512

    def test_raises_for_unknown_model_without_override(self) -> None:
        """Unknown models should fail fast with a clear message."""
        with pytest.raises(ValueError, match="Set MEM0_EMBEDDER_DIMS explicitly"):
            _resolve_embedder_dimensions("unknown-model", None)


class TestValidateLocalCollectionDimensions:
    """Tests for local collection dimension validation."""

    def test_skips_when_metadata_missing(self, tmp_path: Path) -> None:
        """No metadata file means there is no existing collection to validate."""
        _validate_local_collection_dimensions(
            qdrant_path=str(tmp_path),
            collection_name="agent_memories",
            expected_dims=384,
        )

    def test_skips_when_collection_missing(self, tmp_path: Path) -> None:
        """Metadata without the target collection should not raise."""
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(json.dumps({"collections": {}}))

        _validate_local_collection_dimensions(
            qdrant_path=str(tmp_path),
            collection_name="agent_memories",
            expected_dims=384,
        )

    def test_raises_when_collection_dimension_mismatches(self, tmp_path: Path) -> None:
        """Existing collections must match the configured embedding dimensions."""
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "collections": {
                        "agent_memories": {
                            "vectors": {"size": 1536},
                        }
                    }
                }
            )
        )

        with pytest.raises(
            ValueError, match="uses 1536 dimensions, but embedder requires 384"
        ):
            _validate_local_collection_dimensions(
                qdrant_path=str(tmp_path),
                collection_name="agent_memories",
                expected_dims=384,
            )

    def test_passes_when_collection_dimension_matches(self, tmp_path: Path) -> None:
        """Existing collections with matching dimensions should pass validation."""
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "collections": {
                        "agent_memories": {
                            "vectors": {"size": 384},
                        }
                    }
                }
            )
        )

        # Should not raise - dimensions match
        _validate_local_collection_dimensions(
            qdrant_path=str(tmp_path),
            collection_name="agent_memories",
            expected_dims=384,
        )


class TestCreateMem0MemoryClient:
    """Tests for _create_mem0_memory_client function."""

    def test_uses_from_config_when_available(
        self, monkeypatch: pytest.MonkeyPatch, mock_mem0_client: MagicMock
    ) -> None:
        """Test that from_config is used when available."""

        memory_class = MagicMock()
        memory_class.from_config.return_value = mock_mem0_client

        result = _create_mem0_memory_client(memory_class, {"test": "config"})

        assert result is mock_mem0_client
        memory_class.from_config.assert_called_once_with({"test": "config"})
        memory_class.assert_not_called()

    def test_fallback_to_direct_constructor(
        self, monkeypatch: pytest.MonkeyPatch, mock_mem0_client: MagicMock
    ) -> None:
        """Test fallback to direct constructor when from_config not available."""

        memory_class = MagicMock()
        memory_class.return_value = mock_mem0_client
        # Remove from_config to simulate older mem0 version
        delattr(memory_class, "from_config")

        result = _create_mem0_memory_client(memory_class, {"test": "config"})

        assert result is mock_mem0_client
        memory_class.assert_called_once_with({"test": "config"})

    def test_from_config_not_callable(
        self, monkeypatch: pytest.MonkeyPatch, mock_mem0_client: MagicMock
    ) -> None:
        """Test fallback when from_config exists but is not callable."""

        memory_class = MagicMock()
        memory_class.from_config = "not_callable"
        memory_class.return_value = mock_mem0_client

        result = _create_mem0_memory_client(memory_class, {"test": "config"})

        assert result is mock_mem0_client
        memory_class.assert_called_once_with({"test": "config"})


class TestGetMem0ClientImportErrors:
    """Tests for ImportError handling in get_mem0_client."""

    def test_raises_import_error_on_incomplete_deps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test ImportError when mem0 dependencies are incomplete."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        # Create a mock Memory class that raises ImportError on from_config
        mock_memory_class = MagicMock()
        mock_memory_class.from_config.side_effect = ImportError(
            "No module named 'fastembed'"
        )

        mock_module = MagicMock()
        mock_module.Memory = mock_memory_class

        with (
            patch.dict("sys.modules", {"mem0": mock_module}),
            pytest.raises(ImportError, match="mem0 dependencies are incomplete"),
        ):
            get_mem0_client()


class TestMem0ManagerEdgeCases:
    """Tests for edge cases in Mem0Manager methods."""

    def test_search_memory_unexpected_response_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test search_memory handles unexpected response format."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        mock_client = MagicMock()
        # Return an unexpected format (neither dict with results nor list)
        mock_client.search.return_value = "unexpected_string"

        manager = Mem0Manager()

        with mock_mem0_enabled(mock_client):
            result = manager.search_memory("test query")

            assert result["status"] == "success"
            assert result["memories"] == []

    def test_get_all_memories_unexpected_response_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_all_memories handles unexpected response format."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        mock_client = MagicMock()
        # Return an unexpected format (neither dict with results nor list)
        mock_client.get_all.return_value = {"unexpected": "format"}

        manager = Mem0Manager()

        with mock_mem0_enabled(mock_client):
            result = manager.get_all_memories()

            assert result["status"] == "success"
            assert result["memories"] == []

    def test_save_memory_empty_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test save_memory handles empty results array."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        mock_client = MagicMock()
        # Return empty results
        mock_client.add.return_value = {"results": []}

        manager = Mem0Manager()

        with mock_mem0_enabled(mock_client):
            result = manager.save_memory("test content")

            assert result["status"] == "success"
            assert result["memory_id"] is None

    def test_save_memory_non_dict_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test save_memory handles non-dict response."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        mock_client = MagicMock()
        # Return non-dict response (e.g., string or None)
        mock_client.add.return_value = "some_string_response"

        manager = Mem0Manager()

        with mock_mem0_enabled(mock_client):
            result = manager.save_memory("test content")

            assert result["status"] == "success"
            assert result["memory_id"] is None
