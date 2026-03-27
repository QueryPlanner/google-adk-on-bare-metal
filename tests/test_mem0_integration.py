"""Unit tests for mem0_integration module."""

import logging
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from agent.mem0_integration import (
    Mem0Manager,
    get_mem0_client,
    get_mem0_manager,
    is_mem0_enabled,
    save_memory,
    search_memory,
)


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
    """Create a mock mem0 client."""
    client = MagicMock()
    client.add.return_value = {"id": "memory-123"}
    client.search.return_value = [
        {"id": "mem-1", "memory": "test memory 1"},
        {"id": "mem-2", "memory": "test memory 2"},
    ]
    client.get_all.return_value = [
        {"id": "mem-1", "memory": "test memory 1"},
    ]
    return client


@pytest.fixture(autouse=True)
def reset_mem0_globals() -> Generator[None]:
    """Reset global mem0 state before and after each test."""
    import agent.mem0_integration as mem0_module

    # Store original state
    original_client = mem0_module._mem0_client
    original_enabled = mem0_module._mem0_enabled
    original_manager = mem0_module._mem0_manager

    # Reset before test
    mem0_module._mem0_client = None
    mem0_module._mem0_enabled = None
    mem0_module._mem0_manager = None

    yield

    # Restore after test
    mem0_module._mem0_client = original_client
    mem0_module._mem0_enabled = original_enabled
    mem0_module._mem0_manager = original_manager


class TestIsMem0Enabled:
    """Tests for is_mem0_enabled function."""

    def test_returns_cached_enabled_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that cached enabled value is returned without re-checking."""
        import agent.mem0_integration as mem0_module

        # Set cached value directly
        mem0_module._mem0_enabled = True

        # Should return cached value without checking env
        result = is_mem0_enabled()
        assert result is True

    def test_returns_cached_disabled_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that cached disabled value is returned without re-checking."""
        import agent.mem0_integration as mem0_module

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
            "agent.mem0_integration.get_mem0_client",
            side_effect=Exception("Init failed"),
        ):
            result = is_mem0_enabled()

        assert result is False
        assert "Failed to initialize mem0 client" in caplog.text

    def test_enabled_on_successful_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test enabled when client initializes successfully."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        with patch("agent.mem0_integration.get_mem0_client", return_value=MagicMock()):
            result = is_mem0_enabled()

        assert result is True


class TestGetMem0Client:
    """Tests for get_mem0_client function."""

    def test_returns_cached_client(self) -> None:
        """Test that cached client is returned without re-initializing."""
        import agent.mem0_integration as mem0_module

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
            {"mem0": MagicMock(Memory=MagicMock(return_value=mock_mem0_client))},
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
            {"mem0": MagicMock(Memory=MagicMock(return_value=mock_mem0_client))},
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
            {"mem0": MagicMock(Memory=MagicMock(return_value=mock_mem0_client))},
        ):
            result = get_mem0_client()
            assert result is mock_mem0_client

    def test_uses_custom_qdrant_config(
        self, monkeypatch: pytest.MonkeyPatch, mock_mem0_client: MagicMock
    ) -> None:
        """Test that custom Qdrant configuration is used."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")
        monkeypatch.setenv("MEM0_COLLECTION_NAME", "custom_collection")
        monkeypatch.setenv("MEM0_QDRANT_HOST", "custom-host")
        monkeypatch.setenv("MEM0_QDRANT_PORT", "7333")

        with patch.dict(
            "sys.modules",
            {"mem0": MagicMock(Memory=MagicMock(return_value=mock_mem0_client))},
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

        with patch("agent.mem0_integration.get_mem0_client") as mock_get_client:
            mock_get_client.return_value = mock_mem0_client
            client = manager.client

            assert client is mock_mem0_client
            mock_get_client.assert_called_once()

    def test_client_property_returns_cached_client(
        self, monkeypatch: pytest.MonkeyPatch, mock_mem0_client: MagicMock
    ) -> None:
        """Test that client property returns cached client."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        manager = Mem0Manager()

        with patch("agent.mem0_integration.get_mem0_client") as mock_get_client:
            mock_get_client.return_value = mock_mem0_client
            # Access client twice
            _ = manager.client
            _ = manager.client

            # Should only initialize once
            mock_get_client.assert_called_once()


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

    def test_save_memory_success(
        self, monkeypatch: pytest.MonkeyPatch, mock_mem0_client: MagicMock
    ) -> None:
        """Test successful save_memory operation."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        manager = Mem0Manager()

        with patch(
            "agent.mem0_integration.get_mem0_client", return_value=mock_mem0_client
        ):
            result = manager.save_memory("test content", user_id="test_user")

            assert result["status"] == "success"
            assert result["memory_id"] == "memory-123"
            mock_mem0_client.add.assert_called_once()

    def test_save_memory_with_metadata(
        self, monkeypatch: pytest.MonkeyPatch, mock_mem0_client: MagicMock
    ) -> None:
        """Test save_memory with metadata."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        manager = Mem0Manager()
        metadata = {"source": "test", "priority": "high"}

        with patch(
            "agent.mem0_integration.get_mem0_client", return_value=mock_mem0_client
        ):
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

        with patch(
            "agent.mem0_integration.get_mem0_client", return_value=mock_mem0_client
        ):
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

    def test_search_memory_success(
        self, monkeypatch: pytest.MonkeyPatch, mock_mem0_client: MagicMock
    ) -> None:
        """Test successful search_memory operation."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        manager = Mem0Manager()

        with patch(
            "agent.mem0_integration.get_mem0_client", return_value=mock_mem0_client
        ):
            result = manager.search_memory("test query", user_id="test_user", limit=5)

            assert result["status"] == "success"
            assert len(result["memories"]) == 2
            mock_mem0_client.search.assert_called_once_with(
                "test query", user_id="test_user", limit=5
            )

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

        with patch(
            "agent.mem0_integration.get_mem0_client", return_value=mock_mem0_client
        ):
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

    def test_get_all_memories_success(
        self, monkeypatch: pytest.MonkeyPatch, mock_mem0_client: MagicMock
    ) -> None:
        """Test successful get_all_memories operation."""
        monkeypatch.setenv("MEM0_LLM_API_KEY", "test-key")

        manager = Mem0Manager()

        with patch(
            "agent.mem0_integration.get_mem0_client", return_value=mock_mem0_client
        ):
            result = manager.get_all_memories(user_id="test_user")

            assert result["status"] == "success"
            assert len(result["memories"]) == 1
            mock_mem0_client.get_all.assert_called_once_with(user_id="test_user")

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

        with patch(
            "agent.mem0_integration.get_mem0_client", return_value=mock_mem0_client
        ):
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

        with patch(
            "agent.mem0_integration.get_mem0_client", return_value=mock_mem0_client
        ):
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

        with patch(
            "agent.mem0_integration.get_mem0_client", return_value=mock_mem0_client
        ):
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

        with patch(
            "agent.mem0_integration.get_mem0_client", return_value=mock_mem0_client
        ):
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

        with patch(
            "agent.mem0_integration.get_mem0_client", return_value=mock_mem0_client
        ):
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

        with patch(
            "agent.mem0_integration.get_mem0_client", return_value=mock_mem0_client
        ):
            result = search_memory(mock_tool_context_no_state, "test query")

            assert result["status"] == "success"
            call_kwargs = mock_mem0_client.search.call_args[1]
            assert call_kwargs["user_id"] == "default"
