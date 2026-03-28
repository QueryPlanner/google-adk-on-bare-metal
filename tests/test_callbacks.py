"""Unit tests for the callbacks module."""

import logging
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from conftest import MockMemoryCallbackContext, MockState
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from agent.callbacks import add_memories_to_context, add_session_to_memory


def as_callback_context(context: MockMemoryCallbackContext) -> CallbackContext:
    """Treat mock callback contexts as real CallbackContext objects for typing."""
    return cast(CallbackContext, context)


class TestAddSessionToMemory:
    """Tests for the add_session_to_memory callback function."""

    @pytest.mark.asyncio
    async def test_add_session_to_memory_success(
        self,
        mock_memory_callback_context: MockMemoryCallbackContext,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that callback succeeds when context.add_session_to_memory succeeds."""
        caplog.set_level(logging.INFO)

        # Execute callback
        await add_session_to_memory(as_callback_context(mock_memory_callback_context))

        # Verify add_session_to_memory was called on the context
        assert mock_memory_callback_context.add_session_to_memory_called

        # Verify logging
        assert "*** Starting add_session_to_memory callback ***" in caplog.text

    @pytest.mark.asyncio
    async def test_add_session_to_memory_handles_value_error(
        self,
        mock_memory_callback_context_no_service: MockMemoryCallbackContext,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that callback handles ValueError (e.g., no memory service)."""
        caplog.set_level(logging.WARNING)

        # Execute callback - should not raise
        await add_session_to_memory(
            as_callback_context(mock_memory_callback_context_no_service)
        )

        # Verify the method was attempted
        assert mock_memory_callback_context_no_service.add_session_to_memory_called

        # Verify warning was logged
        assert (
            "Cannot add session to memory: memory service is not available."
            in caplog.text
        )

    @pytest.mark.asyncio
    async def test_add_session_to_memory_handles_attribute_error(
        self,
        mock_memory_callback_context_with_attribute_error: MockMemoryCallbackContext,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that callback handles AttributeError gracefully."""
        caplog.set_level(logging.WARNING)

        # Execute callback - should not raise
        await add_session_to_memory(
            as_callback_context(mock_memory_callback_context_with_attribute_error)
        )

        # Verify the method was attempted
        ctx = mock_memory_callback_context_with_attribute_error
        assert ctx.add_session_to_memory_called

        # Verify warning was logged with exception details
        assert "Failed to add session to memory" in caplog.text
        assert "AttributeError" in caplog.text

    @pytest.mark.asyncio
    async def test_add_session_to_memory_handles_runtime_error(
        self,
        mock_memory_callback_context_with_runtime_error: MockMemoryCallbackContext,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that callback handles RuntimeError gracefully."""
        caplog.set_level(logging.WARNING)

        # Execute callback - should not raise
        await add_session_to_memory(
            as_callback_context(mock_memory_callback_context_with_runtime_error)
        )

        # Verify the method was attempted
        ctx = mock_memory_callback_context_with_runtime_error
        assert ctx.add_session_to_memory_called

        # Verify warning was logged with exception details
        assert "Failed to add session to memory" in caplog.text
        assert "RuntimeError" in caplog.text
        assert "Memory service connection failed" in caplog.text

    @pytest.mark.asyncio
    async def test_add_session_to_memory_logging_levels(
        self,
        mock_memory_callback_context: MockMemoryCallbackContext,
        mock_memory_callback_context_no_service: MockMemoryCallbackContext,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that callback uses appropriate logging levels."""
        # Test case 1: Success (INFO level)
        caplog.set_level(logging.INFO)
        caplog.clear()

        await add_session_to_memory(as_callback_context(mock_memory_callback_context))

        # Check for INFO log (starting callback)
        info_records = [r for r in caplog.records if r.levelname == "INFO"]
        assert len(info_records) == 1
        assert "Starting add_session_to_memory" in info_records[0].message

        # Test case 2: ValueError (WARNING level)
        caplog.set_level(logging.WARNING)
        caplog.clear()

        await add_session_to_memory(
            as_callback_context(mock_memory_callback_context_no_service)
        )

        warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warning_records) == 1
        assert (
            "Cannot add session to memory: memory service is not available."
            in warning_records[0].message
        )

    @pytest.mark.asyncio
    async def test_add_session_to_memory_returns_none(
        self,
        mock_memory_callback_context: MockMemoryCallbackContext,
    ) -> None:
        """Test that callback always returns None."""
        await add_session_to_memory(as_callback_context(mock_memory_callback_context))

    @pytest.mark.asyncio
    async def test_add_session_to_memory_multiple_calls(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that callback can be called multiple times."""
        from conftest import MockMemoryCallbackContext

        caplog.set_level(logging.INFO)

        # Create multiple contexts
        ctx1 = MockMemoryCallbackContext()
        ctx2 = MockMemoryCallbackContext()

        # Execute callbacks
        await add_session_to_memory(as_callback_context(ctx1))
        await add_session_to_memory(as_callback_context(ctx2))

        # Verify both completed successfully
        assert ctx1.add_session_to_memory_called
        assert ctx2.add_session_to_memory_called

        # Verify both were logged
        info_records = [r for r in caplog.records if r.levelname == "INFO"]
        assert len(info_records) == 2


class MockMemoriesCallbackContext:
    """Mock CallbackContext for add_memories_to_memory testing."""

    def __init__(self, state: MockState | None = None) -> None:
        """Initialize mock callback context."""
        self.state = state


class MockLlmRequestWithContents:
    """Mock LlmRequest with role-based contents for memory injection tests."""

    def __init__(self, contents: list[Any] | None = None) -> None:
        """Initialize mock LLM request."""
        self.contents = contents if contents is not None else []


class MockPart:
    """Mock Part with text attribute."""

    def __init__(self, text: str | None = None) -> None:
        """Initialize mock part."""
        self.text = text


class MockContentWithRole:
    """Mock Content with role and parts attributes."""

    def __init__(self, role: str, parts: list[Any] | None = None) -> None:
        """Initialize mock content with role."""
        self.role = role
        self.parts = parts if parts is not None else []


class TestAddMemoriesToContext:
    """Tests for the add_memories_to_context callback function."""

    @pytest.mark.asyncio
    async def test_skips_when_mem0_disabled(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that callback skips when mem0 is not enabled."""
        caplog.set_level(logging.DEBUG)

        with patch("agent.callbacks.is_mem0_enabled", return_value=False):
            context = MockMemoriesCallbackContext()
            request = MockLlmRequestWithContents()

            await add_memories_to_context(
                cast(CallbackContext, context), cast(Any, request)
            )

            assert "mem0 not enabled, skipping memory injection" in caplog.text

    @pytest.mark.asyncio
    async def test_skips_when_no_user_message(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that callback skips when no user message found in request."""
        caplog.set_level(logging.DEBUG)

        # Create request without user message
        request = MockLlmRequestWithContents(
            contents=[
                MockContentWithRole(role="system", parts=[MockPart("system prompt")]),
                MockContentWithRole(role="assistant", parts=[MockPart("hello")]),
            ]
        )

        with patch("agent.callbacks.is_mem0_enabled", return_value=True):
            context = MockMemoriesCallbackContext()
            await add_memories_to_context(
                cast(CallbackContext, context), cast(Any, request)
            )

            assert "No user message found, skipping memory injection" in caplog.text

    @pytest.mark.asyncio
    async def test_injects_memories_found(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that callback injects memories when found."""
        caplog.set_level(logging.INFO)

        # Create request with user message
        request = MockLlmRequestWithContents(
            contents=[
                MockContentWithRole(
                    role="user", parts=[MockPart("What do you know about me?")]
                ),
            ]
        )

        mock_manager = MagicMock()
        mock_manager.search_memory.return_value = {
            "memories": [
                {"memory": "User likes Python"},
                {"memory": "User prefers dark mode"},
            ]
        }

        with (
            patch("agent.callbacks.is_mem0_enabled", return_value=True),
            patch("agent.callbacks.get_mem0_manager", return_value=mock_manager),
            patch.dict("os.environ", {"MEM0_SEARCH_LIMIT": "5"}),
        ):
            context = MockMemoriesCallbackContext(
                state=MockState({"user_id": "test_user"})
            )
            await add_memories_to_context(
                cast(CallbackContext, context), cast(Any, request)
            )

            assert len(request.contents) == 2  # Original + injected memory
            # Check the injected content is at the beginning
            injected = request.contents[0]
            assert isinstance(injected, types.Content)
            assert injected.role == "user"
            assert injected.parts is not None
            assert len(injected.parts) > 0
            parts_text = injected.parts[0].text
            assert parts_text is not None
            assert "Context from memory" in parts_text
            assert "User likes Python" in parts_text
            assert "Injected 2 memories" in caplog.text

    @pytest.mark.asyncio
    async def test_skips_when_no_memories_found(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that callback skips when no memories found."""
        caplog.set_level(logging.DEBUG)

        request = MockLlmRequestWithContents(
            contents=[
                MockContentWithRole(role="user", parts=[MockPart("Hello")]),
            ]
        )

        mock_manager = MagicMock()
        mock_manager.search_memory.return_value = {"memories": []}

        with (
            patch("agent.callbacks.is_mem0_enabled", return_value=True),
            patch("agent.callbacks.get_mem0_manager", return_value=mock_manager),
        ):
            context = MockMemoriesCallbackContext()
            await add_memories_to_context(
                cast(CallbackContext, context), cast(Any, request)
            )

            assert len(request.contents) == 1  # Only original content
            assert "No relevant memories found" in caplog.text

    @pytest.mark.asyncio
    async def test_handles_exception_gracefully(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that callback handles exceptions gracefully."""
        caplog.set_level(logging.WARNING)

        request = MockLlmRequestWithContents(
            contents=[
                MockContentWithRole(role="user", parts=[MockPart("Hello")]),
            ]
        )

        mock_manager = MagicMock()
        mock_manager.search_memory.side_effect = RuntimeError(
            "Database connection failed"
        )

        with (
            patch("agent.callbacks.is_mem0_enabled", return_value=True),
            patch("agent.callbacks.get_mem0_manager", return_value=mock_manager),
        ):
            context = MockMemoriesCallbackContext()
            await add_memories_to_context(
                cast(CallbackContext, context), cast(Any, request)
            )

            assert "Failed to inject memories into context" in caplog.text

    @pytest.mark.asyncio
    async def test_uses_custom_search_limit(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that callback uses MEM0_SEARCH_LIMIT from environment."""
        caplog.set_level(logging.INFO)

        request = MockLlmRequestWithContents(
            contents=[
                MockContentWithRole(role="user", parts=[MockPart("Hello")]),
            ]
        )

        mock_manager = MagicMock()
        mock_manager.search_memory.return_value = {"memories": [{"memory": "test"}]}

        with (
            patch("agent.callbacks.is_mem0_enabled", return_value=True),
            patch("agent.callbacks.get_mem0_manager", return_value=mock_manager),
            patch.dict("os.environ", {"MEM0_SEARCH_LIMIT": "10"}),
        ):
            context = MockMemoriesCallbackContext()
            await add_memories_to_context(
                cast(CallbackContext, context), cast(Any, request)
            )

            # Verify search_memory was called with limit=10
            mock_manager.search_memory.assert_called_once()
            call_kwargs = mock_manager.search_memory.call_args[1]
            assert call_kwargs["limit"] == 10

    @pytest.mark.asyncio
    async def test_extracts_first_user_message_from_multiple(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test extraction of user message when multiple user messages exist."""
        caplog.set_level(logging.INFO)

        request = MockLlmRequestWithContents(
            contents=[
                MockContentWithRole(role="user", parts=[MockPart("First message")]),
                MockContentWithRole(role="assistant", parts=[MockPart("Response")]),
                MockContentWithRole(role="user", parts=[MockPart("Second message")]),
            ]
        )

        mock_manager = MagicMock()
        mock_manager.search_memory.return_value = {"memories": [{"memory": "test"}]}

        with (
            patch("agent.callbacks.is_mem0_enabled", return_value=True),
            patch("agent.callbacks.get_mem0_manager", return_value=mock_manager),
        ):
            context = MockMemoriesCallbackContext()
            await add_memories_to_context(
                cast(CallbackContext, context), cast(Any, request)
            )

            # Should use the second user message (reversed iteration picks first text)
            mock_manager.search_memory.assert_called_once()
            call_kwargs = mock_manager.search_memory.call_args[1]
            # The reversed iteration means "Second message" is seen first
            assert call_kwargs["query"] == "Second message"

    @pytest.mark.asyncio
    async def test_handles_user_id_from_state(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that callback extracts user_id from state if available."""
        caplog.set_level(logging.INFO)

        request = MockLlmRequestWithContents(
            contents=[
                MockContentWithRole(role="user", parts=[MockPart("Hello")]),
            ]
        )

        mock_manager = MagicMock()
        mock_manager.search_memory.return_value = {"memories": [{"memory": "test"}]}

        with (
            patch("agent.callbacks.is_mem0_enabled", return_value=True),
            patch("agent.callbacks.get_mem0_manager", return_value=mock_manager),
        ):
            context = MockMemoriesCallbackContext(
                state=MockState({"user_id": "user_abc"})
            )
            await add_memories_to_context(
                cast(CallbackContext, context), cast(Any, request)
            )

            # Verify user_id was passed
            call_kwargs = mock_manager.search_memory.call_args[1]
            assert call_kwargs["user_id"] == "user_abc"

    @pytest.mark.asyncio
    async def test_handles_part_with_no_text(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that callback handles parts with no text attribute."""
        caplog.set_level(logging.DEBUG)

        # Create part without text
        part_no_text = MagicMock()
        part_no_text.text = None

        request = MockLlmRequestWithContents(
            contents=[
                MockContentWithRole(role="user", parts=[part_no_text]),
                MockContentWithRole(role="user", parts=[MockPart("Valid text")]),
            ]
        )

        mock_manager = MagicMock()
        mock_manager.search_memory.return_value = {"memories": [{"memory": "test"}]}

        with (
            patch("agent.callbacks.is_mem0_enabled", return_value=True),
            patch("agent.callbacks.get_mem0_manager", return_value=mock_manager),
        ):
            context = MockMemoriesCallbackContext()
            await add_memories_to_context(
                cast(CallbackContext, context), cast(Any, request)
            )

            # Should find the valid text message
            call_kwargs = mock_manager.search_memory.call_args[1]
            assert call_kwargs["query"] == "Valid text"

    @pytest.mark.asyncio
    async def test_formats_memories_correctly(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that memories are formatted correctly in injected content."""
        caplog.set_level(logging.INFO)

        request = MockLlmRequestWithContents(
            contents=[
                MockContentWithRole(role="user", parts=[MockPart("Hello")]),
            ]
        )

        mock_manager = MagicMock()
        mock_manager.search_memory.return_value = {
            "memories": [
                {"memory": "Fact 1"},
                {"memory": "Fact 2"},
                {"memory": "Fact 3"},
            ]
        }

        with (
            patch("agent.callbacks.is_mem0_enabled", return_value=True),
            patch("agent.callbacks.get_mem0_manager", return_value=mock_manager),
        ):
            context = MockMemoriesCallbackContext()
            await add_memories_to_context(
                cast(CallbackContext, context), cast(Any, request)
            )

            # Check memory formatting
            injected = request.contents[0]
            text = injected.parts[0].text
            assert "- Fact 1" in text
            assert "- Fact 2" in text
            assert "- Fact 3" in text

    @pytest.mark.asyncio
    async def test_handles_empty_parts_in_content(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that callback handles content with empty parts list."""
        caplog.set_level(logging.DEBUG)

        request = MockLlmRequestWithContents(
            contents=[
                MockContentWithRole(role="user", parts=[]),  # Empty parts
                MockContentWithRole(role="user", parts=[MockPart("Valid message")]),
            ]
        )

        mock_manager = MagicMock()
        mock_manager.search_memory.return_value = {"memories": [{"memory": "test"}]}

        with (
            patch("agent.callbacks.is_mem0_enabled", return_value=True),
            patch("agent.callbacks.get_mem0_manager", return_value=mock_manager),
        ):
            context = MockMemoriesCallbackContext()
            await add_memories_to_context(
                cast(CallbackContext, context), cast(Any, request)
            )

            # Should find the valid message
            call_kwargs = mock_manager.search_memory.call_args[1]
            assert call_kwargs["query"] == "Valid message"

    @pytest.mark.asyncio
    async def test_handles_none_parts_in_content(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that callback handles content with None parts."""
        caplog.set_level(logging.DEBUG)

        content_with_none = MagicMock()
        content_with_none.role = "user"
        content_with_none.parts = None

        request = MockLlmRequestWithContents(
            contents=[
                content_with_none,
                MockContentWithRole(role="user", parts=[MockPart("Valid message")]),
            ]
        )

        mock_manager = MagicMock()
        mock_manager.search_memory.return_value = {"memories": [{"memory": "test"}]}

        with (
            patch("agent.callbacks.is_mem0_enabled", return_value=True),
            patch("agent.callbacks.get_mem0_manager", return_value=mock_manager),
        ):
            context = MockMemoriesCallbackContext()
            await add_memories_to_context(
                cast(CallbackContext, context), cast(Any, request)
            )

            # Should find the valid message
            call_kwargs = mock_manager.search_memory.call_args[1]
            assert call_kwargs["query"] == "Valid message"

    @pytest.mark.asyncio
    async def test_handles_part_with_empty_string_text(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test callback handles parts with empty string text before valid text."""
        caplog.set_level(logging.DEBUG)

        # Create part with empty string text
        part_empty = MagicMock()
        part_empty.text = ""

        request = MockLlmRequestWithContents(
            contents=[
                MockContentWithRole(
                    role="user", parts=[part_empty, MockPart("Valid text")]
                ),
            ]
        )

        mock_manager = MagicMock()
        mock_manager.search_memory.return_value = {"memories": [{"memory": "test"}]}

        with (
            patch("agent.callbacks.is_mem0_enabled", return_value=True),
            patch("agent.callbacks.get_mem0_manager", return_value=mock_manager),
        ):
            context = MockMemoriesCallbackContext()
            await add_memories_to_context(
                cast(CallbackContext, context), cast(Any, request)
            )

            # Should skip empty string and find the valid text message
            call_kwargs = mock_manager.search_memory.call_args[1]
            assert call_kwargs["query"] == "Valid text"

    @pytest.mark.asyncio
    async def test_handles_all_parts_without_truthy_text(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test callback handles user content where all parts have falsy text values."""
        caplog.set_level(logging.DEBUG)

        # Create parts with only falsy text values (None, empty string, 0, False)
        part_falsy1 = MagicMock()
        part_falsy1.text = None
        part_falsy2 = MagicMock()
        part_falsy2.text = ""
        part_falsy3 = MagicMock()
        part_falsy3.text = 0

        request = MockLlmRequestWithContents(
            contents=[
                MockContentWithRole(
                    role="user", parts=[part_falsy1, part_falsy2, part_falsy3]
                ),
                MockContentWithRole(role="assistant", parts=[MockPart("Hello")]),
                # Another user message with valid text should still be found
                MockContentWithRole(role="user", parts=[MockPart("Found me")]),
            ]
        )

        mock_manager = MagicMock()
        mock_manager.search_memory.return_value = {"memories": [{"memory": "test"}]}

        with (
            patch("agent.callbacks.is_mem0_enabled", return_value=True),
            patch("agent.callbacks.get_mem0_manager", return_value=mock_manager),
        ):
            context = MockMemoriesCallbackContext()
            await add_memories_to_context(
                cast(CallbackContext, context), cast(Any, request)
            )

            # Should find the text from the second user message
            call_kwargs = mock_manager.search_memory.call_args[1]
            assert call_kwargs["query"] == "Found me"

    @pytest.mark.asyncio
    async def test_handles_last_user_content_with_all_falsy_parts(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test callback handles last user content (first in reversed) with falsy."""
        caplog.set_level(logging.DEBUG)

        # Create parts with only falsy text values
        part_none = MagicMock()
        part_none.text = None
        part_empty = MagicMock()
        part_empty.text = ""

        request = MockLlmRequestWithContents(
            contents=[
                MockContentWithRole(role="user", parts=[MockPart("Valid text")]),
                MockContentWithRole(role="assistant", parts=[MockPart("Response")]),
                # This user content is LAST in the list, so FIRST in reversed iteration
                # All its parts have falsy text
                MockContentWithRole(role="user", parts=[part_none, part_empty]),
            ]
        )

        mock_manager = MagicMock()
        mock_manager.search_memory.return_value = {"memories": [{"memory": "test"}]}

        with (
            patch("agent.callbacks.is_mem0_enabled", return_value=True),
            patch("agent.callbacks.get_mem0_manager", return_value=mock_manager),
        ):
            context = MockMemoriesCallbackContext()
            await add_memories_to_context(
                cast(CallbackContext, context), cast(Any, request)
            )

            # Should find the valid text from the first user message
            call_kwargs = mock_manager.search_memory.call_args[1]
            assert call_kwargs["query"] == "Valid text"
