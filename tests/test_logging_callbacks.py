"""Unit tests for the LoggingCallbacks class in the agent.logging_callbacks module."""

import logging
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

# Import mock classes from conftest
from conftest import (
    MockBaseTool,
    MockContent,
    MockLlmRequest,
    MockLlmResponse,
    MockLoggingCallbackContext,
    MockState,
    MockToolContext,
)

from agent_foundation.callbacks import LoggingCallbacks

# Note: Custom mock classes (conftest.py) use duck typing to match ADK interfaces.


# Test fixtures
@pytest.fixture
def custom_logger() -> logging.Logger:
    """Create a custom logger for testing."""
    logger = logging.getLogger("test.custom.logger")
    logger.setLevel(logging.DEBUG)
    return logger


class TestLoggerInjection:
    """Tests for logger injection and initialization."""

    def test_logging_callbacks_default_logger(self) -> None:
        """Verify LoggingCallbacks creates a logger with correct module name."""
        callbacks = LoggingCallbacks()

        assert callbacks.logger is not None
        assert callbacks.logger.name == "agent_foundation.callbacks"

    def test_logging_callbacks_custom_logger(
        self, custom_logger: logging.Logger
    ) -> None:
        """Verify LoggingCallbacks uses injected logger instance."""
        callbacks = LoggingCallbacks(logger=custom_logger)

        assert callbacks.logger is custom_logger
        assert callbacks.logger.name == "test.custom.logger"

    def test_callbacks_with_custom_logger_logs_correctly(
        self,
        custom_logger: logging.Logger,
        mock_logging_callback_context: MockLoggingCallbackContext,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify custom logger receives all log messages from callback operations."""
        # Set caplog to capture from the custom logger
        caplog.set_level(logging.DEBUG, logger="test.custom.logger")

        callbacks = LoggingCallbacks(logger=custom_logger)

        # Test various callbacks
        callbacks.before_agent(mock_logging_callback_context)
        callbacks.after_agent(mock_logging_callback_context)

        # Verify logs were sent to custom logger
        custom_logger_records = [
            r for r in caplog.records if r.name == "test.custom.logger"
        ]

        assert len(custom_logger_records) > 0
        assert any("Starting agent" in r.message for r in custom_logger_records)
        assert any("Leaving agent" in r.message for r in custom_logger_records)


class TestAgentCallbacks:
    """Tests for agent lifecycle callbacks (before_agent and after_agent)."""

    def test_before_agent_with_full_context(
        self,
        mock_logging_callback_context: MockLoggingCallbackContext,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify before_agent logs agent name, ID, state, and user content properly."""
        caplog.set_level(logging.DEBUG)
        callbacks = LoggingCallbacks()

        result = callbacks.before_agent(mock_logging_callback_context)

        # Verify return value
        assert result is None

        # Verify INFO level logging
        assert (
            "*** Starting agent 'my_agent' with invocation_id 'inv-789' ***"
            in caplog.text
        )

        # Verify DEBUG level logging
        assert "State keys:" in caplog.text
        assert "User Content: {'text': 'Hello, agent!'}" in caplog.text

    def test_before_agent_without_user_content(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify before_agent skips user content logging when not provided."""
        caplog.set_level(logging.DEBUG)
        callbacks = LoggingCallbacks()
        context = MockLoggingCallbackContext(user_content=None)

        result = callbacks.before_agent(context)

        assert result is None
        assert "*** Starting agent 'test_agent'" in caplog.text
        assert "User Content:" not in caplog.text

    def test_after_agent_with_full_context(
        self,
        mock_logging_callback_context: MockLoggingCallbackContext,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify after_agent logs agent exit with ID, state, and user content."""
        caplog.set_level(logging.DEBUG)
        callbacks = LoggingCallbacks()

        result = callbacks.after_agent(mock_logging_callback_context)

        # Verify return value
        assert result is None

        # Verify INFO level logging
        assert (
            "*** Leaving agent 'my_agent' with invocation_id 'inv-789' ***"
            in caplog.text
        )

        # Verify DEBUG level logging
        assert "State keys:" in caplog.text
        assert "User Content: {'text': 'Hello, agent!'}" in caplog.text

    def test_after_agent_without_user_content(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify after_agent skips user content logging when not provided."""
        caplog.set_level(logging.DEBUG)
        callbacks = LoggingCallbacks()
        context = MockLoggingCallbackContext(user_content=None)

        result = callbacks.after_agent(context)

        assert result is None
        assert "*** Leaving agent 'test_agent'" in caplog.text
        assert "User Content:" not in caplog.text


class TestModelCallbacks:
    """Tests for model/LLM callbacks (before_model and after_model)."""

    def test_before_model_with_multiple_messages(
        self,
        mock_logging_callback_context: MockLoggingCallbackContext,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify before_model logs all LLM request messages with content details."""
        caplog.set_level(logging.DEBUG)
        callbacks = LoggingCallbacks()

        llm_request = MockLlmRequest(
            contents=[
                MockContent({"role": "system", "text": "You are a helpful assistant"}),
                MockContent({"role": "user", "text": "What is 2+2?"}),
                MockContent({"role": "assistant", "text": "2+2 equals 4"}),
            ]
        )

        result = callbacks.before_model(mock_logging_callback_context, llm_request)

        assert result is None
        assert "*** Before LLM call for agent 'my_agent'" in caplog.text
        assert "LLM request contains 3 messages:" in caplog.text
        assert "Content 1: {'role': 'system'" in caplog.text
        assert "Content 2: {'role': 'user'" in caplog.text
        assert "Content 3: {'role': 'assistant'" in caplog.text

    def test_before_model_without_user_content(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify before_model skips user content logging while logging requests."""
        caplog.set_level(logging.DEBUG)
        callbacks = LoggingCallbacks()
        context = MockLoggingCallbackContext(user_content=None)
        llm_request = MockLlmRequest()

        result = callbacks.before_model(context, llm_request)

        assert result is None
        assert "*** Before LLM call" in caplog.text
        assert "User Content:" not in caplog.text
        assert "LLM request contains 2 messages:" in caplog.text

    def test_after_model_with_response(
        self,
        mock_logging_callback_context: MockLoggingCallbackContext,
        mock_llm_response: MockLlmResponse,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify after_model logs LLM response content at DEBUG level."""
        caplog.set_level(logging.DEBUG)
        callbacks = LoggingCallbacks()

        result = callbacks.after_model(mock_logging_callback_context, mock_llm_response)

        assert result is None
        assert "*** After LLM call for agent 'my_agent'" in caplog.text
        assert (
            "LLM response: {'text': 'The answer is 42', 'confidence': 0.95}"
            in caplog.text
        )

    def test_after_model_without_llm_content(
        self,
        mock_logging_callback_context: MockLoggingCallbackContext,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify after_model skips response logging when LLM content is absent."""
        caplog.set_level(logging.DEBUG)
        callbacks = LoggingCallbacks()

        llm_response = MockLlmResponse(content=None)

        result = callbacks.after_model(mock_logging_callback_context, llm_response)

        assert result is None
        assert "*** After LLM call" in caplog.text
        assert "LLM response:" not in caplog.text


class TestToolCallbacks:
    """Tests for tool invocation callbacks (before_tool and after_tool)."""

    def test_before_tool_with_arguments(
        self,
        mock_tool_context: MockToolContext,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify before_tool logs tool name, arguments, and event actions at DEBUG."""
        caplog.set_level(logging.DEBUG)
        callbacks = LoggingCallbacks()

        tool = MockBaseTool(name="calculator")
        args = {"operation": "add", "x": 5, "y": 3}

        result = callbacks.before_tool(tool, args, mock_tool_context)

        assert result is None
        assert (
            "*** Before invoking tool 'calculator' in agent 'tool_agent'" in caplog.text
        )
        assert "args: {'operation': 'add', 'x': 5, 'y': 3}" in caplog.text
        assert (
            "EventActions: {'action': 'run', 'params': ['arg1', 'arg2']}" in caplog.text
        )

    def test_before_tool_without_user_content(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify before_tool skips user content logging while logging tool details."""
        caplog.set_level(logging.DEBUG)
        callbacks = LoggingCallbacks()

        tool = MockBaseTool()
        args = {"param": "value"}
        context = MockToolContext(user_content=None)

        result = callbacks.before_tool(tool, args, context)

        assert result is None
        assert "*** Before invoking tool 'test_tool'" in caplog.text
        assert "User Content:" not in caplog.text

    def test_after_tool_with_response(
        self,
        mock_tool_context: MockToolContext,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify after_tool logs tool name, arguments, and response data properly."""
        caplog.set_level(logging.DEBUG)
        callbacks = LoggingCallbacks()

        tool = MockBaseTool(name="database_query")
        args = {"query": "SELECT * FROM users"}
        tool_response = {"rows": [{"id": 1, "name": "Alice"}], "count": 1}

        result = callbacks.after_tool(tool, args, mock_tool_context, tool_response)

        assert result is None
        assert "*** After invoking tool 'database_query'" in caplog.text
        assert "args: {'query': 'SELECT * FROM users'}" in caplog.text
        assert (
            "Tool response: {'rows': [{'id': 1, 'name': 'Alice'}], 'count': 1}"
            in caplog.text
        )

    def test_after_tool_without_user_content(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify after_tool skips user content logging while logging tool response."""
        caplog.set_level(logging.DEBUG)
        callbacks = LoggingCallbacks()

        tool = MockBaseTool()
        args: dict[str, Any] = {}
        context = MockToolContext(user_content=None)
        tool_response = {"status": "success"}

        result = callbacks.after_tool(tool, args, context, tool_response)

        assert result is None
        assert "*** After invoking tool 'test_tool'" in caplog.text
        assert "User Content:" not in caplog.text
        assert "Tool response: {'status': 'success'}" in caplog.text


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_callbacks_with_empty_state(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify callbacks correctly log empty state dictionaries at DEBUG level."""
        caplog.set_level(logging.DEBUG)
        callbacks = LoggingCallbacks()

        context = MockLoggingCallbackContext(state=MockState({}))

        # Test before_agent with empty state
        result = callbacks.before_agent(context)
        assert result is None
        assert "State keys: dict_keys([])" in caplog.text

    def test_callbacks_with_complex_nested_state(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify callbacks log state keys for complex nested state."""
        caplog.set_level(logging.DEBUG)
        callbacks = LoggingCallbacks()

        complex_state = MockState(
            {
                "user": {
                    "id": "123",
                    "profile": {
                        "name": "Test User",
                        "settings": {
                            "theme": "dark",
                            "notifications": True,
                        },
                    },
                },
                "session": {
                    "tokens": ["token1", "token2"],
                    "metadata": None,
                },
            }
        )

        context = MockLoggingCallbackContext(state=complex_state)

        result = callbacks.before_agent(context)
        assert result is None
        # Now we only log state keys, not the actual values
        assert "State keys: dict_keys(['user', 'session'])" in caplog.text

    def test_logging_levels(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify callbacks emit INFO logs for main events and DEBUG for details."""
        callbacks = LoggingCallbacks()
        context = MockLoggingCallbackContext()

        # Test INFO level (should show main events)
        caplog.set_level(logging.INFO)
        caplog.clear()

        callbacks.before_agent(context)
        info_records = [r for r in caplog.records if r.levelname == "INFO"]
        debug_records = [r for r in caplog.records if r.levelname == "DEBUG"]

        assert len(info_records) == 1
        assert "Starting agent" in info_records[0].message
        assert len(debug_records) == 0  # DEBUG not captured at INFO level

        # Test DEBUG level (should show details)
        caplog.set_level(logging.DEBUG)
        caplog.clear()

        callbacks.before_agent(context)
        info_records = [r for r in caplog.records if r.levelname == "INFO"]
        debug_records = [r for r in caplog.records if r.levelname == "DEBUG"]

        assert len(info_records) == 1
        assert len(debug_records) >= 1  # At least state should be logged
        assert any("State keys:" in r.message for r in debug_records)

    def test_model_dump_serialization(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify callbacks serialize content using model_dump with proper params."""
        caplog.set_level(logging.DEBUG)
        callbacks = LoggingCallbacks()

        # Create mock content with spy on model_dump
        mock_content = Mock(spec=MockContent)
        mock_content.model_dump = MagicMock(return_value={"mocked": "data"})

        context = MockLoggingCallbackContext(user_content=mock_content)

        callbacks.before_agent(context)

        # Verify model_dump was called with correct parameters
        mock_content.model_dump.assert_called_once_with(exclude_none=True, mode="json")
        assert "User Content: {'mocked': 'data'}" in caplog.text

    def test_all_callbacks_return_none(
        self,
        mock_llm_request: MockLlmRequest,
        mock_llm_response: MockLlmResponse,
        mock_base_tool: MockBaseTool,
    ) -> None:
        """Verify all callback methods return None to allow normal agent flow."""
        callbacks = LoggingCallbacks()

        # Create test objects
        callback_context = MockLoggingCallbackContext()
        tool_context = MockToolContext()
        args = {"test": "arg"}
        tool_response = {"result": "success"}

        # Test all callbacks return None
        assert callbacks.before_agent(callback_context) is None
        assert callbacks.after_agent(callback_context) is None
        assert callbacks.before_model(callback_context, mock_llm_request) is None
        assert callbacks.after_model(callback_context, mock_llm_response) is None
        assert callbacks.before_tool(mock_base_tool, args, tool_context) is None
        assert (
            callbacks.after_tool(mock_base_tool, args, tool_context, tool_response)
            is None
        )


class TestWalrusOperators:
    """Tests for walrus operator usage in logging callbacks."""

    def test_walrus_operator_assignment_and_usage(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify walrus operators correctly assign AND use values in callbacks.

        This test explicitly validates that the walrus operator (:=) both assigns
        and evaluates the value, preventing regressions if refactored.

        Tests all 7 walrus operator usages in LoggingCallbacks:
        1. before_agent (line 80): user_content assignment
        2. after_agent (line 103): user_content assignment
        3. before_model (line 132): user_content assignment
        4. after_model (line 166): user_content assignment
        5. after_model (line 170): llm_content assignment
        6. before_tool (line 201): user_content assignment (as 'content')
        7. after_tool (line 239): user_content assignment (as 'content')
        """
        caplog.set_level(logging.DEBUG)
        callbacks = LoggingCallbacks()

        # Test 1: before_agent walrus operator with unique content
        unique_content_before_agent = MockContent(
            {
                "unique_walrus_marker": "test_walrus_12345_before_agent",
                "test_type": "walrus_operator_validation",
            }
        )
        context_before_agent = MockLoggingCallbackContext(
            agent_name="walrus_test_agent", user_content=unique_content_before_agent
        )

        caplog.clear()
        callbacks.before_agent(context_before_agent)

        # Verify the exact unique content was assigned and used
        assert "unique_walrus_marker" in caplog.text
        assert "test_walrus_12345_before_agent" in caplog.text
        assert "test_type" in caplog.text
        assert "walrus_operator_validation" in caplog.text

        # Test 2: after_agent walrus operator with unique content
        unique_content_after_agent = MockContent(
            {
                "unique_walrus_marker": "test_walrus_67890_after_agent",
                "validation_id": "after_agent_walrus",
            }
        )
        context_after_agent = MockLoggingCallbackContext(
            agent_name="walrus_test_agent", user_content=unique_content_after_agent
        )

        caplog.clear()
        callbacks.after_agent(context_after_agent)

        assert "unique_walrus_marker" in caplog.text
        assert "test_walrus_67890_after_agent" in caplog.text
        assert "validation_id" in caplog.text
        assert "after_agent_walrus" in caplog.text

        # Test 3: before_model walrus operator with unique content
        unique_content_before_model = MockContent(
            {
                "unique_walrus_marker": "test_walrus_11111_before_model",
                "model_test": "before_model_walrus_check",
            }
        )
        context_before_model = MockLoggingCallbackContext(
            agent_name="walrus_test_agent", user_content=unique_content_before_model
        )
        llm_request = MockLlmRequest()

        caplog.clear()
        callbacks.before_model(context_before_model, llm_request)

        assert "unique_walrus_marker" in caplog.text
        assert "test_walrus_11111_before_model" in caplog.text
        assert "model_test" in caplog.text
        assert "before_model_walrus_check" in caplog.text

        # Test 4: after_model walrus operator with user_content
        unique_content_after_model = MockContent(
            {
                "unique_walrus_marker": "test_walrus_22222_after_model_user",
                "after_model_user": "walrus_validation",
            }
        )
        context_after_model = MockLoggingCallbackContext(
            agent_name="walrus_test_agent", user_content=unique_content_after_model
        )

        # Test 5: after_model walrus operator with llm_content
        unique_llm_content = MockContent(
            {
                "unique_llm_marker": "test_walrus_33333_llm_response",
                "llm_response_id": "walrus_llm_test",
            }
        )
        llm_response = MockLlmResponse(content=unique_llm_content)

        caplog.clear()
        callbacks.after_model(context_after_model, llm_response)

        # Verify both user_content and llm_content walrus operators worked
        assert "unique_walrus_marker" in caplog.text
        assert "test_walrus_22222_after_model_user" in caplog.text
        assert "after_model_user" in caplog.text
        assert "walrus_validation" in caplog.text

        assert "unique_llm_marker" in caplog.text
        assert "test_walrus_33333_llm_response" in caplog.text
        assert "llm_response_id" in caplog.text
        assert "walrus_llm_test" in caplog.text

        # Test 6: before_tool walrus operator (uses 'content' variable name)
        unique_content_before_tool = MockContent(
            {
                "unique_walrus_marker": "test_walrus_44444_before_tool",
                "tool_context_id": "before_tool_walrus",
            }
        )
        tool_context_before = MockToolContext(
            agent_name="walrus_tool_agent", user_content=unique_content_before_tool
        )
        tool = MockBaseTool(name="walrus_test_tool")
        args = {"test": "walrus"}

        caplog.clear()
        callbacks.before_tool(tool, args, tool_context_before)

        assert "unique_walrus_marker" in caplog.text
        assert "test_walrus_44444_before_tool" in caplog.text
        assert "tool_context_id" in caplog.text
        assert "before_tool_walrus" in caplog.text

        # Test 7: after_tool walrus operator (uses 'content' variable name)
        unique_content_after_tool = MockContent(
            {
                "unique_walrus_marker": "test_walrus_55555_after_tool",
                "after_tool_validation": "walrus_complete",
            }
        )
        tool_context_after = MockToolContext(
            agent_name="walrus_tool_agent", user_content=unique_content_after_tool
        )
        tool_response = {"result": "walrus_test_success"}

        caplog.clear()
        callbacks.after_tool(tool, args, tool_context_after, tool_response)

        assert "unique_walrus_marker" in caplog.text
        assert "test_walrus_55555_after_tool" in caplog.text
        assert "after_tool_validation" in caplog.text
        assert "walrus_complete" in caplog.text

        # Verify tool response was logged (not a walrus operator, but part of method)
        assert "walrus_test_success" in caplog.text

    def test_walrus_operator_with_none_values(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify walrus operators correctly handle None values.

        This test ensures that when the walrus operator evaluates to None/falsy,
        the conditional block is properly skipped and no logging occurs.
        """
        caplog.set_level(logging.DEBUG)
        callbacks = LoggingCallbacks()

        # Create contexts with None values for user_content
        context_none = MockLoggingCallbackContext(
            agent_name="none_test_agent",
            user_content=None,  # This should cause walrus operator to skip the block
        )
        tool_context_none = MockToolContext(
            agent_name="none_tool_agent", user_content=None
        )
        llm_response_none = MockLlmResponse(content=None)  # None llm_content

        # Test before_agent with None user_content
        caplog.clear()
        callbacks.before_agent(context_none)
        assert "User Content:" not in caplog.text

        # Test after_agent with None user_content
        caplog.clear()
        callbacks.after_agent(context_none)
        assert "User Content:" not in caplog.text

        # Test before_model with None user_content
        caplog.clear()
        callbacks.before_model(context_none, MockLlmRequest())
        assert "User Content:" not in caplog.text

        # Test after_model with None user_content and None llm_content
        caplog.clear()
        callbacks.after_model(context_none, llm_response_none)
        assert "User Content:" not in caplog.text
        assert "LLM response:" not in caplog.text

        # Test before_tool with None user_content
        caplog.clear()
        callbacks.before_tool(MockBaseTool(), {}, tool_context_none)
        assert "User Content:" not in caplog.text

        # Test after_tool with None user_content
        caplog.clear()
        callbacks.after_tool(MockBaseTool(), {}, tool_context_none, {})
        assert "User Content:" not in caplog.text
