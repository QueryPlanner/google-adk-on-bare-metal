"""Unit tests for custom tools."""

import logging

import pytest

# Import mock classes from conftest
from conftest import MockState, MockToolContext

from agent_foundation.tools import example_tool


class TestExampleTool:
    """Tests for the example_tool function."""

    def test_example_tool_returns_success(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that example_tool returns success status and message."""
        # Setup logging to capture INFO level
        caplog.set_level(logging.INFO)

        # Create mock tool context with state
        state = MockState({"user_id": "test_user", "session_key": "value"})
        tool_context = MockToolContext(state=state)

        # Execute tool
        result = example_tool(tool_context)

        # Verify return value
        assert result["status"] == "success"
        assert result["message"] == "Successfully used example_tool."

    def test_example_tool_logs_state_keys(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that example_tool logs session state keys."""
        # Setup logging to capture INFO level
        caplog.set_level(logging.INFO)

        # Create mock tool context with state
        state = MockState({"key1": "value1", "key2": "value2"})
        tool_context = MockToolContext(state=state)

        # Execute tool
        example_tool(tool_context)

        # Verify logging
        assert "Session state keys:" in caplog.text
        assert "Successfully used example_tool." in caplog.text

        # Verify INFO level was used
        info_records = [r for r in caplog.records if r.levelname == "INFO"]
        assert len(info_records) == 2

    def test_example_tool_with_empty_state(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that example_tool handles empty state correctly."""
        # Setup logging
        caplog.set_level(logging.INFO)

        # Create mock tool context with empty state
        state = MockState({})
        tool_context = MockToolContext(state=state)

        # Execute tool
        result = example_tool(tool_context)

        # Verify success even with empty state
        assert result["status"] == "success"
        assert result["message"] == "Successfully used example_tool."

        # Verify logging occurred
        assert "Session state keys:" in caplog.text
