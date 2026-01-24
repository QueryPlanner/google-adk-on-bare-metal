"""Shared pytest fixtures for all tests."""

from collections.abc import Callable, Generator
from contextlib import AbstractContextManager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ADK Callback Mock Objects for testing callbacks
class MockState:
    """Mock State object for ADK callback testing.

    Supports both dictionary-style access and to_dict() method
    to match ADK's state interface.
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        """Initialize mock state with optional data."""
        self._data = data if data is not None else {}

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary."""
        return self._data

    def get(self, key: str, default: Any = None) -> Any:
        """Get item from state with optional default."""
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Get item using dictionary syntax."""
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item using dictionary syntax."""
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists in state."""
        return key in self._data


class MockContent:
    """Mock Content object for ADK callback testing.

    Used for user_content and llm_content in callbacks.
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        """Initialize mock content with optional data."""
        self._data = data if data is not None else {"text": "test content"}

    def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
        """Serialize content to dictionary."""
        return self._data


class MockSession:
    """Mock ADK Session for testing.

    Minimal mock used by MockReadonlyContext.
    """

    def __init__(self, user_id: str = "test_user_123") -> None:
        """Initialize mock session with user_id."""
        self.user_id = user_id


class MockMemoryCallbackContext:
    """Minimal mock CallbackContext for add_session_to_memory callback testing.

    Controls behavior through constructor parameters instead of rebuilding
    ADK's internal logic. This keeps tests independent of ADK implementation.
    """

    def __init__(
        self,
        should_raise: type[Exception] | None = None,
        error_message: str = "",
    ) -> None:
        """Initialize mock callback context with controlled behavior.

        Args:
            should_raise: Exception type to raise when add_session_to_memory is called.
                         None means the call succeeds.
            error_message: Message for the exception if should_raise is set.
        """
        self._should_raise = should_raise
        self._error_message = error_message
        self.add_session_to_memory_called = False

    async def add_session_to_memory(self) -> None:
        """Mock implementation that either succeeds or raises controlled exception.

        Raises:
            Exception: The exception type configured in __init__ if should_raise is set.
        """
        self.add_session_to_memory_called = True
        if self._should_raise:
            raise self._should_raise(self._error_message)


class MockLoggingCallbackContext:
    """Mock CallbackContext for LoggingCallbacks testing.

    Used for agent and model callbacks testing.
    """

    def __init__(
        self,
        agent_name: str = "test_agent",
        invocation_id: str = "test-invocation-123",
        state: MockState | None = None,
        user_content: MockContent | None = None,
    ) -> None:
        """Initialize mock callback context for logging callbacks."""
        self.agent_name = agent_name
        self.invocation_id = invocation_id
        self.state = state if state is not None else MockState()
        self.user_content = user_content


class MockLlmRequest:
    """Mock LlmRequest for model callbacks."""

    def __init__(self, contents: list[MockContent] | None = None) -> None:
        """Initialize mock LLM request."""
        if contents is None:
            contents = [
                MockContent({"text": "system prompt"}),
                MockContent({"text": "user message"}),
            ]
        self.contents = contents


class MockLlmResponse:
    """Mock LlmResponse for model callbacks."""

    def __init__(self, content: MockContent | None = None) -> None:
        """Initialize mock LLM response."""
        self.content = content


class MockEventActions:
    """Mock EventActions for tool callbacks."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        """Initialize mock event actions."""
        self._data = data if data is not None else {"action": "execute"}

    def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
        """Serialize actions to dictionary."""
        return self._data


class MockToolContext:
    """Mock ToolContext for tool callbacks."""

    def __init__(
        self,
        agent_name: str = "test_agent",
        invocation_id: str = "test-invocation-456",
        state: MockState | None = None,
        user_content: MockContent | None = None,
        actions: MockEventActions | None = None,
    ) -> None:
        """Initialize mock tool context."""
        self.agent_name = agent_name
        self.invocation_id = invocation_id
        self.state = state if state is not None else MockState()
        self.user_content = user_content
        self.actions = actions if actions is not None else MockEventActions()


class MockBaseTool:
    """Mock BaseTool for tool callbacks."""

    def __init__(self, name: str = "test_tool") -> None:
        """Initialize mock tool."""
        self.name = name


class MockReadonlyContext:
    """Mock ReadonlyContext for testing InstructionProvider functions.

    Provides read-only access to invocation metadata and session state,
    matching the interface of google.adk.agents.readonly_context.ReadonlyContext.
    """

    def __init__(
        self,
        agent_name: str = "test_agent",
        invocation_id: str = "test-inv-readonly",
        state: dict[str, Any] | None = None,
        user_content: MockContent | None = None,
        session: MockSession | None = None,
    ) -> None:
        """Initialize mock readonly context.

        Args:
            agent_name: Name of the agent.
            invocation_id: ID of the current invocation.
            state: Session state dictionary (read-only).
            user_content: Optional user content that started the invocation.
            session: Optional session object.
        """
        self._agent_name = agent_name
        self._invocation_id = invocation_id
        self._state = state if state is not None else {}
        self._user_content = user_content
        self._session = session if session is not None else MockSession()

    @property
    def agent_name(self) -> str:
        """The name of the agent that is currently running."""
        return self._agent_name

    @property
    def invocation_id(self) -> str:
        """The current invocation id."""
        return self._invocation_id

    @property
    def state(self) -> dict[str, Any]:
        """The state of the current session (read-only)."""
        return self._state.copy()  # Return a copy to enforce read-only

    @property
    def user_content(self) -> MockContent | None:
        """The user content that started this invocation."""
        return self._user_content

    @property
    def session(self) -> MockSession:
        """The current session for this invocation."""
        return self._session


# Fixtures for ADK callback testing
@pytest.fixture
def mock_state() -> MockState:
    """Create a mock state with test data."""
    return MockState({"user_id": "user123", "session_data": {"key": "value"}})


@pytest.fixture
def mock_content() -> MockContent:
    """Create a mock content with test data."""
    return MockContent({"text": "Hello, agent!"})


@pytest.fixture
def mock_logging_callback_context(
    mock_state: MockState, mock_content: MockContent
) -> MockLoggingCallbackContext:
    """Create a mock logging callback context with full data."""
    return MockLoggingCallbackContext(
        agent_name="my_agent",
        invocation_id="inv-789",
        state=mock_state,
        user_content=mock_content,
    )


@pytest.fixture
def mock_llm_request() -> MockLlmRequest:
    """Create a mock LLM request with default messages."""
    return MockLlmRequest(
        contents=[
            MockContent({"text": "system prompt"}),
            MockContent({"text": "user message"}),
        ]
    )


@pytest.fixture
def mock_llm_response() -> MockLlmResponse:
    """Create a mock LLM response with content."""
    return MockLlmResponse(
        content=MockContent({"text": "The answer is 42", "confidence": 0.95})
    )


@pytest.fixture
def mock_event_actions() -> MockEventActions:
    """Create mock event actions with test data."""
    return MockEventActions({"action": "run", "params": ["arg1", "arg2"]})


@pytest.fixture
def mock_tool_context(
    mock_event_actions: MockEventActions,
) -> MockToolContext:
    """Create a mock tool context with full data."""
    return MockToolContext(
        agent_name="tool_agent",
        invocation_id="tool-inv-123",
        state=MockState({"tool_state": "active"}),
        user_content=MockContent({"text": "Execute tool"}),
        actions=mock_event_actions,
    )


@pytest.fixture
def mock_base_tool() -> MockBaseTool:
    """Create a mock tool with default name."""
    return MockBaseTool(name="test_tool")


@pytest.fixture
def mock_memory_callback_context() -> MockMemoryCallbackContext:
    """Create a mock callback context that succeeds."""
    return MockMemoryCallbackContext()


@pytest.fixture
def mock_memory_callback_context_no_service() -> MockMemoryCallbackContext:
    """Create a mock callback context that raises ValueError (no service)."""
    return MockMemoryCallbackContext(
        should_raise=ValueError,
        error_message="Cannot add session to memory: memory service is not available.",
    )


@pytest.fixture
def mock_memory_callback_context_with_runtime_error() -> MockMemoryCallbackContext:
    """Create a mock callback context that raises RuntimeError."""
    return MockMemoryCallbackContext(
        should_raise=RuntimeError,
        error_message="Memory service connection failed",
    )


@pytest.fixture
def mock_memory_callback_context_with_attribute_error() -> MockMemoryCallbackContext:
    """Create a mock callback context that raises AttributeError."""
    return MockMemoryCallbackContext(
        should_raise=AttributeError,
        error_message="'MockMemoryCallbackContext' has no invocation context",
    )


@pytest.fixture
def mock_readonly_context() -> MockReadonlyContext:
    """Create a mock readonly context for InstructionProvider testing."""
    return MockReadonlyContext(
        agent_name="instruction_test_agent",
        invocation_id="readonly-inv-123",
        state={"user_tier": "premium", "language": "en"},
    )


# Config testing fixtures
@pytest.fixture
def valid_server_env() -> dict[str, str]:
    """Valid environment variables for ServerEnv model.

    Returns:
        Dictionary with minimal required fields for ServerEnv.
    """
    return {
        "GOOGLE_CLOUD_PROJECT": "test-project",
        "AGENT_NAME": "test-agent",
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true",
    }


class MockEnviron(dict[str, str]):
    """Mock os.environ-like object for testing.

    Mimics os._Environ behavior while being a dict subclass.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize mock environ."""
        super().__init__(*args, **kwargs)


@pytest.fixture
def mock_environ() -> type[MockEnviron]:
    """Mock os.environ class for testing.

    Returns:
        MockEnviron class that behaves like os._Environ.
    """
    return MockEnviron


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clean environment variables before each test.

    Removes any existing environment variables that might interfere
    with tests to ensure isolation.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    env_vars_to_clean = [
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "AGENT_NAME",
        "LOG_LEVEL",
        "SERVE_WEB_INTERFACE",
        "RELOAD_AGENTS",
        "AGENT_ENGINE",
        "ARTIFACT_SERVICE_URI",
        "ALLOW_ORIGINS",
        "HOST",
        "PORT",
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
    ]

    for var in env_vars_to_clean:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def mock_load_dotenv() -> Generator[MagicMock]:
    """Mock load_dotenv function for testing.

    Yields:
        Mock object for load_dotenv function.
    """
    with patch("agent_foundation.utils.config.load_dotenv") as mock:
        yield mock


@pytest.fixture
def mock_sys_exit() -> Generator[MagicMock]:
    """Mock sys.exit with SystemExit side effect for testing validation failures.

    Yields:
        Mock object for sys.exit that raises SystemExit(1).
    """
    with patch("sys.exit", side_effect=SystemExit(1)) as mock:
        yield mock


@pytest.fixture
def set_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[dict[str, str]], None]:
    """Helper fixture to set multiple environment variables at once.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Function that takes a dictionary and sets all key-value pairs as env vars.
    """

    def _set_env(env_dict: dict[str, str]) -> None:
        """Set multiple environment variables from a dictionary.

        Args:
            env_dict: Dictionary of environment variable names and values.
        """
        for key, value in env_dict.items():
            monkeypatch.setenv(key, value)

    return _set_env


@pytest.fixture
def mock_print_config() -> Callable[[type], AbstractContextManager[MagicMock]]:
    """Context manager factory for mocking print_config on any model class.

    Returns:
        Factory function that creates a context manager for mocking print_config.
    """
    from contextlib import contextmanager
    from unittest.mock import patch

    @contextmanager
    def _mock_print_config(model_class: type) -> Generator[MagicMock]:
        """Create a context manager for mocking print_config on a model class.

        Args:
            model_class: The Pydantic model class to mock print_config on.

        Yields:
            Mock object for the print_config method.
        """
        with patch.object(model_class, "print_config", autospec=True) as mock:
            yield mock

    return _mock_print_config
