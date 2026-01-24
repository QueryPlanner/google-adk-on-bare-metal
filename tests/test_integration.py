"""Integration tests for agent configuration and component wiring.

This module validates the basic structure and wiring of ADK app components.
Tests are pattern-based and validate integration points regardless of specific
implementation choices (plugins, tools, etc.).

Future: Container-based smoke tests for CI/CD will be added here.
"""

from agent_foundation import app


class TestAppIntegration:
    """Pattern-based integration tests for App configuration and wiring."""

    def test_app_is_properly_instantiated(self) -> None:
        """Verify app container is properly instantiated."""
        assert app is not None
        assert app.name is not None
        assert isinstance(app.name, str)
        assert len(app.name) > 0

    def test_app_has_root_agent(self) -> None:
        """Verify app is wired to root agent."""
        assert app.root_agent is not None

    def test_app_plugins_are_valid_if_configured(self) -> None:
        """Verify plugins (if any) are properly initialized."""
        # Plugins are optional - if configured, they should be a list
        if app.plugins is not None:
            assert isinstance(app.plugins, list)
            # Each plugin should be an object instance
            for plugin in app.plugins:
                assert plugin is not None
                assert hasattr(plugin, "__class__")


class TestAgentIntegration:
    """Pattern-based integration tests for Agent configuration."""

    def test_agent_has_required_configuration(self) -> None:
        """Verify agent has required configuration fields."""
        agent = app.root_agent

        # Required: agent name
        assert agent.name is not None
        assert isinstance(agent.name, str)
        assert len(agent.name) > 0

        # Required: agent model
        assert agent.model is not None
        assert isinstance(agent.model, str)
        assert len(agent.model) > 0

    def test_agent_instructions_are_valid_if_configured(self) -> None:
        """Verify agent instructions (if configured) are valid strings."""
        agent = app.root_agent

        # Instruction is optional - if configured, should be non-empty string
        if agent.instruction is not None:
            assert isinstance(agent.instruction, str)
            assert len(agent.instruction) > 0

        # Description is optional - if configured, should be non-empty string
        if agent.description is not None:
            assert isinstance(agent.description, str)
            assert len(agent.description) > 0

    def test_agent_tools_are_valid_if_configured(self) -> None:
        """Verify agent tools (if any) are properly initialized."""
        agent = app.root_agent

        # Tools are optional - if configured, should be a list
        if agent.tools is not None:
            assert isinstance(agent.tools, list)
            # Each tool should be an object instance
            for tool in agent.tools:
                assert tool is not None
                assert hasattr(tool, "__class__")
