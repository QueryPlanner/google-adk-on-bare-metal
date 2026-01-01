import os
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm

# Import tools and sub-agents
from .tools.calculator import add_tool, multiply_tool
from .sub_agents.researcher import researcher_agent

api_base_url = "https://openrouter.ai/api/v1"

root_agent = Agent(
    model=LiteLlm(
        model="openrouter/openai/gpt-oss-20b",
        api_base=api_base_url,
        api_key=os.getenv("OPENROUTER_API_KEY"),
    ),
    name="root_agent",
    description="A helpful assistant for user questions.",
    instruction="""
    Answer user questions to the best of your knowledge.
    You can use the 'researcher' agent for complex topics by using the transfer tool.
    You have calculator tools for math.
    """,
    # Register the sub-agent structurally
    sub_agents=[researcher_agent()],
    # Give the agent tools to do its job (including transferring)
    tools=[add_tool, multiply_tool],
)
