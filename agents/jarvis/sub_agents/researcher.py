import os
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm

def researcher_agent() -> Agent:
    # Shared model config (could be different for this agent)
    api_base_url = "https://openrouter.ai/api/v1"
    model=LiteLlm(
        model="openrouter/openai/gpt-oss-20b",
        api_base=api_base_url,
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    return Agent(
        model=model,
        name="researcher",
        description="Can perform detailed research on a topic.",
        instruction="""
        You are a researcher. When asked to research something, provide a detailed summary.
        If you don't know, just say you don't know.
        """,
    )
