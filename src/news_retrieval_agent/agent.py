import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from strands.models.openai import OpenAIModel
from strands_tools import calculator

from news_retrieval_agent.geocoding import get_place_coordinates
from news_retrieval_agent.newsretrieval import get_news_descriptions
from news_retrieval_agent.weather import get_weather_forecast
from strands import Agent

# Load Strands/.env no matter which folder the terminal is in
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_agent: Agent | None = None


@dataclass(frozen=True)
class AgentAnswer:
    """Agent response plus the tools used to produce it."""

    response: str
    tools_used: list[str]


def build_agent() -> Agent:
    """Create a Strands agent with all project tools registered."""
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"OPENAI_API_KEY is not set. Add it to {PROJECT_ROOT / '.env'}"
        )

    model = OpenAIModel(
        client_args={
            "api_key": api_key,
        },
        model_id="gpt-4o",
        params={
            "max_tokens": 1000,
            "temperature": 0.7,
        },
    )

    return Agent(
        model=model,
        tools=[
            calculator,
            get_news_descriptions,
            get_place_coordinates,
            get_weather_forecast,
        ],
    )


def get_agent() -> Agent:
    """Return a cached agent instance for repeated requests."""
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def ask_agent(prompt: str) -> str:
    """Ask the LLM agent a prompt and return its response text."""
    return ask_agent_with_tools(prompt).response


def ask_agent_with_tools(prompt: str) -> AgentAnswer:
    """Ask the LLM agent and return the response with tool usage."""
    agent = get_agent()
    message_start = len(agent.messages)
    result = agent(prompt)
    tools_used = _extract_tools_used(agent.messages[message_start:])

    return AgentAnswer(response=str(result), tools_used=tools_used)


def _extract_tools_used(messages: list[dict[str, Any]]) -> list[str]:
    """Extract tool names from Strands message content blocks."""
    tools_used = []
    seen = set()

    for message in messages:
        for content in message.get("content", []):
            tool_name = _extract_tool_name(content)
            if tool_name and tool_name not in seen:
                seen.add(tool_name)
                tools_used.append(tool_name)

    return tools_used


def _extract_tool_name(content: dict[str, Any]) -> str | None:
    tool_use = content.get("toolUse") or content.get("tool_use")
    if not isinstance(tool_use, dict):
        return None

    name = tool_use.get("name") or tool_use.get("toolName") or tool_use.get("tool_name")
    return name if isinstance(name, str) else None
