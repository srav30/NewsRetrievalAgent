import os
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

from dotenv import load_dotenv
from strands.models.openai import OpenAIModel
from strands_tools import calculator

from news_retrieval_agent.geocoding import get_place_coordinates
from news_retrieval_agent.newsretrieval import get_news_descriptions
from news_retrieval_agent.rag_retrieval import retrieve_rag_context
from news_retrieval_agent.weather import get_weather_forecast
from strands import Agent

# Load Strands/.env no matter which folder the terminal is in
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_DEBUG_LOG = PROJECT_ROOT / "logs" / "agent_debug.log"
_agent: Agent | None = None
SYSTEM_PROMPT = """
You are Agentic AI POC, an agent that answers user questions by using tools when
they can provide fresher or private context.

Use retrieve_rag_context before answering any question that may depend on local
or private documents, project-specific facts, uploaded files, company/financial
records, or terms that are not common public knowledge. This includes questions
about Aethelgard, the Aethelgard financial document, Zorblax-9, fund activity,
internal protocols, financial metrics, or any question where you are not certain
the answer is general knowledge.

When retrieve_rag_context returns context, ground your answer in that context.
If the retrieved context is insufficient, say what is missing instead of making
up details. Use the weather, geocoding, news, and calculator tools when the user
asks for those specific capabilities.
""".strip()


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
        system_prompt=SYSTEM_PROMPT,
        tools=[
            calculator,
            get_news_descriptions,
            get_place_coordinates,
            retrieve_rag_context,
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
    _print_agent_request(prompt)
    result = agent(prompt)
    new_messages = agent.messages[message_start:]
    tools_used = _extract_tools_used(new_messages)
    _print_agent_conversation(new_messages, str(result), tools_used)

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


def _agent_debug_enabled() -> bool:
    enabled_values = {"1", "true", "yes", "on"}
    return os.getenv("AGENT_DEBUG_CONSOLE", "true").lower() in enabled_values


def _print_agent_request(prompt: str) -> None:
    if not _agent_debug_enabled():
        return

    _debug_print("\n========== AGENT REQUEST ==========")
    _debug_print("System prompt:")
    _debug_print(SYSTEM_PROMPT)
    _debug_print("\nUser prompt:")
    _debug_print(prompt)
    _debug_print("===================================\n")


def _print_agent_conversation(
    messages: list[dict[str, Any]],
    final_response: str,
    tools_used: list[str],
) -> None:
    if not _agent_debug_enabled():
        return

    _debug_print("\n======= AGENT CONVERSATION =======")
    for index, message in enumerate(messages, start=1):
        _debug_print(f"\nMessage #{index} role={message.get('role')}")
        for content in message.get("content", []):
            _print_content_block(content)

    _debug_print("\nTools used:")
    _debug_print(tools_used or [])
    _debug_print("\nFinal response:")
    _debug_print(final_response)
    _debug_print("==================================\n")


def _print_content_block(content: dict[str, Any]) -> None:
    if "text" in content:
        _debug_print("Text:")
        _debug_print(content["text"])
        return

    tool_use = content.get("toolUse") or content.get("tool_use")
    if tool_use:
        _debug_print("Tool use:")
        _debug_print(pformat(tool_use, width=120))
        return

    tool_result = content.get("toolResult") or content.get("tool_result")
    if tool_result:
        _debug_print("Tool result:")
        _debug_print(pformat(tool_result, width=120))
        return

    _debug_print("Content:")
    _debug_print(pformat(content, width=120))


def _debug_print(value: Any) -> None:
    text = str(value)
    print(text, flush=True)
    AGENT_DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with AGENT_DEBUG_LOG.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{text}\n")
