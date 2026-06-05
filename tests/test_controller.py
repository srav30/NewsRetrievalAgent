from fastapi.testclient import TestClient

from news_retrieval_agent import controller
from news_retrieval_agent.agent import AgentAnswer


def test_ask_something_returns_agent_response(monkeypatch) -> None:
    monkeypatch.setattr(
        controller,
        "ask_agent_with_tools",
        lambda prompt: AgentAnswer(
            response=f"answered: {prompt}",
            tools_used=["get_weather_forecast"],
        ),
    )

    client = TestClient(controller.app)
    response = client.post("/askSomething", json={"question": "weather in Seattle"})

    assert response.status_code == 200
    assert response.json() == {
        "response": "answered: weather in Seattle",
        "tools_used": ["get_weather_forecast"],
    }


def test_ask_something_page_returns_html() -> None:
    client = TestClient(controller.app)
    response = client.get("/askSomething")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Ask Agentic AI POC" in response.text
