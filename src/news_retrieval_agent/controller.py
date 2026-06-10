"""HTTP controller for asking Agentic AI POC."""

from pathlib import Path
from pprint import pformat

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from news_retrieval_agent.agent import ask_agent_with_tools

APP_NAME = "Agentic AI POC"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUEST_LOG = PROJECT_ROOT / "logs" / "agent_debug.log"

app = FastAPI(title=APP_NAME)


class AskSomethingRequest(BaseModel):
    """Request body for asking the agent a question."""

    question: str = Field(..., min_length=1)


class AskSomethingResponse(BaseModel):
    """Response body returned by the agent."""

    response: str
    tools_used: list[str]


@app.get("/askSomething", response_class=HTMLResponse)
def ask_something_page() -> str:
    """Render a simple page for asking the agent questions."""
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agentic AI POC</title>
  <style>
    body {
      color: #1f2937;
      font-family: Arial, sans-serif;
      margin: 2rem auto;
      max-width: 760px;
      padding: 0 1rem;
    }
    textarea {
      box-sizing: border-box;
      font: inherit;
      min-height: 110px;
      padding: 0.75rem;
      width: 100%;
    }
    button {
      cursor: pointer;
      font: inherit;
      margin-top: 0.75rem;
      padding: 0.65rem 1rem;
    }
    .card {
      border: 1px solid #d1d5db;
      border-radius: 8px;
      margin-top: 1.5rem;
      padding: 1rem;
    }
    pre {
      white-space: pre-wrap;
    }
  </style>
</head>
<body>
  <h1>Ask Agentic AI</h1>
  <form id="ask-form">
    <label for="question">Question</label>
    <textarea id="question" required>What is the weather in Seattle?</textarea>
    <button type="submit">Ask</button>
  </form>

  <section class="card">
    <h2>Response</h2>
    <pre id="response">Submit a question to see the answer.</pre>
  </section>

  <section class="card">
    <h2>Tools Used</h2>
    <ul id="tools-used"></ul>
    <pre id="tools-debug"></pre>
  </section>

  <script>
    const form = document.getElementById("ask-form");
    const responseEl = document.getElementById("response");
    const toolsEl = document.getElementById("tools-used");
    const toolsDebugEl = document.getElementById("tools-debug");

    function renderToolsUsed(toolsUsed) {
      toolsEl.replaceChildren();
      toolsDebugEl.textContent = "";

      if (!Array.isArray(toolsUsed) || toolsUsed.length === 0) {
        const item = document.createElement("li");
        item.textContent = "No tools were used.";
        toolsEl.appendChild(item);
        return;
      }

      for (const tool of toolsUsed) {
        const item = document.createElement("li");
        item.textContent = tool;
        toolsEl.appendChild(item);
      }
      toolsDebugEl.textContent = `Raw tools_used: ${JSON.stringify(toolsUsed)}`;
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      responseEl.textContent = "Asking the agent...";
      renderToolsUsed([]);

      const question = document.getElementById("question").value;
      const response = await fetch("/askSomething", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question}),
      });
      const data = await response.json();

      if (!response.ok) {
        responseEl.textContent = data.detail || "Request failed";
        return;
      }

      responseEl.textContent = data.response;
      renderToolsUsed(data.tools_used);
    });
  </script>
</body>
</html>
"""


@app.post("/askSomething", response_model=AskSomethingResponse)
def ask_something(request: AskSomethingRequest) -> AskSomethingResponse:
    """Ask the LLM agent and return its tool-assisted response."""
    _debug_log(
        {
            "event": "http_request",
            "endpoint": "/askSomething",
            "question": request.question,
        }
    )
    try:
        answer = ask_agent_with_tools(request.question.strip())
    except Exception as exc:
        _debug_log(
            {
                "event": "http_error",
                "endpoint": "/askSomething",
                "error": str(exc),
            }
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    _debug_log(
        {
            "event": "http_response",
            "endpoint": "/askSomething",
            "response": answer.response,
            "tools_used": answer.tools_used,
        }
    )
    return AskSomethingResponse(
        response=answer.response,
        tools_used=answer.tools_used,
    )


def _debug_log(value: object) -> None:
    text = pformat(value, width=120)
    print(text, flush=True)
    REQUEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with REQUEST_LOG.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{text}\n")
