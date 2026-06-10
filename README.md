# Agentic AI POC

Agentic AI POC is a Python project that combines a Strands agent with tools for news retrieval, place geocoding, weather forecasts, and local RAG retrieval.

## Functionality

- Loads environment variables from the project `.env` file.
- Uses `OPENAI_API_KEY` to configure a Strands `Agent` backed by the OpenAI `gpt-4o` model.
- Provides a `get_news_descriptions` Strands tool that calls the NewsData latest-news API.
- Uses `NEWSDATA_API_KEY` to fetch articles by search keyword and country.
- Returns the first five non-empty article descriptions from the NewsData response.
- Uses a bundled ChromaDB vector store in `data/chroma` for private document retrieval.
- Includes a simple package entry point that prints a CLI greeting.

## File Structure

```text
AgenticAIPOC/
├── README.md
├── data/
│   └── chroma/
├── pyproject.toml
├── src/
│   └── news_retrieval_agent/
│       ├── __init__.py
│       ├── __main__.py
│       ├── agent.py
│       ├── main.py
│       ├── newsretrieval.py
│       ├── rag_retrieval.py
│       └── weather.py
└── tests/
    └── test_main.py
```

- `pyproject.toml` defines the package metadata, build backend, CLI script, pytest settings, and Ruff lint rules.
- `src/news_retrieval_agent/agent.py` configures the Strands agent, loads the OpenAI API key, and registers the available tools.
- `src/news_retrieval_agent/controller.py` exposes the `/askSomething` API endpoint and HTML page.
- `src/news_retrieval_agent/geocoding.py` resolves free-form place names to latitude and longitude.
- `src/news_retrieval_agent/newsretrieval.py` contains the NewsData API client and the `get_news_descriptions` Strands tool.
- `src/news_retrieval_agent/rag_retrieval.py` queries the bundled ChromaDB vector store for relevant private document context.
- `src/news_retrieval_agent/weather.py` contains the Open-Meteo weather client and forecast tool.
- `src/news_retrieval_agent/main.py` contains the CLI entry point.
- `src/news_retrieval_agent/__main__.py` allows the package to run through Python module execution.
- `tests/test_main.py` verifies the CLI greeting output.
