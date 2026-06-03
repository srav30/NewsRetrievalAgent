# NewsRetrievalAgent

NewsRetrievalAgent is a Python project that combines a Strands agent with a NewsData retrieval tool.

## Functionality

- Loads environment variables from the project `.env` file.
- Uses `OPENAI_API_KEY` to configure a Strands `Agent` backed by the OpenAI `gpt-4o` model.
- Provides a `get_news_descriptions` Strands tool that calls the NewsData latest-news API.
- Uses `NEWSDATA_API_KEY` to fetch articles by search keyword and country.
- Returns the first five non-empty article descriptions from the NewsData response.
- Includes a simple package entry point that prints a CLI greeting.

## File Structure

```text
NewsRetrievalAgent/
├── README.md
├── pyproject.toml
├── src/
│   └── news_retrieval_agent/
│       ├── __init__.py
│       ├── __main__.py
│       ├── agent.py
│       ├── main.py
│       └── newsretrieval.py
└── tests/
    └── test_main.py
```

- `pyproject.toml` defines the package metadata, build backend, CLI script, pytest settings, and Ruff lint rules.
- `src/news_retrieval_agent/agent.py` configures the Strands agent, loads the OpenAI API key, registers tools, and runs a sample stock-market news prompt.
- `src/news_retrieval_agent/newsretrieval.py` contains the NewsData API client and the `get_news_descriptions` Strands tool.
- `src/news_retrieval_agent/main.py` contains the CLI entry point.
- `src/news_retrieval_agent/__main__.py` allows the package to run through Python module execution.
- `tests/test_main.py` verifies the CLI greeting output.
