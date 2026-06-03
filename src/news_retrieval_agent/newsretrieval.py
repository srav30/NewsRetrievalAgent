"""Retrieve news descriptions from NewsData."""

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from strands import tool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEWSDATA_URL = "https://newsdata.io/api/1/latest"


class NewsRetrieval:
    """Client for fetching top news descriptions from NewsData."""

    def __init__(self, api_key: str | None = None) -> None:
        load_dotenv(PROJECT_ROOT / ".env")
        self.api_key = api_key or os.getenv("NEWSDATA_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                f"NEWSDATA_API_KEY is not set. Add it to {PROJECT_ROOT / '.env'}"
            )

    def get_top_descriptions(self, search_keyword: str, country: str) -> list[str]:
        """Return the first five non-empty descriptions for a keyword and country."""
        query_params = {
            "apikey": self.api_key,
            "q": search_keyword,
            "country": country,
        }
        request = Request(
            f"{NEWSDATA_URL}?{urlencode(query_params)}",
            headers={"Accept": "application/json"},
        )

        try:
            with urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"NewsData request failed with status {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"NewsData request failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("NewsData response was not valid JSON") from exc

        articles = payload.get("results", payload) if isinstance(payload, dict) else payload
        if not isinstance(articles, list):
            raise RuntimeError("NewsData response did not include a list of articles")

        descriptions = [
            article["description"]
            for article in articles
            if isinstance(article, dict) and article.get("description")
        ]
        return descriptions[:5]


@tool
def get_news_descriptions(search_keyword: str, country: str) -> list[str]:
    """Fetch the top five news descriptions for a search keyword and country."""
    return NewsRetrieval().get_top_descriptions(search_keyword, country)
