import os
from pathlib import Path

from dotenv import load_dotenv
from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools import calculator

from news_retrieval_agent.newsretrieval import get_news_descriptions

# Load Strands/.env no matter which folder the terminal is in
PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    # **model_config
    model_id="gpt-4o",
    params={
        "max_tokens": 1000,
        "temperature": 0.7,
    }
)

agent = Agent(model=model, tools=[calculator, get_news_descriptions])
response = agent("Give me the latest news about the stock market")
print(response)
