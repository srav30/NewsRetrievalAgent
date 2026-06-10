"""Retrieve relevant RAG context from a persisted ChromaDB collection."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from strands import tool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"
DEFAULT_COLLECTION_NAME = "rag_chunks"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass(frozen=True)
class RagContext:
    """Retrieved context from ChromaDB."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict[str, str]


class ChromaRagRetriever:
    """Read-only retriever for a persisted ChromaDB collection."""

    def __init__(
        self,
        chroma_path: str | Path | None = None,
        collection_name: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        load_dotenv(PROJECT_ROOT / ".env")

        self.chroma_path = Path(
            chroma_path
            or os.getenv("RAG_CHROMA_PATH")
            or DEFAULT_CHROMA_PATH
        )
        self.collection_name = (
            collection_name
            or os.getenv("RAG_CHROMA_COLLECTION")
            or DEFAULT_COLLECTION_NAME
        )
        self.embedding_model = (
            embedding_model
            or os.getenv("RAG_EMBEDDING_MODEL")
            or DEFAULT_EMBEDDING_MODEL
        )
        self.openai_client = OpenAI()
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_path))
        self.collection = self.chroma_client.get_collection(name=self.collection_name)

    def retrieve(self, prompt: str, top_k: int = 3) -> list[RagContext]:
        """Return the most relevant Chroma chunks for a user prompt."""
        normalized_prompt = prompt.strip()
        if not normalized_prompt:
            raise ValueError("prompt must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_embedding = self._embed(normalized_prompt)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances", "metadatas"],
        )
        return _contexts_from_chroma(result)

    def _embed(self, text: str) -> list[float]:
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=[text],
        )
        return response.data[0].embedding


def _contexts_from_chroma(result: dict[str, Any]) -> list[RagContext]:
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    distances = result.get("distances", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    contexts = []
    for chunk_id, text, distance, metadata in zip(
        ids,
        documents,
        distances,
        metadatas,
        strict=True,
    ):
        metadata = {key: str(value) for key, value in dict(metadata or {}).items()}
        contexts.append(
            RagContext(
                chunk_id=metadata.pop("chunk_id", str(chunk_id)),
                document_id=metadata.pop("document_id", ""),
                text=text or "",
                score=1 - float(distance),
                metadata=metadata,
            )
        )

    return contexts


@tool
def retrieve_rag_context(prompt: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Search private/local RAG documents in ChromaDB for relevant context.

    Use this tool when the user asks about information that may live in the
    local document corpus rather than in the model's general knowledge. The
    current corpus includes Aethelgard financial/project documents, including
    details about Zorblax-9, internal protocols, fund activity, and document
    facts. Pass the user's full question as the prompt.
    """
    contexts = ChromaRagRetriever().retrieve(prompt, top_k)
    return [
        {
            "chunk_id": context.chunk_id,
            "document_id": context.document_id,
            "text": context.text,
            "score": round(context.score, 4),
            "metadata": context.metadata,
        }
        for context in contexts
    ]
