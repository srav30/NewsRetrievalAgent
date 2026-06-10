from news_retrieval_agent.rag_retrieval import _contexts_from_chroma


def test_contexts_from_chroma_converts_query_results() -> None:
    contexts = _contexts_from_chroma(
        {
            "ids": [["fallback-id"]],
            "documents": [["Retrieved context text"]],
            "distances": [[0.2]],
            "metadatas": [
                [
                    {
                        "chunk_id": "chunk-1",
                        "document_id": "doc-1",
                        "path": "source.txt",
                    }
                ]
            ],
        }
    )

    assert len(contexts) == 1
    assert contexts[0].chunk_id == "chunk-1"
    assert contexts[0].document_id == "doc-1"
    assert contexts[0].text == "Retrieved context text"
    assert contexts[0].score == 0.8
    assert contexts[0].metadata == {"path": "source.txt"}
