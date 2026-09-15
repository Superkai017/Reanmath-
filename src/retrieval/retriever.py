from reanmath.config import TOP_K
from reanmath.embeddings.embedder import embed_query
from reanmath.vectorstore.qdrant_store import search


def retrieve_context(query: str, top_k: int | None = None, min_score: float = 0.75) -> list[str]:
    """Embed the user's question and pull the top-k most similar chunks.

    min_score filters out weak matches so an unrelated question doesn't
    get padded with irrelevant curriculum text (see build_augmented_prompt
    in prompts.py, which falls back gracefully to an empty list anyway).
    """
    vector = embed_query(query)
    hits = search(vector, top_k=top_k or TOP_K)
    return [hit.payload["text"] for hit in hits if hit.score >= min_score]
