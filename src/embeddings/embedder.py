"""multilingual-e5-large embeddings.

E5 models are trained with "query: " / "passage: " prefixes -- omitting
them silently degrades retrieval quality, so both helpers add them.
"""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from reanmath.config import EMBED_MODEL


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


def embed_passages(texts: list[str]) -> list[list[float]]:
    prefixed = [f"passage: {t}" for t in texts]
    vectors = _model().encode(prefixed, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    vector = _model().encode(f"query: {text}", normalize_embeddings=True)
    return vector.tolist()
