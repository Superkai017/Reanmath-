"""Dense sentence embeddings, plus an offline feature-hashing fallback.

All embedders return L2-normalised float32 arrays, so a dot product is the
cosine similarity.
"""
from __future__ import annotations

import hashlib
import logging
import math
import threading
from collections import Counter
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from src.ingestion.khmer_segment import tokenize_for_search

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from src.config import Settings

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    name: str

    @property
    def dimension(self) -> int: ...

    @property
    def is_loaded(self) -> bool: ...

    def warmup(self) -> None: ...

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32, copy=False)


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model, loaded on first use.

    E5-family models are trained with ``query: `` / ``passage: `` prefixes;
    leaving them out quietly degrades retrieval, so they are added
    automatically when the model name contains ``e5``.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 32,
        cache_folder: str | None = None,
    ) -> None:
        self.name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache_folder = cache_folder
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()
        uses_e5_prefixes = "e5" in model_name.lower()
        self.query_prefix = "query: " if uses_e5_prefixes else ""
        self.document_prefix = "passage: " if uses_e5_prefixes else ""

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading embedding model %s on %s", self.name, self.device)
                    self._model = SentenceTransformer(
                        self.name, device=self.device, cache_folder=self.cache_folder
                    )
        return self._model

    def warmup(self) -> None:
        self._get_model()

    @property
    def dimension(self) -> int:
        model = self._get_model()
        getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        dimension = getter()
        if dimension is None:
            dimension = int(self._encode(["dimension probe"]).shape[1])
        return int(dimension)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._get_model().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return self._encode([self.document_prefix + text for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])[0]


class HashingEmbedder:
    """Signed feature hashing over lexical tokens and token bigrams.

    Works fully offline with no model download. Khmer is tokenised into
    orthographic clusters, so cluster bigrams approximate Khmer words; LaTeX
    command names (``\\lim``, ``\\int``) become features too. Retrieval
    quality is lexical, below a dense model, but deterministic and fast.
    """

    def __init__(self, dimension: int = 1024) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self.name = f"hashing-{dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def is_loaded(self) -> bool:
        return True

    def warmup(self) -> None:
        return None

    @staticmethod
    def _features(text: str) -> Counter[str]:
        tokens = tokenize_for_search(text)
        features: Counter[str] = Counter(f"u:{token}" for token in tokens)
        features.update(f"b:{left}\x00{right}" for left, right in zip(tokens, tokens[1:]))
        return features

    def _slot(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self._dimension, (1.0 if value >> 63 else -1.0)

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dimension, dtype=np.float32)
        for feature, count in self._features(text).items():
            slot, sign = self._slot(feature)
            weight = 1.0 + math.log(count)
            if feature.startswith("b:"):
                weight *= 0.5
            vector[slot] += sign * weight
        return vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return _l2_normalize(np.stack([self._embed(text) for text in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2_normalize(self._embed(text))


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "hashing":
        return HashingEmbedder(settings.hashing_dim)
    return SentenceTransformerEmbedder(
        settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
