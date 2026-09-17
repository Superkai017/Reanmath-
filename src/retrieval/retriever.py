"""Query-time retrieval: embed, search, threshold, restore formulas."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from src.embeddings.embedder import Embedder
from src.ingestion.khmer_segment import strip_word_boundaries
from src.ingestion.latex_guard import unmask_latex
from src.vectorstore import InMemoryVectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    source: str
    chunk_index: int
    page: int | None
    score: float
    text: str  # formulas restored
    title: str = ""
    heading: str = ""


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of ``a`` and rows of ``b``.

    1-D inputs are treated as single rows; the result has shape (len(a), len(b)).
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[a_norm == 0] = 1.0
    b_norm[b_norm == 0] = 1.0
    return (a / a_norm) @ (b / b_norm).T


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: InMemoryVectorStore,
        *,
        top_k: int = 5,
        score_threshold: float = 0.0,
        max_context_chars: int = 12000,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.max_context_chars = max_context_chars

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        sources: Iterable[str] | None = None,
    ) -> list[RetrievedChunk]:
        query = query.strip()
        if not query or len(self.store) == 0:
            return []
        vector = self.embedder.embed_query(query)
        hits = self.store.search(
            vector,
            top_k=top_k if top_k is not None else self.top_k,
            score_threshold=self.score_threshold if score_threshold is None else score_threshold,
            sources=sources,
        )
        return [
            RetrievedChunk(
                id=hit.record.id,
                source=hit.record.source,
                chunk_index=hit.record.chunk_index,
                page=hit.record.page,
                score=round(hit.score, 6),
                text=strip_word_boundaries(unmask_latex(hit.record.text, hit.record.vault)),
                title=hit.record.metadata.get("title") or "",
                heading=hit.record.metadata.get("heading") or "",
            )
            for hit in hits
        ]

    def fit_context(self, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        """Keep the best chunks that fit within ``max_context_chars``.

        The top chunk is always kept so a single long passage is never dropped.
        """
        selected: list[RetrievedChunk] = []
        used = 0
        for chunk in chunks:
            if selected and used + len(chunk.text) > self.max_context_chars:
                continue
            selected.append(chunk)
            used += len(chunk.text)
        return selected


@dataclass(frozen=True)
class RetrievalCase:
    """A query and the chunk ids (or sources) that count as relevant."""

    query: str
    relevant_ids: frozenset[str] = frozenset()
    relevant_sources: frozenset[str] = frozenset()

    def is_relevant(self, chunk: RetrievedChunk) -> bool:
        return chunk.id in self.relevant_ids or chunk.source in self.relevant_sources


@dataclass(frozen=True)
class RetrievalMetrics:
    cases: int
    k: int
    hit_rate: float
    mrr: float
    precision_at_k: float
    recall_at_k: float


def evaluate_retrieval(
    retriever: Retriever, cases: Sequence[RetrievalCase], k: int = 5
) -> RetrievalMetrics:
    """Hit rate, MRR, precision@k and recall@k over labelled queries.

    Recall uses ``relevant_ids`` when given; for source-level labels it is the
    share of relevant sources that appear in the top k.
    """
    if not cases:
        raise ValueError("at least one retrieval case is required")
    hits = reciprocal_ranks = precision = recall = 0.0
    for case in cases:
        results = retriever.retrieve(case.query, top_k=k, score_threshold=-1.0)
        flags = [case.is_relevant(chunk) for chunk in results]
        if any(flags):
            hits += 1
            reciprocal_ranks += 1.0 / (flags.index(True) + 1)
        precision += sum(flags) / k
        if case.relevant_ids:
            found = {chunk.id for chunk in results} & case.relevant_ids
            recall += len(found) / len(case.relevant_ids)
        elif case.relevant_sources:
            found = {chunk.source for chunk in results} & case.relevant_sources
            recall += len(found) / len(case.relevant_sources)
    n = len(cases)
    return RetrievalMetrics(
        cases=n,
        k=k,
        hit_rate=hits / n,
        mrr=reciprocal_ranks / n,
        precision_at_k=precision / n,
        recall_at_k=recall / n,
    )
