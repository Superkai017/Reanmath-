"""In-memory cosine-similarity vector index with atomic on-disk persistence.

The whole index lives in one ``index.npz`` file (vectors + JSON records +
manifest), written to a temporary file and swapped in with ``os.replace``, so
a crash mid-save never leaves a half-written index behind.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

INDEX_FILENAME = "index.npz"
FORMAT_VERSION = 1


class VectorStoreError(RuntimeError):
    pass


class EmbeddingMismatchError(VectorStoreError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ChunkRecord:
    """A stored chunk. ``text`` is masked; ``vault`` restores its formulas."""

    id: str
    source: str
    chunk_index: int
    text: str
    vault: dict[str, str] = field(default_factory=dict)
    page: int | None = None
    format: str = "text"
    created_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkRecord":
        return cls(
            id=data["id"],
            source=data["source"],
            chunk_index=int(data["chunk_index"]),
            text=data["text"],
            vault=dict(data.get("vault") or {}),
            page=data.get("page"),
            format=data.get("format", "text"),
            created_at=data.get("created_at") or utc_now(),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class SearchHit:
    record: ChunkRecord
    score: float


@dataclass(frozen=True)
class SourceSummary:
    source: str
    chunks: int
    pages: int
    format: str
    ingested_at: str


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class InMemoryVectorStore:
    def __init__(self, path: str | Path | None = None, embedding_model: str = "unknown") -> None:
        self.path = Path(path) if path is not None else None
        self.embedding_model = embedding_model
        self._vectors = np.zeros((0, 0), dtype=np.float32)
        self._records: list[ChunkRecord] = []
        self._id_to_row: dict[str, int] = {}
        self._lock = threading.RLock()

    # -- construction -------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str | Path,
        embedding_model: str,
        *,
        allow_model_change: bool = False,
    ) -> "InMemoryVectorStore":
        """Open the index at ``path`` (a directory), or start an empty one.

        Raises ``EmbeddingMismatchError`` if the index was built with another
        embedding model, unless ``allow_model_change`` is set, in which case
        the old index is ignored (and overwritten on the next save).
        """
        store = cls(path, embedding_model)
        index_file = store.index_file
        if index_file is None or not index_file.exists():
            return store

        try:
            with np.load(index_file, allow_pickle=False) as archive:
                manifest = json.loads(archive["manifest"].tobytes().decode("utf-8"))
                vectors = archive["vectors"].astype(np.float32, copy=False)
                records_raw = json.loads(archive["records"].tobytes().decode("utf-8"))
        except (OSError, KeyError, ValueError) as exc:
            raise VectorStoreError(f"Could not read vector index {index_file}: {exc}") from exc

        stored_model = manifest.get("embedding_model")
        if stored_model != embedding_model:
            message = (
                f"Vector index at {index_file} was built with embedding model "
                f"'{stored_model}', but '{embedding_model}' is configured. Re-index with "
                "`uv run python scripts/ingest_corpus.py --reset`."
            )
            if allow_model_change:
                logger.warning("%s Starting from an empty index.", message)
                return store
            raise EmbeddingMismatchError(message)

        records = [ChunkRecord.from_dict(item) for item in records_raw]
        if len(records) != vectors.shape[0]:
            raise VectorStoreError(
                f"Corrupt vector index {index_file}: {len(records)} records, {vectors.shape[0]} vectors"
            )
        store._records = records
        store._vectors = vectors
        store._rebuild_id_map()
        logger.info("Loaded %d chunks from %s", len(records), index_file)
        return store

    @property
    def index_file(self) -> Path | None:
        return self.path / INDEX_FILENAME if self.path is not None else None

    # -- properties -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    @property
    def dimension(self) -> int | None:
        return self._vectors.shape[1] if self._records else None

    def get(self, record_id: str) -> ChunkRecord | None:
        with self._lock:
            row = self._id_to_row.get(record_id)
            return self._records[row] if row is not None else None

    def records(self) -> list[ChunkRecord]:
        with self._lock:
            return list(self._records)

    def has_source(self, source: str) -> bool:
        with self._lock:
            return any(record.source == source for record in self._records)

    def sources(self) -> list[SourceSummary]:
        with self._lock:
            grouped: dict[str, list[ChunkRecord]] = {}
            for record in self._records:
                grouped.setdefault(record.source, []).append(record)
        summaries = []
        for source, records in sorted(grouped.items()):
            pages = {record.page for record in records if record.page is not None}
            summaries.append(
                SourceSummary(
                    source=source,
                    chunks=len(records),
                    pages=len(pages),
                    format=records[0].format,
                    ingested_at=min(record.created_at for record in records),
                )
            )
        return summaries

    # -- mutation ---------------------------------------------------------------

    def _rebuild_id_map(self) -> None:
        self._id_to_row = {record.id: row for row, record in enumerate(self._records)}

    def _validate(self, records: Sequence[ChunkRecord], vectors: np.ndarray) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("vectors must be a 2-D array")
        if matrix.shape[0] != len(records):
            raise ValueError(f"{len(records)} records but {matrix.shape[0]} vectors")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("vectors contain NaN or infinite values")
        if self._records and matrix.shape[0] and matrix.shape[1] != self._vectors.shape[1]:
            raise EmbeddingMismatchError(
                f"Vector dimension {matrix.shape[1]} does not match index dimension {self._vectors.shape[1]}"
            )
        ids = [record.id for record in records]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate record ids in batch")
        return _normalize_rows(matrix)

    def _remove_rows(self, keep: np.ndarray) -> int:
        removed = int((~keep).sum())
        if removed:
            self._vectors = self._vectors[keep]
            self._records = [record for record, kept in zip(self._records, keep) if kept]
            self._rebuild_id_map()
        return removed

    def add(self, records: Sequence[ChunkRecord], vectors: np.ndarray) -> int:
        """Insert records, replacing any with the same id."""
        if not records:
            return 0
        with self._lock:
            matrix = self._validate(records, vectors)
            new_ids = {record.id for record in records}
            if self._records:
                keep = np.array([record.id not in new_ids for record in self._records], dtype=bool)
                self._remove_rows(keep)
            if self._records:
                self._vectors = np.vstack([self._vectors, matrix])
            else:
                self._vectors = matrix.copy()
            self._records.extend(records)
            self._rebuild_id_map()
            return len(records)

    def delete_source(self, source: str) -> int:
        with self._lock:
            if not self._records:
                return 0
            keep = np.array([record.source != source for record in self._records], dtype=bool)
            return self._remove_rows(keep)

    def replace_source(
        self, source: str, records: Sequence[ChunkRecord], vectors: np.ndarray
    ) -> tuple[int, int]:
        """Atomically swap every chunk of ``source``. Returns (removed, added)."""
        with self._lock:
            if any(record.source != source for record in records):
                raise ValueError("all records must belong to the source being replaced")
            if records:
                self._validate(records, vectors)  # fail before deleting anything
            removed = self.delete_source(source)
            added = self.add(records, vectors)
            return removed, added

    def clear(self) -> None:
        with self._lock:
            self._records = []
            self._vectors = np.zeros((0, 0), dtype=np.float32)
            self._id_to_row = {}

    # -- search -----------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        score_threshold: float | None = None,
        sources: Iterable[str] | None = None,
    ) -> list[SearchHit]:
        """Top-k records by cosine similarity, best first."""
        if top_k <= 0:
            return []
        with self._lock:
            if not self._records:
                return []
            query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
            if query.shape[0] != self._vectors.shape[1]:
                raise EmbeddingMismatchError(
                    f"Query dimension {query.shape[0]} does not match index dimension {self._vectors.shape[1]}"
                )
            norm = float(np.linalg.norm(query))
            if norm == 0.0:
                return []
            scores = self._vectors @ (query / norm)

            candidates = np.arange(len(self._records))
            if sources is not None:
                wanted = set(sources)
                candidates = np.array(
                    [row for row, record in enumerate(self._records) if record.source in wanted],
                    dtype=np.int64,
                )
                if candidates.size == 0:
                    return []
            if score_threshold is not None:
                candidates = candidates[scores[candidates] >= score_threshold]
                if candidates.size == 0:
                    return []

            k = min(top_k, candidates.size)
            candidate_scores = scores[candidates]
            if k < candidates.size:
                top = np.argpartition(-candidate_scores, k - 1)[:k]
            else:
                top = np.arange(candidates.size)
            order = top[np.argsort(-candidate_scores[top], kind="stable")]
            return [
                SearchHit(self._records[int(candidates[i])], float(candidate_scores[i]))
                for i in order
            ]

    # -- persistence --------------------------------------------------------------

    def save(self) -> Path | None:
        """Write the index atomically. No-op for memory-only stores."""
        if self.path is None:
            return None
        with self._lock:
            self.path.mkdir(parents=True, exist_ok=True)
            manifest = {
                "format_version": FORMAT_VERSION,
                "embedding_model": self.embedding_model,
                "dimension": self.dimension,
                "count": len(self._records),
                "saved_at": utc_now(),
            }
            records_json = json.dumps(
                [record.to_dict() for record in self._records], ensure_ascii=False
            ).encode("utf-8")
            descriptor, temp_name = tempfile.mkstemp(dir=self.path, prefix=".index-", suffix=".npz")
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    np.savez(
                        handle,
                        vectors=self._vectors,
                        records=np.frombuffer(records_json, dtype=np.uint8),
                        manifest=np.frombuffer(json.dumps(manifest).encode("utf-8"), dtype=np.uint8),
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_name, self.index_file)
            except BaseException:
                Path(temp_name).unlink(missing_ok=True)
                raise
            return self.index_file


__all__ = [
    "ChunkRecord",
    "EmbeddingMismatchError",
    "InMemoryVectorStore",
    "SearchHit",
    "SourceSummary",
    "VectorStoreError",
]
