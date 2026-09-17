"""Ingestion pipeline: extract -> normalise -> mask LaTeX -> segment Khmer ->
chunk -> embed -> index."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from src.ingestion.chunk import Chunk, RecursiveCharacterTextSplitter, chunk_text
from src.ingestion.extract import (
    SUPPORTED_EXTENSIONS,
    ExtractedDocument,
    ExtractionError,
    Section,
    UnsupportedFileTypeError,
    extract_document,
    extract_file,
)
from src.ingestion.khmer_segment import (
    KhmerSegmenter,
    detect_language,
    get_segmenter,
    normalize_khmer_text,
    segment_khmer,
)
from src.ingestion.latex_guard import find_placeholders, mask_latex, unmask_latex
from src.vectorstore import ChunkRecord, InMemoryVectorStore

logger = logging.getLogger(__name__)


class LatexIntegrityError(RuntimeError):
    """A formula was lost between masking and chunking."""


class DocumentEmbedder(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class PreparedDocument:
    source: str
    format: str
    chunks: list[Chunk]
    formulas: int
    characters: int
    pages: int | None = None
    title: str = ""
    warnings: list[str] = field(default_factory=list)
    ocr_page_numbers: frozenset[int] = frozenset()


@dataclass
class IndexResult:
    source: str
    format: str
    chunks_added: int
    chunks_removed: int
    formulas: int
    characters: int
    pages: int | None
    warnings: list[str]
    ocr_pages: int = 0


class DuplicateSourceError(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass


_STREAM_SEPARATOR = "\n\n"
MAX_CONTEXT_HEADER_CHARS = 160


def _heading_streams(sections: list[Section]) -> list[list[Section]]:
    """Group sections into runs of text under one heading.

    A run continues across page breaks and ends where a new heading starts,
    so a chunk can span two pages but never two headings.
    """
    streams: list[list[Section]] = []
    for section in sections:
        if streams and not section.starts_heading and section.heading == streams[-1][-1].heading:
            streams[-1].append(section)
        else:
            streams.append([section])
    return streams


def context_header(title: str, heading: str) -> str:
    """Document title and heading path, prepended to a chunk for embedding."""
    header = " › ".join(part for part in (title.strip(), heading.strip()) if part)
    if len(header) > MAX_CONTEXT_HEADER_CHARS:
        header = "…" + header[-(MAX_CONTEXT_HEADER_CHARS - 1):]
    return header


def embedding_text(title: str, chunk: Chunk) -> str:
    header = context_header(title, chunk.heading)
    return f"{header}\n\n{chunk.restored_text}" if header else chunk.restored_text


def prepare_document(
    document: ExtractedDocument,
    *,
    chunk_size: int,
    chunk_overlap: int,
    segmenter: KhmerSegmenter,
) -> PreparedDocument:
    """Run the LaTeX-guarded text pipeline over a document, one heading run at a time."""
    chunks: list[Chunk] = []
    formulas = 0
    characters = 0
    for stream in _heading_streams(document.sections):
        parts: list[str] = []
        page_starts: list[tuple[int, int | None]] = []
        vault: dict[str, str] = {}
        expected: set[str] = set()
        offset = 0
        for section in stream:
            masked, section_vault = mask_latex(section.text)
            normalized = normalize_khmer_text(masked)
            if not normalized:
                continue
            if parts:
                offset += len(_STREAM_SEPARATOR)
            page_starts.append((offset, section.page))
            segmented = segmenter.insert_word_boundaries(normalized)
            parts.append(segmented)
            offset += len(segmented)
            vault.update(section_vault)
            expected.update(find_placeholders(normalized))
            formulas += len(section_vault)
            characters += len(unmask_latex(normalized, section_vault))
        if not parts:
            continue
        stream_chunks = chunk_text(
            _STREAM_SEPARATOR.join(parts),
            vault,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            page_starts=page_starts,
            heading=stream[0].heading,
            start_index=len(chunks),
        )
        covered = {token for chunk in stream_chunks for token in find_placeholders(chunk.text)}
        missing = expected - covered
        if missing:
            raise LatexIntegrityError(
                f"{len(missing)} formula(s) from {document.source} were not preserved in any chunk"
            )
        chunks.extend(stream_chunks)
    return PreparedDocument(
        source=document.source,
        format=document.format,
        chunks=chunks,
        formulas=formulas,
        characters=characters,
        pages=document.pages,
        title=document.title,
        warnings=list(document.warnings),
        ocr_page_numbers=frozenset(
            section.page for section in document.sections if section.ocr and section.page is not None
        ),
    )


def chunk_id(source: str, chunk: Chunk) -> str:
    digest = hashlib.sha256(
        f"{source}\x00{chunk.index}\x00{chunk.page}\x00{chunk.text}".encode("utf-8")
    ).hexdigest()
    return digest[:32]


def index_document(
    document: ExtractedDocument,
    *,
    store: InMemoryVectorStore,
    embedder: DocumentEmbedder,
    segmenter: KhmerSegmenter,
    chunk_size: int,
    chunk_overlap: int,
    replace: bool = True,
    persist: bool = True,
) -> IndexResult:
    """Prepare, embed and store a document. Existing chunks of the same source
    are replaced when ``replace`` is set, otherwise ``DuplicateSourceError``."""
    if not replace and store.has_source(document.source):
        raise DuplicateSourceError(f"'{document.source}' is already indexed")

    prepared = prepare_document(
        document, chunk_size=chunk_size, chunk_overlap=chunk_overlap, segmenter=segmenter
    )
    if not prepared.chunks:
        hint = " ".join(prepared.warnings)
        raise EmptyDocumentError(f"No extractable text in '{document.source}'. {hint}".strip())

    vectors = embedder.embed_documents([embedding_text(prepared.title, chunk) for chunk in prepared.chunks])
    records = [
        ChunkRecord(
            id=chunk_id(prepared.source, chunk),
            source=prepared.source,
            chunk_index=chunk.index,
            text=chunk.text,
            vault=chunk.vault,
            page=chunk.page,
            format=prepared.format,
            metadata={
                "language": detect_language(chunk.restored_text),
                "ocr": chunk.page in prepared.ocr_page_numbers,
                "title": prepared.title,
                "heading": chunk.heading,
                "page_end": chunk.last_page,
            },
        )
        for chunk in prepared.chunks
    ]
    removed, added = store.replace_source(prepared.source, records, vectors)
    if persist:
        store.save()
    logger.info(
        "Indexed %s: %d chunks (%d replaced), %d formulas",
        prepared.source, added, removed, prepared.formulas,
    )
    return IndexResult(
        source=prepared.source,
        format=prepared.format,
        chunks_added=added,
        chunks_removed=removed,
        formulas=prepared.formulas,
        characters=prepared.characters,
        pages=prepared.pages,
        warnings=prepared.warnings,
        ocr_pages=len(prepared.ocr_page_numbers),
    )


__all__ = [
    "SUPPORTED_EXTENSIONS",
    "Chunk",
    "DuplicateSourceError",
    "EmptyDocumentError",
    "ExtractedDocument",
    "ExtractionError",
    "IndexResult",
    "KhmerSegmenter",
    "LatexIntegrityError",
    "PreparedDocument",
    "RecursiveCharacterTextSplitter",
    "Section",
    "UnsupportedFileTypeError",
    "chunk_text",
    "detect_language",
    "extract_document",
    "extract_file",
    "get_segmenter",
    "index_document",
    "mask_latex",
    "normalize_khmer_text",
    "prepare_document",
    "segment_khmer",
    "unmask_latex",
]
