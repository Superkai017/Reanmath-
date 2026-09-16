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


def prepare_document(
    document: ExtractedDocument,
    *,
    chunk_size: int,
    chunk_overlap: int,
    segmenter: KhmerSegmenter,
) -> PreparedDocument:
    """Run the LaTeX-guarded text pipeline over every section of a document."""
    chunks: list[Chunk] = []
    formulas = 0
    characters = 0
    for section in document.sections:
        masked, vault = mask_latex(section.text)
        normalized = normalize_khmer_text(masked)
        if not normalized:
            continue
        segmented = segmenter.insert_word_boundaries(normalized)
        section_chunks = chunk_text(
            segmented,
            vault,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            page=section.page,
            start_index=len(chunks),
        )
        covered = {token for chunk in section_chunks for token in find_placeholders(chunk.text)}
        missing = set(find_placeholders(normalized)) - covered
        if missing:
            raise LatexIntegrityError(
                f"{len(missing)} formula(s) from {document.source} were not preserved in any chunk"
            )
        chunks.extend(section_chunks)
        formulas += len(vault)
        characters += len(unmask_latex(normalized, vault))
    return PreparedDocument(
        source=document.source,
        format=document.format,
        chunks=chunks,
        formulas=formulas,
        characters=characters,
        pages=document.pages,
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

    vectors = embedder.embed_documents([chunk.restored_text for chunk in prepared.chunks])
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
