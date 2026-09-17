"""Request and response models for the HTTP API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SOURCE_NAME_PATTERN = r"^[^\x00-\x1f<>\"|?*]+$"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# --- /api/query -------------------------------------------------------------

class ChatTurn(_StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)


class QueryRequest(_StrictModel):
    prompt: str = Field(min_length=1, max_length=4000, description="The student's question")
    top_k: int | None = Field(None, ge=1, le=50, description="Chunks to retrieve (default: TOP_K)")
    score_threshold: float | None = Field(
        None, ge=0.0, le=1.0, description="Minimum cosine similarity (default: SCORE_THRESHOLD)"
    )
    history: list[ChatTurn] = Field(default_factory=list, max_length=40)
    sources: list[str] | None = Field(
        None, max_length=100, description="Restrict retrieval to these document sources"
    )
    generate: bool = Field(True, description="Set false to return retrieved chunks only")


class SourceChunk(BaseModel):
    id: str
    source: str
    chunk_index: int
    page: int | None = None
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    language: Literal["km", "en"]
    grounded: bool = Field(description="True when at least one chunk passed the threshold")
    sources: list[SourceChunk]
    provider: Literal["anthropic", "gemini", "none"]
    model: str | None = None
    stop_reason: str | None = None
    latency_ms: float


# --- /api/ingest ------------------------------------------------------------

class IngestTextRequest(_StrictModel):
    text: str = Field(min_length=1, max_length=5_000_000)
    source_name: str = Field(min_length=1, max_length=200, pattern=SOURCE_NAME_PATTERN)
    format: Literal["text", "markdown"] = "markdown"
    replace: bool = True

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class IngestResponse(BaseModel):
    source: str
    format: str
    chunks_added: int
    chunks_replaced: int
    formulas_protected: int
    characters: int
    pages: int | None = None
    ocr_pages: int = Field(0, description="PDF pages or images transcribed by OCR (Gemini or Kiri)")
    warnings: list[str] = Field(default_factory=list)
    total_chunks: int


# --- /api/documents ---------------------------------------------------------

class DocumentInfo(BaseModel):
    source: str
    chunks: int
    pages: int
    format: str
    ingested_at: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int


class DeleteDocumentResponse(BaseModel):
    source: str
    chunks_removed: int
    total_chunks: int


# --- /health ------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    embedding_backend: str
    embedding_model: str
    embedding_loaded: bool
    khmer_segmenter: str
    ocr_enabled: bool
    ocr_engine: Literal["gemini", "kiri"] | None = None
    ocr_model: str | None = None
    llm_provider: Literal["anthropic", "gemini", "none"]
    llm_model: str | None = None
    default_top_k: int
    default_score_threshold: float
    max_upload_mb: int
    documents: int
    chunks: int


class ErrorResponse(BaseModel):
    detail: str
