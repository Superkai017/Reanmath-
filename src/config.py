"""Application settings, loaded from environment variables and `.env`."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "Frontend"

LLMProvider = Literal["auto", "anthropic", "gemini", "none"]
EmbeddingBackend = Literal["sentence-transformers", "hashing"]
KhmerSegmenterBackend = Literal["auto", "crf", "regex"]
OCREngineSetting = Literal["auto", "gemini", "kiri"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Server ---
    app_name: str = "Reanmath RAG"
    app_version: str = "0.3.0"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"
    max_upload_mb: int = Field(25, ge=1, le=500)

    # --- Answer generation ---
    llm_provider: LLMProvider = "auto"
    llm_max_tokens: int = Field(16000, ge=256, le=64000)
    llm_timeout_seconds: float = Field(300.0, gt=0)

    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = Field(
        "claude-opus-5", validation_alias=AliasChoices("anthropic_model", "claude_model")
    )
    anthropic_effort: Effort = "high"
    anthropic_fallbacks: bool = True

    gemini_api_key: SecretStr | None = Field(
        None, validation_alias=AliasChoices("gemini_api_key", "google_api_key")
    )
    gemini_model: str = "gemini-3.8-flash"
    gemini_temperature: float = Field(0.2, ge=0.0, le=2.0)

    # --- Embeddings ---
    embedding_backend: EmbeddingBackend = "sentence-transformers"
    embedding_model: str = Field(
        "intfloat/multilingual-e5-small",
        validation_alias=AliasChoices("embedding_model", "embed_model"),
    )
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(32, ge=1, le=1024)
    hashing_dim: int = Field(1024, ge=64, le=65536)
    preload_embedder: bool = False

    # --- Storage ---
    vector_store_path: Path = PROJECT_ROOT / "storage" / "vector_index"
    data_dir: Path = PROJECT_ROOT / "data"

    # --- Ingestion ---
    chunk_size: int = Field(700, ge=50, le=20000)
    chunk_overlap: int = Field(100, ge=0)
    khmer_segmenter: KhmerSegmenterBackend = "auto"

    # --- OCR for scanned PDFs and images ---
    ocr_mode: Literal["auto", "always", "never"] = "auto"
    ocr_engine: OCREngineSetting = "auto"
    ocr_model: str = "gemini-3.5-flash"
    ocr_min_chars: int = Field(20, ge=0)
    ocr_concurrency: int = Field(4, ge=1, le=32)
    ocr_max_retries: int = Field(4, ge=0, le=10)
    ocr_requests_per_minute: int = Field(0, ge=0, le=10000)
    ocr_thinking_level: Literal["minimal", "low", "medium", "high"] = "low"
    ocr_cache_dir: Path = PROJECT_ROOT / "storage" / "ocr_cache"

    # Kiri OCR (local, open source; Khmer words only by default)
    kiri_model: str = "mrrtmob/kiri-ocr"
    kiri_device: str = "cpu"
    kiri_decode_method: Literal["fast", "accurate", "beam"] = "accurate"
    kiri_min_confidence: float = Field(0.2, ge=0.0, le=1.0)
    kiri_khmer_only: bool = True
    kiri_render_scale: float = Field(2.0, ge=0.5, le=6.0)

    # --- Retrieval ---
    top_k: int = Field(5, ge=1, le=50)
    score_threshold: float = Field(0.75, ge=0.0, le=1.0)
    max_context_chars: int = Field(12000, ge=500)

    @field_validator("vector_store_path", "data_dir", "ocr_cache_dir", mode="after")
    @classmethod
    def _resolve_relative(cls, value: Path) -> Path:
        value = value.expanduser()
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @field_validator("anthropic_api_key", "gemini_api_key", mode="after")
    @classmethod
    def _blank_key_is_none(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None or not value.get_secret_value().strip():
            return None
        return value

    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be smaller than CHUNK_SIZE ({self.chunk_size})"
            )
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def resolved_llm_provider(self) -> Literal["anthropic", "gemini", "none"]:
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.anthropic_api_key is not None:
            return "anthropic"
        if self.gemini_api_key is not None:
            return "gemini"
        return "none"

    @property
    def embedding_identity(self) -> str:
        """Name recorded in the vector index so mismatched embeddings are detected."""
        if self.embedding_backend == "hashing":
            return f"hashing-{self.hashing_dim}"
        return self.embedding_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
