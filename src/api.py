"""FastAPI application: retrieval-augmented Q&A over the ingested curriculum.

Run with:  uv run uvicorn src.api:app --reload
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from prompts import SYSTEM_PROMPT, build_extractive_answer, build_user_message
from schemas import (
    ChatTurn,
    DeleteDocumentResponse,
    DocumentInfo,
    DocumentListResponse,
    ErrorResponse,
    HealthResponse,
    IngestResponse,
    IngestTextRequest,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)
from src.config import FRONTEND_DIR, Settings, get_settings
from src.embeddings.embedder import Embedder, build_embedder
from src.ingestion import (
    DuplicateSourceError,
    EmptyDocumentError,
    ExtractedDocument,
    ExtractionError,
    LatexIntegrityError,
    Section,
    UnsupportedFileTypeError,
    extract_document,
    index_document,
)
from src.ingestion.extract import clean_markdown, detect_format
from src.ingestion.khmer_segment import KhmerSegmenter, detect_language
from src.ingestion.ocr import CachedPageOCR, build_ocr
from src.retrieval.retriever import RetrievedChunk, Retriever
from src.vectorstore import EmbeddingMismatchError, InMemoryVectorStore

logger = logging.getLogger("reanmath")

Provider = Literal["anthropic", "gemini", "none"]


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

class LLMError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    stop_reason: str | None = None
    model: str | None = None


class AnswerGenerator(Protocol):
    provider: Provider
    model: str | None

    async def generate(
        self,
        *,
        system: str,
        history: Sequence[ChatTurn],
        user_message: str,
        chunks: Sequence[RetrievedChunk],
        language: Literal["km", "en"],
    ) -> GeneratedAnswer: ...


def _normalized_history(history: Sequence[ChatTurn]) -> list[ChatTurn]:
    """Drop leading assistant turns: both APIs require a user turn first."""
    turns = list(history)
    while turns and turns[0].role != "user":
        turns.pop(0)
    return turns


class AnthropicGenerator:
    provider: Provider = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        max_tokens: int,
        timeout: float,
        effort: str,
        use_fallbacks: bool,
    ) -> None:
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout)
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.use_fallbacks = use_fallbacks

    async def generate(self, *, system, history, user_message, chunks, language) -> GeneratedAnswer:
        anthropic = self._anthropic
        messages = [
            {"role": turn.role, "content": turn.content} for turn in _normalized_history(history)
        ]
        messages.append({"role": "user", "content": user_message})
        request: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            # The system prompt never changes, so cache it across requests.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
            "output_config": {"effort": self.effort},
        }
        if self.use_fallbacks:
            request["betas"] = ["server-side-fallback-2026-07-01"]
            request["fallbacks"] = "default"
        try:
            response = await self._client.beta.messages.create(**request)
        except anthropic.AuthenticationError as exc:
            raise LLMError(502, "Anthropic rejected the API key (check ANTHROPIC_API_KEY)") from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMError(502, f"Anthropic permission denied: {exc.message}") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(502, f"Unknown Anthropic model '{self.model}': {exc.message}") from exc
        except anthropic.BadRequestError as exc:
            raise LLMError(502, f"Anthropic rejected the request: {exc.message}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError(429, "Anthropic rate limit reached; retry shortly") from exc
        except anthropic.InternalServerError as exc:
            raise LLMError(503, f"Anthropic is temporarily unavailable ({exc.status_code})") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(502, f"Anthropic API error ({exc.status_code}): {exc.message}") from exc
        except anthropic.APITimeoutError as exc:
            raise LLMError(504, "Anthropic request timed out") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(503, "Could not reach the Anthropic API") from exc

        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None) if response.stop_details else None
            logger.warning("Anthropic declined the request (category=%s)", category)
            raise LLMError(422, "The model declined to answer this request.")
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if response.stop_reason == "max_tokens":
            logger.warning("Anthropic response truncated at max_tokens=%d", self.max_tokens)
        return GeneratedAnswer(text=text, stop_reason=response.stop_reason, model=response.model)


class GeminiGenerator:
    provider: Provider = "gemini"

    def __init__(
        self, api_key: str, model: str, *, max_tokens: int, timeout: float, temperature: float
    ) -> None:
        from google import genai
        from google.genai import errors, types

        self._types = types
        self._errors = errors
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout * 1000))
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def generate(self, *, system, history, user_message, chunks, language) -> GeneratedAnswer:
        types, errors = self._types, self._errors
        contents = [
            types.Content(
                role="user" if turn.role == "user" else "model",
                parts=[types.Part.from_text(text=turn.content)],
            )
            for turn in _normalized_history(history)
        ]
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        try:
            response = await self._client.aio.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        except errors.APIError as exc:
            code = exc.code or 502
            if code in (401, 403):
                raise LLMError(502, "Gemini rejected the API key (check GEMINI_API_KEY)") from exc
            if code == 404:
                raise LLMError(502, f"Unknown Gemini model '{self.model}'") from exc
            if code == 429:
                raise LLMError(429, "Gemini rate limit reached; retry shortly") from exc
            if code >= 500:
                raise LLMError(503, f"Gemini is temporarily unavailable ({code}): {exc.message}") from exc
            raise LLMError(502, f"Gemini API error ({code}): {exc.message}") from exc
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise LLMError(504, "Gemini request timed out") from exc
        except OSError as exc:
            raise LLMError(503, "Could not reach the Gemini API") from exc

        finish_reason = None
        if response.candidates:
            reason = response.candidates[0].finish_reason
            finish_reason = getattr(reason, "value", None) or (str(reason) if reason else None)
        text = (response.text or "").strip()
        if not text:
            block_reason = getattr(response.prompt_feedback, "block_reason", None)
            if block_reason or finish_reason in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST"):
                raise LLMError(422, "The model declined to answer this request.")
            raise LLMError(502, f"Gemini returned an empty response (finish_reason={finish_reason})")
        return GeneratedAnswer(text=text, stop_reason=finish_reason, model=self.model)


class ExtractiveGenerator:
    """No LLM: answer with the retrieved passages themselves."""

    provider: Provider = "none"
    model = None

    async def generate(self, *, system, history, user_message, chunks, language) -> GeneratedAnswer:
        return GeneratedAnswer(text=build_extractive_answer(chunks, language), stop_reason="extractive")


def build_generator(settings: Settings) -> AnswerGenerator:
    provider = settings.resolved_llm_provider
    if provider == "anthropic":
        if settings.anthropic_api_key is None:
            raise RuntimeError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY")
        return AnthropicGenerator(
            settings.anthropic_api_key.get_secret_value(),
            settings.anthropic_model,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
            effort=settings.anthropic_effort,
            use_fallbacks=settings.anthropic_fallbacks,
        )
    if provider == "gemini":
        if settings.gemini_api_key is None:
            raise RuntimeError("LLM_PROVIDER=gemini requires GEMINI_API_KEY")
        return GeminiGenerator(
            settings.gemini_api_key.get_secret_value(),
            settings.gemini_model,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
            temperature=settings.gemini_temperature,
        )
    return ExtractiveGenerator()


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

@dataclass
class RAGState:
    settings: Settings
    embedder: Embedder
    store: InMemoryVectorStore
    retriever: Retriever
    segmenter: KhmerSegmenter
    generator: AnswerGenerator
    ocr: CachedPageOCR | None
    write_lock: asyncio.Lock


def build_state(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    generator: AnswerGenerator | None = None,
    ocr: CachedPageOCR | None = None,
) -> RAGState:
    embedder = embedder or build_embedder(settings)
    try:
        store = InMemoryVectorStore.load(settings.vector_store_path, embedder.name)
    except EmbeddingMismatchError:
        logger.error("Embedding model changed since the index was built")
        raise
    retriever = Retriever(
        embedder,
        store,
        top_k=settings.top_k,
        score_threshold=settings.score_threshold,
        max_context_chars=settings.max_context_chars,
    )
    return RAGState(
        settings=settings,
        embedder=embedder,
        store=store,
        retriever=retriever,
        segmenter=KhmerSegmenter(settings.khmer_segmenter),
        generator=generator or build_generator(settings),
        ocr=ocr if ocr is not None else build_ocr(settings),
        write_lock=asyncio.Lock(),
    )


def get_state(request: Request) -> RAGState:
    return request.app.state.rag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_source_name(name: str | None, fallback: str) -> str:
    candidate = Path((name or "").replace("\\", "/")).name.strip() or fallback
    candidate = "".join(char for char in candidate if char.isprintable() and char not in '<>"|?*')
    if not candidate:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid source name")
    return candidate[:200]


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    buffer = bytearray()
    while chunk := await upload.read(1024 * 1024):
        buffer.extend(chunk)
        if len(buffer) > limit:
            raise HTTPException(
                413,
                f"File exceeds the {limit // (1024 * 1024)} MB upload limit",
            )
    return bytes(buffer)


async def _index(state: RAGState, document: ExtractedDocument, replace: bool) -> IngestResponse:
    settings = state.settings
    async with state.write_lock:
        try:
            result = await run_in_threadpool(
                index_document,
                document,
                store=state.store,
                embedder=state.embedder,
                segmenter=state.segmenter,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                replace=replace,
            )
        except DuplicateSourceError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        except EmptyDocumentError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (LatexIntegrityError, EmbeddingMismatchError) as exc:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)) from exc
    return IngestResponse(
        source=result.source,
        format=result.format,
        chunks_added=result.chunks_added,
        chunks_replaced=result.chunks_removed,
        formulas_protected=result.formulas,
        characters=result.characters,
        pages=result.pages,
        ocr_pages=result.ocr_pages,
        warnings=result.warnings,
        total_chunks=len(state.store),
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

ERROR_RESPONSES = {
    code: {"model": ErrorResponse} for code in (400, 404, 409, 413, 415, 422, 429, 502, 503, 504)
}


def create_app(
    settings: Settings | None = None,
    *,
    embedder: Embedder | None = None,
    generator: AnswerGenerator | None = None,
    ocr: CachedPageOCR | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = build_state(settings, embedder=embedder, generator=generator, ocr=ocr)
        app.state.rag = state
        logger.info(
            "Ready: %d chunks indexed, embeddings=%s, llm=%s",
            len(state.store), state.embedder.name, state.generator.provider,
        )
        if settings.preload_embedder:
            await run_in_threadpool(state.embedder.warmup)
        yield

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Retrieval-augmented math tutor for the Khmer Grade 12 curriculum.",
        lifespan=lifespan,
    )
    origins = settings.cors_origin_list
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials="*" not in origins,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(LLMError)
    async def _llm_error(_: Request, exc: LLMError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health(request: Request) -> HealthResponse:
        state = get_state(request)
        return HealthResponse(
            status="ok",
            version=settings.app_version,
            embedding_backend=settings.embedding_backend,
            embedding_model=state.embedder.name,
            embedding_loaded=state.embedder.is_loaded,
            khmer_segmenter=settings.khmer_segmenter,
            ocr_enabled=state.ocr is not None,
            ocr_engine=state.ocr.engine if state.ocr is not None else None,
            ocr_model=state.ocr.model if state.ocr is not None else None,
            llm_provider=state.generator.provider,
            llm_model=state.generator.model,
            default_top_k=settings.top_k,
            default_score_threshold=settings.score_threshold,
            max_upload_mb=settings.max_upload_mb,
            documents=len(state.store.sources()),
            chunks=len(state.store),
        )

    @app.post("/api/query", response_model=QueryResponse, responses=ERROR_RESPONSES, tags=["rag"])
    async def query(payload: QueryRequest, request: Request) -> QueryResponse:
        state = get_state(request)
        started = time.perf_counter()
        language = detect_language(payload.prompt)

        try:
            retrieved = await run_in_threadpool(
                state.retriever.retrieve,
                payload.prompt,
                payload.top_k,
                payload.score_threshold,
                payload.sources,
            )
        except EmbeddingMismatchError as exc:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)) from exc
        context = state.retriever.fit_context(retrieved)

        answer = GeneratedAnswer(text="", stop_reason=None, model=state.generator.model)
        if payload.generate:
            answer = await state.generator.generate(
                system=SYSTEM_PROMPT,
                history=payload.history,
                user_message=build_user_message(payload.prompt, context, language),
                chunks=context,
                language=language,
            )

        return QueryResponse(
            answer=answer.text,
            language=language,
            grounded=bool(context),
            sources=[
                SourceChunk(
                    id=chunk.id,
                    source=chunk.source,
                    chunk_index=chunk.chunk_index,
                    page=chunk.page,
                    score=chunk.score,
                    text=chunk.text,
                )
                for chunk in context
            ],
            provider=state.generator.provider,
            model=answer.model,
            stop_reason=answer.stop_reason,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    @app.post("/api/ingest", response_model=IngestResponse, responses=ERROR_RESPONSES, tags=["ingest"])
    async def ingest(
        request: Request,
        file: UploadFile | None = File(None, description="A .pdf, .md, .markdown, .txt, .png, .jpg, .jpeg or .webp file"),
        text: str | None = Form(None, description="Raw text/Markdown, as an alternative to a file"),
        source_name: str | None = Form(None, max_length=200),
        replace: bool = Form(True),
    ) -> IngestResponse:
        state = get_state(request)
        has_text = text is not None and text.strip() != ""
        if (file is None) == (not has_text):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "Provide exactly one of 'file' or 'text'"
            )

        if file is not None:
            filename = _clean_source_name(file.filename, "upload")
            source = _clean_source_name(source_name, filename) if source_name else filename
            try:
                detect_format(filename)
            except UnsupportedFileTypeError as exc:
                raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
            data = await _read_limited(file, settings.max_upload_bytes)
            await file.close()
            if not data:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty")
            try:
                # Scanned PDFs and images are OCR'd here, which can take minutes.
                document = await run_in_threadpool(extract_document, data, filename, state.ocr)
            except ExtractionError as exc:
                raise HTTPException(422, str(exc)) from exc
            document.source = source
        else:
            if len(text.encode("utf-8")) > settings.max_upload_bytes:
                raise HTTPException(413, "Text exceeds the upload limit")
            source = _clean_source_name(source_name, f"pasted-{int(time.time())}.md")
            document = ExtractedDocument(source, "markdown", [Section(clean_markdown(text))])

        return await _index(state, document, replace)

    @app.post(
        "/api/ingest/text", response_model=IngestResponse, responses=ERROR_RESPONSES, tags=["ingest"]
    )
    async def ingest_text(payload: IngestTextRequest, request: Request) -> IngestResponse:
        state = get_state(request)
        if len(payload.text.encode("utf-8")) > settings.max_upload_bytes:
            raise HTTPException(413, "Text exceeds the upload limit")
        body = clean_markdown(payload.text) if payload.format == "markdown" else payload.text
        document = ExtractedDocument(
            _clean_source_name(payload.source_name, "pasted.md"), payload.format, [Section(body)]
        )
        return await _index(state, document, payload.replace)

    @app.get("/api/documents", response_model=DocumentListResponse, tags=["documents"])
    async def list_documents(request: Request) -> DocumentListResponse:
        state = get_state(request)
        return DocumentListResponse(
            documents=[DocumentInfo(**vars(summary)) for summary in state.store.sources()],
            total_chunks=len(state.store),
        )

    @app.delete(
        "/api/documents/{source:path}",
        response_model=DeleteDocumentResponse,
        responses=ERROR_RESPONSES,
        tags=["documents"],
    )
    async def delete_document(source: str, request: Request) -> DeleteDocumentResponse:
        state = get_state(request)
        async with state.write_lock:
            removed = state.store.delete_source(source)
            if not removed:
                raise HTTPException(status.HTTP_404_NOT_FOUND, f"No document named '{source}'")
            await run_in_threadpool(state.store.save)
        return DeleteDocumentResponse(source=source, chunks_removed=removed, total_chunks=len(state.store))

    app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")
    return app


app = create_app()
