"""End-to-end API tests: schema validation, ingestion and grounded querying.

Run with:  uv run pytest
These tests use the offline hashing embedder, so no model download or API
key is needed.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from schemas import (
    DeleteDocumentResponse,
    DocumentListResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from src.api import GeneratedAnswer, LLMError, create_app
from src.config import Settings
from src.tests.test_pipeline import FakeGeminiClient, make_pdf
from src.vectorstore import EmbeddingMismatchError

LIMITS_NOTE = (
    "# លីមីតនៃអនុគមន៍\n\n"
    "លីមីតសំខាន់៖ $\\lim_{x \\to 0} \\frac{\\sin x}{x} = 1$ ។\n\n"
    "ទម្រង់មិនកំណត់ $\\frac{0}{0}$ ត្រូវដោះស្រាយដោយកត្តាដាក់ជាកត្តា ឬវិធាន L'Hôpital ។"
)
DERIVATIVE_NOTE = (
    "Derivatives\n\n"
    "The product rule states $$(uv)' = u'v + uv'$$ for differentiable functions.\n"
    "The derivative of $x^n$ is $n x^{n-1}$."
)


def make_settings(store_path: Path, **overrides) -> Settings:
    values = dict(
        embedding_backend="hashing",
        hashing_dim=1024,
        llm_provider="none",
        vector_store_path=store_path,
        khmer_segmenter="regex",
        score_threshold=0.05,
        top_k=5,
        max_upload_mb=1,
        chunk_size=500,
        chunk_overlap=50,
        ocr_mode="never",
    )
    values.update(overrides)
    return Settings(_env_file=None, **values)


class RecordingGenerator:
    provider = "anthropic"
    model = "test-model"

    def __init__(self, error: LLMError | None = None) -> None:
        self.calls: list[dict] = []
        self.error = error

    async def generate(self, **kwargs) -> GeneratedAnswer:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return GeneratedAnswer(
            text="ចម្លើយ៖ $$\\lim_{x \\to 0} \\frac{\\sin x}{x} = 1$$ [1]",
            stop_reason="end_turn",
            model=self.model,
        )


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch):
    """Keep exported shell variables from leaking into test settings."""
    for name in (
        "LLM_PROVIDER", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
        "EMBEDDING_BACKEND", "EMBEDDING_MODEL", "EMBED_MODEL", "VECTOR_STORE_PATH", "OCR_MODE",
        "OCR_ENGINE",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "index"


@pytest.fixture
def client(store_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(make_settings(store_path))) as test_client:
        yield test_client


def ingest_text(client: TestClient, name: str, text: str, **extra) -> IngestResponse:
    response = client.post("/api/ingest/text", json={"text": text, "source_name": name, **extra})
    assert response.status_code == 200, response.text
    return IngestResponse.model_validate(response.json())


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------

def test_health_matches_schema(client):
    response = client.get("/health")
    assert response.status_code == 200
    health = HealthResponse.model_validate(response.json())
    assert health.status == "ok"
    assert health.embedding_model == "hashing-1024"
    assert health.llm_provider == "none" and health.llm_model is None
    assert (health.documents, health.chunks) == (0, 0)
    assert health.default_top_k == 5
    assert health.ocr_enabled is False and health.ocr_model is None and health.ocr_engine is None
    assert health.max_upload_mb == 1


def test_frontend_is_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "bondus" in page.text
    assert "katex" in page.text
    for asset in ("/static/app.js", "/static/style.css"):
        assert client.get(asset).status_code == 200


def test_openapi_documents_query_contract(client):
    schema = client.get("/openapi.json").json()
    assert {"/api/query", "/api/ingest", "/api/ingest/text", "/health"} <= set(schema["paths"])
    query_schema = schema["components"]["schemas"]["QueryRequest"]
    assert query_schema["required"] == ["prompt"]
    assert {"prompt", "top_k", "score_threshold"} <= set(query_schema["properties"])


# ---------------------------------------------------------------------------
# /api/query validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"prompt": ""},
        {"prompt": "   "},
        {"prompt": "x" * 4001},
        {"prompt": "hi", "top_k": 0},
        {"prompt": "hi", "top_k": 51},
        {"prompt": "hi", "score_threshold": -0.1},
        {"prompt": "hi", "score_threshold": 1.5},
        {"prompt": "hi", "unexpected": True},
        {"prompt": "hi", "history": [{"role": "system", "content": "x"}]},
        {"prompt": "hi", "history": [{"role": "user", "content": ""}]},
    ],
)
def test_query_rejects_invalid_payloads(client, payload):
    response = client.post("/api/query", json=payload)
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


def test_query_request_model_defaults():
    request = QueryRequest(prompt="  hello  ")
    assert request.prompt == "hello"
    assert request.top_k is None and request.score_threshold is None
    assert request.history == [] and request.generate is True


def test_query_on_empty_index_is_ungrounded(client):
    response = client.post("/api/query", json={"prompt": "តើលីមីតជាអ្វី?"})
    assert response.status_code == 200
    result = QueryResponse.model_validate(response.json())
    assert result.grounded is False
    assert result.sources == []
    assert result.language == "km"
    assert result.provider == "none"
    assert "LLM" in result.answer


# ---------------------------------------------------------------------------
# Ingestion + retrieval
# ---------------------------------------------------------------------------

def test_ingest_text_then_query_returns_intact_latex(client):
    limits = ingest_text(client, "limits.md", LIMITS_NOTE)
    assert limits.chunks_added >= 1
    assert limits.formulas_protected == 2
    assert limits.total_chunks == limits.chunks_added
    ingest_text(client, "derivatives.txt", DERIVATIVE_NOTE, format="text")

    response = client.post("/api/query", json={"prompt": "What is the product rule for derivatives?", "top_k": 1})
    assert response.status_code == 200
    result = QueryResponse.model_validate(response.json())
    assert result.language == "en"
    assert result.grounded is True
    assert len(result.sources) == 1
    top = result.sources[0]
    assert top.source == "derivatives.txt"
    assert "$$(uv)' = u'v + uv'$$" in top.text
    assert "\u27e6" not in top.text and "\u200b" not in top.text
    assert "$$(uv)' = u'v + uv'$$" in result.answer
    assert result.latency_ms >= 0

    khmer = client.post("/api/query", json={"prompt": "គណនាលីមីត sin x លើ x"}).json()
    assert khmer["language"] == "km"
    assert khmer["sources"][0]["source"] == "limits.md"
    assert "$\\lim_{x \\to 0} \\frac{\\sin x}{x} = 1$" in khmer["sources"][0]["text"]


def test_query_respects_threshold_generate_flag_and_source_filter(client):
    ingest_text(client, "limits.md", LIMITS_NOTE)
    ingest_text(client, "derivatives.md", DERIVATIVE_NOTE)

    strict = client.post("/api/query", json={"prompt": "product rule", "score_threshold": 1.0}).json()
    assert strict["grounded"] is False and strict["sources"] == []

    retrieval_only = client.post("/api/query", json={"prompt": "product rule", "generate": False}).json()
    assert retrieval_only["answer"] == ""
    assert retrieval_only["sources"]

    filtered = client.post(
        "/api/query", json={"prompt": "product rule", "sources": ["limits.md"], "score_threshold": 0.0}
    ).json()
    assert {source["source"] for source in filtered["sources"]} == {"limits.md"}


@pytest.mark.parametrize(
    ("filename", "content", "expected_format"),
    [
        ("notes.md", LIMITS_NOTE.encode("utf-8"), "markdown"),
        ("notes.txt", DERIVATIVE_NOTE.encode("utf-8"), "text"),
        ("calculus.pdf", make_pdf(["The derivative of sin x is cos x", "Integration by parts"]), "pdf"),
    ],
)
def test_ingest_file_upload(client, filename, content, expected_format):
    response = client.post("/api/ingest", files={"file": (filename, content, "application/octet-stream")})
    assert response.status_code == 200, response.text
    result = IngestResponse.model_validate(response.json())
    assert result.source == filename
    assert result.format == expected_format
    assert result.chunks_added >= 1
    if expected_format == "pdf":
        assert result.pages == 2
        sources = client.post("/api/query", json={"prompt": "derivative of sin", "top_k": 1}).json()["sources"]
        assert sources[0]["page"] == 1


def test_ingest_form_text_and_custom_source_name(client):
    response = client.post(
        "/api/ingest", data={"text": DERIVATIVE_NOTE, "source_name": "../../etc/derivs.md"}
    )
    assert response.status_code == 200, response.text
    assert response.json()["source"] == "derivs.md"


def test_ingest_rejects_bad_requests(client):
    assert client.post("/api/ingest").status_code == 400
    both = client.post(
        "/api/ingest",
        data={"text": "hello"},
        files={"file": ("a.md", b"hello", "text/markdown")},
    )
    assert both.status_code == 400

    unsupported = client.post("/api/ingest", files={"file": ("slides.pptx", b"x", "application/octet-stream")})
    assert unsupported.status_code == 415

    too_large = client.post(
        "/api/ingest", files={"file": ("big.txt", b"a " * (600 * 1024), "text/plain")}
    )
    assert too_large.status_code == 413

    empty = client.post("/api/ingest", files={"file": ("empty.md", b"", "text/markdown")})
    assert empty.status_code == 400

    scanned = client.post("/api/ingest", files={"file": ("scan.pdf", make_pdf([""]), "application/pdf")})
    assert scanned.status_code == 422
    assert "text layer" in scanned.json()["detail"]

    corrupt = client.post("/api/ingest", files={"file": ("bad.pdf", b"%PDF-1.4 garbage", "application/pdf")})
    assert corrupt.status_code == 422

    blank = client.post("/api/ingest/text", json={"text": "   ", "source_name": "x.md"})
    assert blank.status_code == 422
    bad_name = client.post("/api/ingest/text", json={"text": "hello", "source_name": "a<b>.md"})
    assert bad_name.status_code == 422


def test_reingest_replaces_or_conflicts(client):
    first = ingest_text(client, "limits.md", LIMITS_NOTE)
    second = ingest_text(client, "limits.md", LIMITS_NOTE)
    assert second.chunks_replaced == first.chunks_added
    assert second.total_chunks == first.total_chunks

    conflict = client.post(
        "/api/ingest/text", json={"text": LIMITS_NOTE, "source_name": "limits.md", "replace": False}
    )
    assert conflict.status_code == 409


def test_list_and_delete_documents(client):
    ingest_text(client, "limits.md", LIMITS_NOTE)
    ingest_text(client, "sub/derivatives.md", DERIVATIVE_NOTE)

    listing = DocumentListResponse.model_validate(client.get("/api/documents").json())
    assert [doc.source for doc in listing.documents] == ["derivatives.md", "limits.md"]
    assert listing.total_chunks == sum(doc.chunks for doc in listing.documents)

    deleted = client.delete("/api/documents/limits.md")
    assert deleted.status_code == 200
    result = DeleteDocumentResponse.model_validate(deleted.json())
    assert result.chunks_removed >= 1
    assert result.total_chunks == listing.total_chunks - result.chunks_removed
    assert client.delete("/api/documents/limits.md").status_code == 404
    assert client.get("/health").json()["documents"] == 1


def test_index_persists_across_restarts(store_path):
    settings = make_settings(store_path)
    with TestClient(create_app(settings)) as first:
        ingest_text(first, "limits.md", LIMITS_NOTE)
        chunks = first.get("/health").json()["chunks"]

    with TestClient(create_app(settings)) as second:
        assert second.get("/health").json()["chunks"] == chunks
        sources = second.post("/api/query", json={"prompt": "លីមីត sin x"}).json()["sources"]
        assert sources and sources[0]["source"] == "limits.md"

    with pytest.raises(EmbeddingMismatchError):
        with TestClient(create_app(make_settings(store_path, hashing_dim=512))):
            pass


# ---------------------------------------------------------------------------
# Generation wiring
# ---------------------------------------------------------------------------

def test_generator_receives_grounded_prompt(store_path):
    generator = RecordingGenerator()
    with TestClient(create_app(make_settings(store_path), generator=generator)) as client:
        ingest_text(client, "limits.md", LIMITS_NOTE)
        history = [
            {"role": "assistant", "content": "សួស្តី"},
            {"role": "user", "content": "ជំរាបសួរ"},
            {"role": "assistant", "content": "តើខ្ញុំអាចជួយអ្វីបាន?"},
        ]
        response = client.post("/api/query", json={"prompt": "គណនាលីមីត sin x លើ x", "history": history})
        assert response.status_code == 200, response.text
        result = QueryResponse.model_validate(response.json())

    assert result.provider == "anthropic"
    assert result.model == "test-model"
    assert result.stop_reason == "end_turn"
    call = generator.calls[0]
    assert call["language"] == "km"
    assert len(call["history"]) == 3
    assert "Output rules" in call["system"]
    message = call["user_message"]
    assert message.startswith("<context>")
    assert '<passage id="1" source="limits.md"' in message
    assert "\\lim_{x \\to 0} \\frac{\\sin x}{x} = 1" in message
    assert "<question>\nគណនាលីមីត sin x លើ x\n</question>" in message
    assert message.endswith("<response_language>km</response_language>")
    assert [chunk.source for chunk in call["chunks"]] == ["limits.md"]


def test_generator_without_context_gets_no_context_note(store_path):
    generator = RecordingGenerator()
    with TestClient(create_app(make_settings(store_path), generator=generator)) as client:
        client.post("/api/query", json={"prompt": "Explain eigenvalues"})
    message = generator.calls[0]["user_message"]
    assert "<context>\n</context>" in message
    assert "<note>" in message
    assert message.endswith("<response_language>en</response_language>")


@pytest.mark.parametrize("status_code", [429, 502, 503])
def test_llm_errors_are_mapped_to_http_errors(store_path, status_code):
    generator = RecordingGenerator(error=LLMError(status_code, "upstream problem"))
    with TestClient(create_app(make_settings(store_path), generator=generator)) as client:
        response = client.post("/api/query", json={"prompt": "hello"})
    assert response.status_code == status_code
    assert response.json() == {"detail": "upstream problem"}


def test_settings_validation_and_provider_resolution(tmp_path):
    with pytest.raises(ValueError):
        make_settings(tmp_path, chunk_size=100, chunk_overlap=100)
    assert make_settings(tmp_path, llm_provider="auto").resolved_llm_provider == "none"
    assert (
        make_settings(tmp_path, llm_provider="auto", gemini_api_key="g").resolved_llm_provider == "gemini"
    )
    assert (
        make_settings(tmp_path, llm_provider="auto", gemini_api_key="g", anthropic_api_key="a").resolved_llm_provider
        == "anthropic"
    )
    assert make_settings(tmp_path, anthropic_api_key="  ").anthropic_api_key is None
    relative = make_settings(tmp_path, vector_store_path="storage/x")
    assert relative.vector_store_path.is_absolute()


def test_scanned_pdf_upload_is_ocr_transcribed(store_path, tmp_path):
    from src.ingestion.ocr import GeminiPageOCR

    client = FakeGeminiClient(["ដេរីវេនៃ $\\sin x$ គឺ $\\cos x$ ។"])
    ocr = GeminiPageOCR("key", "gemini-test", client=client, cache_dir=tmp_path / "ocr")
    with TestClient(create_app(make_settings(store_path), ocr=ocr)) as client_app:
        health = client_app.get("/health").json()
        assert (health["ocr_engine"], health["ocr_model"]) == ("gemini", "gemini-test")
        response = client_app.post(
            "/api/ingest", files={"file": ("scan.pdf", make_pdf([""]), "application/pdf")}
        )
        assert response.status_code == 200, response.text
        result = IngestResponse.model_validate(response.json())
        assert result.ocr_pages == 1 and result.formulas_protected == 2 and result.warnings == []

        sources = client_app.post("/api/query", json={"prompt": "ដេរីវេនៃ sin x"}).json()["sources"]
        assert sources[0]["page"] == 1
        assert "$\\cos x$" in sources[0]["text"]


def test_image_upload_is_ocr_transcribed_by_kiri(store_path, tmp_path):
    from src.ingestion.khmer_ocr import KiriPageOCR

    class FakeKiri:
        def __init__(self, **options):
            pass

        def process_document(self, image_path, mode="lines"):
            return [
                {"box": [0, 0, 100, 20], "text": "ដេរីវេនៃអនុគមន៍ f(x)", "confidence": 0.9},
                {"box": [0, 40, 100, 20], "text": "គឺជាលីមីត ។", "confidence": 0.9},
            ]

    ocr = KiriPageOCR(engine_factory=FakeKiri, cache_dir=tmp_path / "ocr")
    with TestClient(create_app(make_settings(store_path), ocr=ocr)) as client_app:
        health = client_app.get("/health").json()
        assert (health["ocr_enabled"], health["ocr_engine"]) == (True, "kiri")

        response = client_app.post(
            "/api/ingest", files={"file": ("lesson.png", b"\x89PNG image", "image/png")}
        )
        assert response.status_code == 200, response.text
        result = IngestResponse.model_validate(response.json())
        assert (result.format, result.pages, result.ocr_pages) == ("image", 1, 1)

        sources = client_app.post("/api/query", json={"prompt": "ដេរីវេនៃអនុគមន៍"}).json()["sources"]
        assert sources[0]["source"] == "lesson.png"
        assert sources[0]["text"] == "ដេរីវេនៃអនុគមន៍\nគឺជាលីមីត ។"


def test_image_upload_without_ocr_is_rejected(client):
    response = client.post("/api/ingest", files={"file": ("photo.jpg", b"jpeg bytes", "image/jpeg")})
    assert response.status_code == 422
    assert "Image uploads need OCR" in response.json()["detail"]
