"""Unit tests for the ingestion, embedding, vector store and retrieval layers."""
from __future__ import annotations

import os
import random
import unicodedata

import numpy as np
import pytest

from src.embeddings.embedder import HashingEmbedder
from src.ingestion import (
    DuplicateSourceError,
    EmptyDocumentError,
    ExtractedDocument,
    Section,
    index_document,
    prepare_document,
)
from src.ingestion.chunk import RecursiveCharacterTextSplitter, chunk_text, restored_length
from src.ingestion.extract import (
    ExtractionError,
    UnsupportedFileTypeError,
    clean_markdown,
    extract_document,
)
from src.ingestion.khmer_segment import (
    ZWSP,
    KhmerSegmenter,
    detect_language,
    normalize_khmer_text,
    regex_segment,
    tokenize_for_search,
)
from src.ingestion.latex_guard import (
    PLACEHOLDER_PATTERN,
    LatexRestoreError,
    extract_formulas,
    find_placeholders,
    has_balanced_braces,
    mask_latex,
    sub_vault,
    unmask_latex,
)
from src.retrieval.retriever import (
    RetrievalCase,
    Retriever,
    cosine_similarity,
    evaluate_retrieval,
)
from src.vectorstore import (
    ChunkRecord,
    EmbeddingMismatchError,
    InMemoryVectorStore,
)


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal valid PDF with one Helvetica text line per page."""
    page_ids = [4 + 2 * index for index in range(len(pages))]
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: (
            f"<< /Type /Pages /Kids [{' '.join(f'{pid} 0 R' for pid in page_ids)}] "
            f"/Count {len(pages)} >>"
        ).encode(),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for page_id, text in zip(page_ids, pages):
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>"
        ).encode()
        objects[page_id + 1] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        )
    output = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for number in sorted(objects):
        offsets[number] = len(output)
        output += f"{number} 0 obj\n".encode() + objects[number] + b"\nendobj\n"
    xref_offset = len(output)
    size = max(objects) + 1
    output += f"xref\n0 {size}\n0000000000 65535 f \n".encode()
    for number in range(1, size):
        output += f"{offsets[number]:010d} 00000 n \n".encode()
    output += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    return bytes(output)


@pytest.fixture
def regex_segmenter() -> KhmerSegmenter:
    return KhmerSegmenter("regex")


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(1024)


# ---------------------------------------------------------------------------
# LaTeX guard
# ---------------------------------------------------------------------------

class TestLatexGuard:
    def test_masks_inline_and_display_and_restores_exactly(self):
        text = r"Let $f(x)=x^2$ so $$\int_0^1 f(x)\,dx = \frac{1}{3}$$ holds."
        masked, vault = mask_latex(text)
        assert "$" not in masked
        assert "\\int" not in masked
        assert list(vault.values()) == ["$f(x)=x^2$", r"$$\int_0^1 f(x)\,dx = \frac{1}{3}$$"]
        assert unmask_latex(masked, vault) == text

    def test_masks_bracket_paren_and_environments(self):
        text = (
            "Inline \\(a+b\\), display \\[c^2\\] and\n"
            "\\begin{align*}\n x &= 1 \\\\\n y &= 2\n\\end{align*}\nend"
        )
        masked, vault = mask_latex(text)
        assert len(vault) == 3
        assert any(value.startswith("\\begin{align*}") and value.endswith("\\end{align*}") for value in vault.values())
        assert "&=" not in masked
        assert unmask_latex(masked, vault) == text

    def test_escaped_dollars_are_not_math(self):
        formulas = extract_formulas(r"Price \$5 and \$10, but $x$ is math")
        assert formulas == ["$x$"]

    def test_currency_is_not_math(self):
        assert extract_formulas("It costs $5 and $10 today") == []

    def test_each_occurrence_gets_a_unique_uuid_token(self):
        masked, vault = mask_latex("$x$ and $x$")
        tokens = find_placeholders(masked)
        assert len(tokens) == 2 and tokens[0] != tokens[1]
        assert all(PLACEHOLDER_PATTERN.fullmatch(token) for token in tokens)
        assert set(vault.values()) == {"$x$"}

    def test_khmer_inside_formula_is_preserved(self):
        text = r"ចម្លើយ $\text{ផលបូក} = 5$ ។"
        masked, vault = mask_latex(text)
        assert list(vault.values()) == [r"$\text{ផលបូក} = 5$"]
        assert unmask_latex(masked, vault) == text

    def test_roundtrip_on_random_text(self):
        rng = random.Random(1234)
        alphabet = list("ab $$\\{}^_ \n12") + ["ក", "្", "រ", "ា", "។", "\u200b"]
        for _ in range(500):
            sample = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
            masked, vault = mask_latex(sample)
            assert unmask_latex(masked, vault, strict=True) == sample

    def test_strict_unmask_rejects_unknown_tokens(self):
        masked, _ = mask_latex("$y$")
        with pytest.raises(LatexRestoreError):
            unmask_latex(masked, {}, strict=True)
        assert unmask_latex(masked, {}) == masked

    def test_sub_vault_only_contains_tokens_in_text(self):
        masked, vault = mask_latex("$a$ then $b$")
        first = find_placeholders(masked)[0]
        assert sub_vault(f"only {first}", vault) == {first: "$a$"}

    def test_balanced_braces(self):
        assert has_balanced_braces(r"\frac{a}{b}")
        assert has_balanced_braces(r"\{x\}")
        assert not has_balanced_braces(r"\frac{a}{b")


# ---------------------------------------------------------------------------
# Khmer segmentation
# ---------------------------------------------------------------------------

class TestKhmerSegmentation:
    def test_regex_segments_orthographic_clusters(self):
        assert regex_segment("ស្រឡាញ់") == ["ស្រ", "ឡា", "ញ់"]
        assert regex_segment("ខ្ញុំ") == ["ខ្ញុំ"]

    def test_segment_pieces_reconstruct_input(self, regex_segmenter):
        rng = random.Random(7)
        alphabet = ["ក", "ខ", "្", "រ", "ា", "ុ", "ំ", "់", "។", " ", "a", "1", "\n", "១", "\u200c"]
        for _ in range(300):
            sample = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
            pieces = regex_segmenter.segment_pieces(sample)
            assert "".join(pieces) == sample
            assert all(pieces)
            # A combining mark never starts a piece when a Khmer base precedes it.
            for previous, piece in zip(pieces, pieces[1:]):
                if unicodedata.category(piece[0]) in ("Mn", "Mc"):
                    assert not ("ក" <= previous[-1] <= "ឳ")

    def test_placeholders_stay_whole(self, regex_segmenter):
        masked, vault = mask_latex("គណនា$x^2$ឥឡូវ")
        token = next(iter(vault))
        pieces = regex_segmenter.segment_pieces(masked)
        assert token in pieces
        assert regex_segmenter.segment(masked).count(token) == 1

    def test_word_boundaries_roundtrip(self, regex_segmenter):
        text = "សួស្តីពិភពលោក hello ពិភព"
        marked = regex_segmenter.insert_word_boundaries(text)
        assert ZWSP in marked
        assert marked.replace(ZWSP, "") == text
        assert f" {ZWSP}" not in marked and f"{ZWSP} " not in marked
        assert not marked.startswith(ZWSP) and not marked.endswith(ZWSP)

    def test_normalize(self):
        raw = "ក\u200bខ   គ\t\tឃ\r\n\n\n\nង\ufeff  \n"
        assert normalize_khmer_text(raw) == "កខ គ ឃ\n\nង"
        decomposed = unicodedata.normalize("NFD", "é")
        assert normalize_khmer_text(decomposed) == "é"

    def test_detect_language(self):
        assert detect_language("តើដេរីវេនៃអនុគមន៍នេះជាអ្វី?") == "km"
        assert detect_language("What is the derivative of this function?") == "en"
        assert detect_language("$x^2 + y^2$") == "km"
        assert detect_language("Solve $\\sqrt{x} = 4$ please") == "en"
        assert detect_language("គណនា $\\lim_{x \\to 0} \\frac{\\sin x}{x}$") == "km"

    def test_tokenize_for_search_includes_latex_commands(self):
        tokens = tokenize_for_search(r"គណនា $\int_0^1 x\,dx$ Area")
        assert "\\int" in tokens and "area" in tokens and "x" in tokens
        assert any("គ" in token for token in tokens)

    def test_crf_segmenter_produces_words(self):
        pytest.importorskip("khmernltk")
        segmenter = KhmerSegmenter("crf")
        text = "ខ្ញុំចូលចិត្តរៀនគណិតវិទ្យា"
        tokens = segmenter.segment(text)
        assert "".join(tokens) == text
        assert "គណិតវិទ្យា" in tokens
        assert segmenter.active_backend == "crf"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _prepare(text: str, segmenter: KhmerSegmenter, size: int, overlap: int):
    masked, vault = mask_latex(text)
    segmented = segmenter.insert_word_boundaries(normalize_khmer_text(masked))
    return chunk_text(segmented, vault, chunk_size=size, chunk_overlap=overlap), vault


class TestChunking:
    def test_chunks_respect_size_and_overlap(self, regex_segmenter):
        text = " ".join(f"Sentence number {i} talks about limits." for i in range(60))
        chunks, _ = _prepare(text, regex_segmenter, 200, 40)
        assert len(chunks) > 3
        assert all(len(chunk.restored_text) <= 200 for chunk in chunks)
        for previous, current in zip(chunks, chunks[1:]):
            head = current.restored_text[:15]
            assert head in previous.restored_text, "consecutive chunks should overlap"
        assert [chunk.index for chunk in chunks] == list(range(len(chunks)))

    def test_formulas_are_never_split(self, regex_segmenter):
        formulas = [rf"$$\int_0^{{{i}}} x^{{{i}}}\,dx = \frac{{{i}^{{{i + 1}}}}}{{{i + 1}}}$$" for i in range(1, 30)]
        text = " ".join(f"ជំហានទី {i} គណនា {formula} ។" for i, formula in enumerate(formulas))
        chunks, vault = _prepare(text, regex_segmenter, 120, 20)
        restored_all = " ".join(chunk.restored_text for chunk in chunks)
        for formula in formulas:
            assert formula in restored_all
        for chunk in chunks:
            restored = chunk.restored_text
            assert not PLACEHOLDER_PATTERN.search(restored)
            assert restored.count("$$") % 2 == 0
            assert set(chunk.vault) <= set(vault)

    def test_oversized_formula_is_kept_whole(self, regex_segmenter):
        formula = "$$" + " + ".join(f"a_{{{i}}}" for i in range(150)) + "$$"
        chunks, _ = _prepare(f"Intro text. {formula} Outro text.", regex_segmenter, 100, 10)
        assert any(chunk.restored_text == formula for chunk in chunks)

    def test_prefers_khmer_sentence_boundaries(self, regex_segmenter):
        sentence = "អនុគមន៍នេះកើនលើចន្លោះកំណត់របស់វា"
        text = "។ ".join(f"{sentence}{i}" for i in range(20)) + "។"
        chunks, _ = _prepare(text, regex_segmenter, 150, 0)
        assert len(chunks) > 1
        assert all(chunk.text.endswith("។") for chunk in chunks)

    def test_long_khmer_without_spaces_splits_on_word_boundaries(self, regex_segmenter):
        text = "ចំនួនកុំផ្លិចនិងលីមីតនៃអនុគមន៍" * 20
        chunks, _ = _prepare(text, regex_segmenter, 50, 10)
        assert len(chunks) > 5
        for chunk in chunks:
            assert len(chunk.restored_text) <= 50
            assert ZWSP not in chunk.text
            assert unicodedata.category(chunk.text[0]) not in ("Mn", "Mc")
            chunk.text.encode("utf-8")

    def test_restored_length_counts_formula_characters(self):
        masked, vault = mask_latex("ab $\\frac{1}{2}$")
        assert restored_length(vault)(masked + ZWSP) == len("ab $\\frac{1}{2}$")

    def test_splitter_validates_arguments(self):
        with pytest.raises(ValueError):
            RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=50)
        with pytest.raises(ValueError):
            RecursiveCharacterTextSplitter(chunk_size=0, chunk_overlap=0)

    def test_prepare_document_tracks_pages_and_formulas(self, regex_segmenter):
        document = ExtractedDocument(
            "book.pdf",
            "pdf",
            [Section("ទំព័រទីមួយ $a^2$", page=1), Section("Page two $$b$$ and $c$", page=2)],
            pages=2,
        )
        prepared = prepare_document(document, chunk_size=500, chunk_overlap=50, segmenter=regex_segmenter)
        assert prepared.formulas == 3
        # Text under the same heading runs across the page break.
        [chunk] = prepared.chunks
        assert (chunk.page, chunk.last_page) == (1, 2)
        assert chunk.restored_text == "ទំព័រទីមួយ $a^2$\n\nPage two $$b$$ and $c$"

        small = prepare_document(document, chunk_size=25, chunk_overlap=0, segmenter=regex_segmenter)
        assert small.formulas == 3
        assert [(c.page, c.last_page) for c in small.chunks] == [(1, 1), (2, 2)]
        assert small.chunks[1].restored_text == "Page two $$b$$ and $c$"

    def test_chunks_follow_headings_across_pages(self, regex_segmenter):
        from src.ingestion import embedding_text

        document = ExtractedDocument(
            "book.pdf",
            "pdf",
            [
                Section("Limits intro", page=1, heading="Limits", starts_heading=True),
                Section("Definition text", page=1, heading="Limits › Definition", starts_heading=True),
                Section("definition continued", page=2, heading="Limits › Definition"),
                Section("Example one", page=2, heading="Limits › Example", starts_heading=True),
            ],
            title="Lesson 2",
        )
        prepared = prepare_document(document, chunk_size=500, chunk_overlap=50, segmenter=regex_segmenter)
        assert [(c.heading, c.page, c.last_page) for c in prepared.chunks] == [
            ("Limits", 1, 1),
            ("Limits › Definition", 1, 2),
            ("Limits › Example", 2, 2),
        ]
        assert embedding_text(prepared.title, prepared.chunks[1]) == (
            "Lesson 2 › Limits › Definition\n\nDefinition text\n\ndefinition continued"
        )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

class TestExtraction:
    def test_clean_markdown_keeps_latex_intact(self):
        markdown = (
            "---\ntitle: Notes\n---\n# លីមីត\n\n"
            "**Definition** see [the book](http://x.y) and ![fig](a.png).\n"
            "<!-- hidden -->\n"
            "Interval $[0,1](x)$ and $$a**b**c$$ and `code`.\n"
        )
        cleaned = clean_markdown(markdown)
        assert "title:" not in cleaned
        assert "# " not in cleaned and "លីមីត" in cleaned
        assert "Definition see the book and fig." in cleaned
        assert "hidden" not in cleaned
        assert "$[0,1](x)$" in cleaned and "$$a**b**c$$" in cleaned
        assert "code" in cleaned and "`" not in cleaned

    def test_text_decoding(self):
        khmer = "គណិតវិទ្យា $x$"
        assert extract_document(khmer.encode("utf-8-sig"), "a.txt").text == khmer
        assert extract_document(khmer.encode("utf-16"), "a.txt").text == khmer

    def test_pdf_pages_become_sections(self):
        data = make_pdf(["Derivative of x squared", "", "Integral page"])
        document = extract_document(data, "calc.pdf")
        assert document.pages == 3
        assert [section.page for section in document.sections] == [1, 3]
        assert "Derivative" in document.sections[0].text
        assert document.warnings and "1 of 3" in document.warnings[0]

    def test_unsupported_and_corrupt_files(self):
        with pytest.raises(UnsupportedFileTypeError):
            extract_document(b"data", "slides.pptx")
        with pytest.raises(ExtractionError):
            extract_document(b"%PDF-1.4 this is not really a pdf", "broken.pdf")

    def test_blank_text_yields_no_sections(self):
        assert extract_document(b"   \n ", "empty.md").sections == []


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

def _record(record_id: str, source: str = "doc.md", index: int = 0, page: int | None = None) -> ChunkRecord:
    return ChunkRecord(id=record_id, source=source, chunk_index=index, text=f"text {record_id}", page=page)


class TestVectorStore:
    def test_search_orders_thresholds_and_filters(self):
        store = InMemoryVectorStore(None, "test")
        vectors = np.array([[1, 0, 0], [0.8, 0.6, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        store.add(
            [_record("a"), _record("b"), _record("c", source="other.md"), _record("d")], vectors
        )
        hits = store.search(np.array([1, 0, 0]), top_k=3)
        assert [hit.record.id for hit in hits] == ["a", "b", "c"]
        assert hits[0].score == pytest.approx(1.0)
        assert hits[1].score == pytest.approx(0.8)
        assert [hit.record.id for hit in store.search(np.array([1, 0, 0]), top_k=5, score_threshold=0.5)] == ["a", "b"]
        assert [hit.record.id for hit in store.search(np.array([0, 1, 0]), top_k=5, sources=["other.md"])] == ["c"]
        assert store.search(np.array([0, 0, 0]), top_k=3) == []
        with pytest.raises(EmbeddingMismatchError):
            store.search(np.array([1, 0]), top_k=1)

    def test_persistence_roundtrip_and_model_mismatch(self, tmp_path):
        store = InMemoryVectorStore(tmp_path, "model-a")
        store.add([_record("a", page=3), _record("b")], np.eye(2, dtype=np.float32))
        store.save()
        assert not list(tmp_path.glob(".index-*")), "temporary files must be cleaned up"

        loaded = InMemoryVectorStore.load(tmp_path, "model-a")
        assert len(loaded) == 2
        assert loaded.get("a").page == 3
        assert loaded.search(np.array([0, 1]), top_k=1)[0].record.id == "b"

        with pytest.raises(EmbeddingMismatchError):
            InMemoryVectorStore.load(tmp_path, "model-b")
        assert len(InMemoryVectorStore.load(tmp_path, "model-b", allow_model_change=True)) == 0

    def test_replace_delete_and_upsert(self):
        store = InMemoryVectorStore(None, "test")
        store.add([_record("a"), _record("b", source="x.md")], np.eye(2, dtype=np.float32))
        store.add([_record("a")], np.array([[0, 1]], dtype=np.float32))
        assert len(store) == 2
        assert store.search(np.array([0, 1]), top_k=2)[0].score == pytest.approx(1.0)

        removed, added = store.replace_source(
            "doc.md", [_record("c"), _record("d")], np.eye(2, dtype=np.float32)
        )
        assert (removed, added) == (1, 2)
        assert {record.id for record in store.records()} == {"b", "c", "d"}
        assert store.delete_source("x.md") == 1
        assert [summary.source for summary in store.sources()] == ["doc.md"]

        with pytest.raises(EmbeddingMismatchError):
            store.add([_record("e")], np.ones((1, 3), dtype=np.float32))
        with pytest.raises(ValueError):
            store.replace_source("doc.md", [_record("f", source="y.md")], np.ones((1, 2)))
        with pytest.raises(ValueError):
            store.add([_record("g")], np.array([[np.nan, 1]]))


# ---------------------------------------------------------------------------
# Embeddings and retrieval
# ---------------------------------------------------------------------------

CORPUS = {
    "limits.md": (
        "# លីមីតនៃអនុគមន៍\n\n"
        "លីមីតសំខាន់ $\\lim_{x \\to 0} \\frac{\\sin x}{x} = 1$ ។ "
        "ប្រសិនបើលីមីតមានរាង $\\frac{0}{0}$ យើងអាចប្រើវិធាន L'Hôpital ។"
    ),
    "derivatives.md": (
        "# ដេរីវេ\n\n"
        "ដេរីវេនៃផលគុណ $(uv)' = u'v + uv'$ ។ ដេរីវេនៃ $x^n$ គឺ $n x^{n-1}$ ។ "
        "The derivative measures the slope of the tangent line."
    ),
    "complex.md": (
        "# ចំនួនកុំផ្លិច\n\n"
        "ចំនួនកុំផ្លិចសរសេរជា $z = a + bi$ ដែល $i^2 = -1$ ។ "
        "ម៉ូឌុលនៃចំនួនកុំផ្លិចគឺ $|z| = \\sqrt{a^2 + b^2}$ ។"
    ),
    "probability.md": (
        "# ប្រូបាប\n\n"
        "ប្រូបាបនៃព្រឹត្តិការណ៍ $P(A) = \\frac{n(A)}{n(S)}$ ។ "
        "Probability of independent events: $P(A \\cap B) = P(A)P(B)$."
    ),
}


@pytest.fixture
def indexed_retriever(embedder, regex_segmenter):
    store = InMemoryVectorStore(None, embedder.name)
    for source, text in CORPUS.items():
        document = ExtractedDocument(source, "markdown", [Section(clean_markdown(text))])
        index_document(
            document,
            store=store,
            embedder=embedder,
            segmenter=regex_segmenter,
            chunk_size=500,
            chunk_overlap=50,
            persist=False,
        )
    return Retriever(embedder, store, top_k=3, score_threshold=0.0, max_context_chars=10_000)


class TestEmbeddingsAndRetrieval:
    def test_hashing_embedder_is_deterministic_and_normalised(self, embedder):
        first = embedder.embed_documents(["ដេរីវេ $x^2$", "limit"])
        second = embedder.embed_documents(["ដេរីវេ $x^2$", "limit"])
        assert first.shape == (2, 1024) and first.dtype == np.float32
        np.testing.assert_allclose(first, second)
        np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, rtol=1e-5)
        np.testing.assert_allclose(embedder.embed_query("limit"), first[1], rtol=1e-5)
        assert embedder.embed_documents([]).shape == (0, 1024)

    def test_cosine_similarity(self):
        a = np.array([[1.0, 0.0], [1.0, 1.0]])
        b = np.array([1.0, 0.0])
        np.testing.assert_allclose(cosine_similarity(a, b), [[1.0], [1 / np.sqrt(2)]])
        np.testing.assert_allclose(cosine_similarity(np.zeros(2), b), [[0.0]])

    def test_retrieval_metrics(self, indexed_retriever):
        cases = [
            RetrievalCase("គណនាលីមីត sin x លើ x", relevant_sources=frozenset({"limits.md"})),
            RetrievalCase("ដេរីវេនៃផលគុណ uv", relevant_sources=frozenset({"derivatives.md"})),
            RetrievalCase("ម៉ូឌុលនៃចំនួនកុំផ្លិច", relevant_sources=frozenset({"complex.md"})),
            RetrievalCase("probability of independent events", relevant_sources=frozenset({"probability.md"})),
            RetrievalCase("slope of the tangent line", relevant_sources=frozenset({"derivatives.md"})),
        ]
        metrics = evaluate_retrieval(indexed_retriever, cases, k=3)
        assert metrics.cases == 5
        assert metrics.hit_rate == 1.0
        assert metrics.mrr >= 0.9
        assert metrics.recall_at_k == 1.0
        assert 0 < metrics.precision_at_k <= 1

    def test_retrieved_text_has_formulas_restored(self, indexed_retriever):
        results = indexed_retriever.retrieve("ម៉ូឌុលនៃចំនួនកុំផ្លិច", top_k=1)
        assert results[0].source == "complex.md"
        assert "$|z| = \\sqrt{a^2 + b^2}$" in results[0].text
        assert not PLACEHOLDER_PATTERN.search(results[0].text)
        assert ZWSP not in results[0].text

    def test_threshold_and_empty_queries(self, indexed_retriever):
        assert indexed_retriever.retrieve("   ") == []
        assert indexed_retriever.retrieve("ដេរីវេ", score_threshold=1.0) == []
        filtered = indexed_retriever.retrieve("ដេរីវេ", top_k=5, sources=["limits.md"])
        assert {chunk.source for chunk in filtered} == {"limits.md"}

    def test_fit_context_limits_characters(self, indexed_retriever):
        chunks = indexed_retriever.retrieve("ចំនួន", top_k=4)
        indexed_retriever.max_context_chars = len(chunks[0].text) + 1
        fitted = indexed_retriever.fit_context(chunks)
        assert fitted[0] == chunks[0]
        assert sum(len(chunk.text) for chunk in fitted) <= indexed_retriever.max_context_chars
        indexed_retriever.max_context_chars = 1
        assert indexed_retriever.fit_context(chunks) == chunks[:1]

    def test_index_document_duplicate_and_empty(self, embedder, regex_segmenter):
        store = InMemoryVectorStore(None, embedder.name)
        document = ExtractedDocument("a.md", "markdown", [Section("$x$ text")])
        kwargs = dict(store=store, embedder=embedder, segmenter=regex_segmenter, chunk_size=500, chunk_overlap=50, persist=False)
        first = index_document(document, **kwargs)
        assert first.chunks_added == 1 and first.formulas == 1
        again = index_document(document, **kwargs)
        assert (again.chunks_added, again.chunks_removed) == (1, 1)
        with pytest.raises(DuplicateSourceError):
            index_document(document, replace=False, **kwargs)
        with pytest.raises(EmptyDocumentError):
            index_document(ExtractedDocument("b.md", "markdown", []), **kwargs)


@pytest.mark.skipif(
    not os.environ.get("REANMATH_TEST_ST_MODEL"),
    reason="set REANMATH_TEST_ST_MODEL=<model name> to test a real sentence-transformers model",
)
def test_sentence_transformer_embedder_ranks_related_text_higher():
    from src.embeddings.embedder import SentenceTransformerEmbedder

    model = SentenceTransformerEmbedder(os.environ["REANMATH_TEST_ST_MODEL"])
    docs = model.embed_documents(["ដេរីវេនៃអនុគមន៍", "The capital of France is Paris"])
    query = model.embed_query("derivative of a function")
    np.testing.assert_allclose(np.linalg.norm(docs, axis=1), 1.0, rtol=1e-4)
    assert docs.shape[1] == model.dimension
    assert float(docs[0] @ query) > float(docs[1] @ query)


# ---------------------------------------------------------------------------
# OCR (Gemini vision, with a stand-in client)
# ---------------------------------------------------------------------------

class FakeGeminiResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.candidates = []
        self.prompt_feedback = None


class FakeGeminiClient:
    """Mimics ``client.models.generate_content``; replays scripted outcomes."""

    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []
        self.models = self

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return FakeGeminiResponse(outcome)


def _api_error(code: int):
    from google.genai import errors

    return errors.APIError(code, {"error": {"code": code, "message": "scripted failure", "status": "X"}})


def _ocr(client, tmp_path, **kwargs):
    from src.ingestion.ocr import GeminiPageOCR

    return GeminiPageOCR("key", "gemini-test", client=client, cache_dir=tmp_path / "ocr", **kwargs)


OCR_PAGE = "```markdown\n## លំហាត់ ១\n\nគណនា **លីមីត** $$\\lim_{x \\to 0} \\frac{\\sin 3x}{x}$$\n```"


class TestOCR:
    def test_scanned_pages_are_transcribed_and_cached(self, tmp_path):
        client = FakeGeminiClient([OCR_PAGE])
        ocr = _ocr(client, tmp_path)
        data = make_pdf(["", "This page already has a proper text layer."])

        document = extract_document(data, "scan.pdf", ocr=ocr)
        assert document.ocr_pages == 1
        assert [(section.page, section.ocr) for section in document.sections] == [(1, True), (2, False)]
        text = document.sections[0].text
        assert "$$\\lim_{x \\to 0} \\frac{\\sin 3x}{x}$$" in text
        assert "លំហាត់ ១" in text and "```" not in text and "**" not in text and "##" not in text
        assert document.warnings == []

        call = client.calls[0]
        assert call["model"] == "gemini-test"
        # Typeset pages are rendered, so the model reads glyphs, not the text layer.
        assert call["contents"][0].inline_data.mime_type == "image/png"
        assert len(list((tmp_path / "ocr").glob("*.md"))) == 1

        again = extract_document(data, "scan.pdf", ocr=ocr)
        assert again.sections[0].text == text
        assert len(client.calls) == 1, "second extraction must be served from the cache"

    def test_transient_errors_are_retried(self, tmp_path, monkeypatch):
        import src.ingestion.ocr as ocr_module

        monkeypatch.setattr(ocr_module.time, "sleep", lambda seconds: None)
        client = FakeGeminiClient([_api_error(503), _api_error(429), "ទំព័រ $x$"])
        ocr = _ocr(client, tmp_path, max_retries=3)
        document = extract_document(make_pdf([""]), "scan.pdf", ocr=ocr)
        assert document.sections[0].text == "ទំព័រ $x$"
        assert len(client.calls) == 3

    def test_failed_pages_become_warnings(self, tmp_path, monkeypatch):
        import src.ingestion.ocr as ocr_module

        monkeypatch.setattr(ocr_module.time, "sleep", lambda seconds: None)
        client = FakeGeminiClient([_api_error(400)])
        ocr = _ocr(client, tmp_path, max_retries=3)
        document = extract_document(make_pdf(["", ""]), "scan.pdf", ocr=ocr)
        assert document.sections == []
        assert any("OCR failed for page(s) 1, 2" in warning for warning in document.warnings)
        assert any("2 of 2" in warning for warning in document.warnings)
        assert len(client.calls) == 2, "non-retryable errors are not retried"
        assert not list((tmp_path / "ocr").glob("*.md")), "failures must not be cached"

    def test_ocr_modes_and_page_payloads(self, tmp_path, monkeypatch):
        from pypdf import PdfReader
        import io

        import src.ingestion.ocr as ocr_module
        from src.ingestion.ocr import page_payload

        auto = _ocr(FakeGeminiClient(["x"]), tmp_path, min_chars=10)
        assert auto.needs_ocr("  short ") and not auto.needs_ocr("long enough text layer")
        always = _ocr(FakeGeminiClient(["x"]), tmp_path, mode="always")
        assert always.needs_ocr("long enough text layer")

        page = PdfReader(io.BytesIO(make_pdf(["hello"]))).pages[0]
        payload = page_payload(page)
        assert payload.mime_type == "image/png" and payload.data.startswith(b"\x89PNG")

        monkeypatch.setattr(ocr_module, "_render_png", lambda *args: 1 / 0)
        fallback = page_payload(page)
        assert fallback.mime_type == "application/pdf"
        assert PdfReader(io.BytesIO(fallback.data)).pages[0].extract_text().strip() == "hello"

    def test_quota_errors_wait_as_asked_and_old_cache_is_reused(self, tmp_path, monkeypatch):
        from google.genai import errors

        import src.ingestion.ocr as ocr_module
        from src.ingestion.ocr import PageImage

        sleeps: list[float] = []
        monkeypatch.setattr(ocr_module.time, "sleep", sleeps.append)
        quota = errors.APIError(429, {"error": {"code": 429, "message": "Quota exceeded. Please retry in 46.1s.", "status": "X"}})
        client = FakeGeminiClient([quota, "ទំព័រ"])
        ocr = _ocr(client, tmp_path)
        payload = PageImage(b"page", "image/png")
        assert ocr.transcribe(payload) == "ទំព័រ"
        assert sleeps and sleeps[0] >= 47

        # A transcript cached under the previous prompt version is still used.
        old = PageImage(b"old page", "image/png")
        ocr._write_cache(ocr._cache_key(old, "1"), "ចាស់")
        assert ocr.transcribe(old) == "ចាស់"
        assert len(client.calls) == 2

    def test_running_headers_and_footers_are_removed(self):
        from src.ingestion.extract import remove_running_lines

        pages = {
            n: f"មេរៀនទី៣ ដេរីវេ\nbody {n}\nដំណោះស្រាយ\nmore {n}\nអ្នករៀបរៀង លឹម ផល្គុន"
            for n in range(1, 7)
        }
        pages[6] = "unique page\nដំណោះស្រាយ"
        cleaned = remove_running_lines(pages)
        assert cleaned[1] == "body 1\nដំណោះស្រាយ\nmore 1"
        assert cleaned[6] == "unique page\nដំណោះស្រាយ", "only lines at the page edges are dropped"
        assert remove_running_lines({1: "a\nb"}) == {1: "a\nb"}, "short documents are left alone"

    def test_garbled_text_layers_need_ocr(self, tmp_path, monkeypatch):
        import src.ingestion.extract as extract_module
        from src.ingestion.khmer_segment import looks_garbled

        legacy = "េមេរៀនទី២ ល\ufffdម\ufffdតៃនអនុគមន៍ ស្រមាប ់ ថាទី១២ េគថចណីាសរេសរ"
        proper = "មេរៀនទី២ លីមីតនៃអនុគមន៍ សម្រាប់ថ្នាក់ទី១២ គេកំណត់សរសេរ $\\lim f(x)$"
        assert looks_garbled(legacy) and not looks_garbled(proper)
        assert not looks_garbled("េ short"), "too little Khmer to judge"
        auto = _ocr(FakeGeminiClient(["x"]), tmp_path)
        assert auto.needs_ocr(legacy) and not auto.needs_ocr(proper)

        # Without OCR a garbled page is skipped instead of indexed as noise.
        layers = {1: legacy, 2: proper}
        monkeypatch.setattr(extract_module, "_page_text", lambda page, number: layers[number])
        document = extract_document(make_pdf(["a", "b"]), "legacy.pdf", ocr=None)
        assert [section.page for section in document.sections] == [2]
        assert any("1 page(s) with an unreadable Khmer text layer" in w for w in document.warnings)

    def test_build_ocr_respects_settings(self, tmp_path, monkeypatch):
        import src.ingestion.ocr as ocr_module
        from src.config import Settings
        from src.ingestion.ocr import build_ocr

        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OCR_ENGINE", "OCR_MODE"):
            monkeypatch.delenv(name, raising=False)
        base = dict(_env_file=None, ocr_cache_dir=tmp_path)
        assert build_ocr(Settings(**base, ocr_mode="never", gemini_api_key="k")) is None
        with pytest.raises(RuntimeError):
            build_ocr(Settings(**base, ocr_mode="auto", ocr_engine="gemini"))
        engine = build_ocr(Settings(**base, ocr_mode="auto", gemini_api_key="k", ocr_model="m"))
        assert engine is not None and engine.engine == "gemini"
        assert engine.model == "m" and engine.cache_dir == tmp_path

        # Without a Gemini key, "auto" falls back to Kiri when it is installed.
        monkeypatch.setattr(ocr_module, "kiri_available", lambda: False)
        assert build_ocr(Settings(**base, ocr_mode="auto")) is None
        with pytest.raises(RuntimeError):
            build_ocr(Settings(**base, ocr_mode="always"))
        with pytest.raises(RuntimeError):
            build_ocr(Settings(**base, ocr_engine="kiri"))

        monkeypatch.setattr(ocr_module, "kiri_available", lambda: True)
        kiri = build_ocr(Settings(**base, ocr_mode="always", kiri_min_confidence=0.7))
        assert kiri.engine == "kiri" and kiri.mode == "always" and kiri.min_confidence == 0.7
        assert build_ocr(Settings(**base, ocr_engine="kiri")).min_confidence == 0.2
        forced = build_ocr(Settings(**base, gemini_api_key="k", ocr_engine="kiri"))
        assert forced.engine == "kiri" and forced.model == "mrrtmob/kiri-ocr"

    def test_ocr_chunks_are_tagged(self, tmp_path, embedder, regex_segmenter):
        ocr = _ocr(FakeGeminiClient([OCR_PAGE]), tmp_path)
        document = extract_document(make_pdf(["", "Typed page with plenty of text"]), "scan.pdf", ocr=ocr)
        store = InMemoryVectorStore(None, embedder.name)
        result = index_document(
            document, store=store, embedder=embedder, segmenter=regex_segmenter,
            chunk_size=500, chunk_overlap=50, persist=False,
        )
        assert result.ocr_pages == 1
        # The typed page continues the OCR page's exercise, so they share a chunk.
        [record] = store.records()
        assert (record.page, record.metadata["page_end"], record.metadata["ocr"]) == (1, 2, True)
        assert record.metadata["heading"] == "លំហាត់ ១"


class TruncatingGeminiClient(FakeGeminiClient):
    """Reports MAX_TOKENS unless the request allows more output."""

    def __init__(self, full_limit: int) -> None:
        super().__init__(["partial"])
        self.full_limit = full_limit

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        complete = config.max_output_tokens >= self.full_limit
        response = FakeGeminiResponse("complete transcript" if complete else "partial transcr")
        reason = type("Candidate", (), {"finish_reason": "STOP" if complete else "MAX_TOKENS"})
        response.candidates = [reason]
        return response


def test_truncated_ocr_is_retried_with_a_larger_limit_and_not_cached(tmp_path):
    ocr = _ocr(TruncatingGeminiClient(full_limit=64000), tmp_path)
    document = extract_document(make_pdf([""]), "scan.pdf", ocr=ocr)
    assert document.sections[0].text == "complete transcript"
    assert document.warnings == []
    assert len(list((tmp_path / "ocr").glob("*.md"))) == 1

    stubborn = _ocr(TruncatingGeminiClient(full_limit=10**9), tmp_path / "other")
    document = extract_document(make_pdf([""]), "scan.pdf", ocr=stubborn)
    assert document.sections[0].text == "partial transcr"
    assert any("may be incomplete for page(s) 1" in warning for warning in document.warnings)
    assert not list((tmp_path / "other" / "ocr").glob("*.md"))


def test_cache_is_shared_across_models(tmp_path):
    from src.ingestion.ocr import GeminiPageOCR, PageImage

    page = PageImage(b"img", "image/jpeg")
    first = FakeGeminiClient(["from model A"])
    assert GeminiPageOCR("k", "model-a", client=first, cache_dir=tmp_path).transcribe(page) == "from model A"

    second = FakeGeminiClient(["from model B"])
    engine = GeminiPageOCR("k", "model-b", client=second, cache_dir=tmp_path)
    assert engine.transcribe(page) == "from model A"
    assert second.calls == []


# ---------------------------------------------------------------------------
# OCR (Kiri, local Khmer OCR, with a stand-in model)
# ---------------------------------------------------------------------------

class FakeKiriEngine:
    """Mimics ``kiri_ocr.OCR.process_document``; records the image it was given."""

    def __init__(self, results, **kwargs) -> None:
        self.results = results
        self.kwargs = kwargs
        self.calls: list[tuple[bytes, str]] = []

    def process_document(self, image_path, mode="lines"):
        with open(image_path, "rb") as handle:
            self.calls.append((handle.read(), mode))
        return self.results


def _kiri(tmp_path, results, **kwargs):
    from src.ingestion.khmer_ocr import KiriPageOCR

    engines: list[FakeKiriEngine] = []

    def factory(**options):
        engines.append(FakeKiriEngine(results, **options))
        return engines[-1]

    ocr = KiriPageOCR(engine_factory=factory, cache_dir=tmp_path / "ocr", **kwargs)
    return ocr, engines


def _region(text, y, confidence=0.95, x=0, h=20):
    return {"box": [x, y, 100, h], "text": text, "confidence": confidence, "det_confidence": 0.9}


class TestKiriOCR:
    def test_keeps_khmer_words_only(self):
        from src.ingestion.khmer_ocr import khmer_words_only

        assert khmer_words_only("គណនា lim x→0 នៃ sin(3x)/x ។") == "គណនា នៃ ។"
        assert khmer_words_only("លំហាត់ ១. f(x) = 2x + 1") == "លំហាត់ ១"
        assert khmer_words_only("Exercise 1: (a) 2x + 3 = 0") == ""
        assert khmer_words_only("។ x = 1") == "", "punctuation alone is not kept"
        # A dependent vowel left behind by a dropped glyph cannot start a word.
        assert khmer_words_only("xាក្យ") == "ក្យ"
        # An ASCII colon inside a word is Kiri's reading of U+17C8.
        assert khmer_words_only("រយ:ពេល ១៥០ នាទី") == "រយ\u17c8ពេល ១៥០ នាទី"
        assert khmer_words_only("សម័យប្រឡង៖ ០៨ x: 1") == "សម័យប្រឡង៖ ០៨"
        # Output is NFC and keeps subscripts (coeng) and Khmer digits intact.
        assert khmer_words_only("ត្រីកោណមាត្រ ២០២៣") == "ត្រីកោណមាត្រ ២០២៣"
        assert khmer_words_only(unicodedata.normalize("NFD", "ដេរីវេ")) == unicodedata.normalize("NFC", "ដេរីវេ")

    def test_groups_regions_into_lines(self):
        from src.ingestion.khmer_ocr import group_lines

        regions = [_region("a", 10), _region("b", 14, x=120), _region("c", 60), _region("d", 100)]
        assert [[r["text"] for r in line] for line in group_lines(regions)] == [["a", "b"], ["c"], ["d"]]
        assert group_lines([]) == []

    def test_image_is_transcribed_filtered_and_cached(self, tmp_path):
        results = [
            _region("លំហាត់ទី ១", 10),
            _region("x² + 1", 12, x=150),
            _region("គណនាដេរីវេ f(x) = x³", 50),
            _region("មិនច្បាស់", 90, confidence=0.2),
            _region("y = 2x", 130),
        ]
        ocr, engines = _kiri(tmp_path, results, min_confidence=0.5)
        assert engines == [], "the model loads lazily"

        document = extract_document(b"\x89PNG fake image", "photo.png", ocr=ocr)
        assert document.format == "image" and document.pages == 1 and document.ocr_pages == 1
        assert [(section.page, section.ocr) for section in document.sections] == [(1, True)]
        assert document.sections[0].text == "លំហាត់ទី ១\nគណនាដេរីវេ"
        assert document.warnings == []

        engine = engines[0]
        assert engine.kwargs == {"model_path": "mrrtmob/kiri-ocr", "device": "cpu", "decode_method": "accurate"}
        assert engine.calls == [(b"\x89PNG fake image", "lines")]
        assert len(list((tmp_path / "ocr").glob("*.md"))) == 1

        again = extract_document(b"\x89PNG fake image", "photo.png", ocr=ocr)
        assert again.sections[0].text == document.sections[0].text
        assert len(engine.calls) == 1, "second extraction must be served from the cache"

    def test_khmer_only_can_be_disabled_and_changes_the_cache_key(self, tmp_path):
        from src.ingestion.ocr import PageImage

        results = [_region("គណនា  f(x) = x", 10)]
        page = PageImage(b"img", "image/jpeg")
        khmer, _ = _kiri(tmp_path, results)
        everything, _ = _kiri(tmp_path, results, khmer_only=False)
        assert khmer.transcribe(page) == "គណនា"
        assert everything.transcribe(page) == "គណនា f(x) = x"
        assert khmer._cache_key(page) != everything._cache_key(page)

    def test_cache_is_not_shared_with_gemini(self, tmp_path):
        from src.ingestion.ocr import GeminiPageOCR, PageImage

        page = PageImage(b"img", "image/png")
        gemini = GeminiPageOCR("k", "m", client=FakeGeminiClient(["gemini"]), cache_dir=tmp_path / "ocr")
        kiri, _ = _kiri(tmp_path, [_region("គីរី", 1)])
        assert gemini._cache_key(page) != kiri._cache_key(page)

    def test_scanned_pdf_page_is_rendered_for_kiri(self, tmp_path):
        pytest.importorskip("pypdfium2")
        ocr, engines = _kiri(tmp_path, [_region("ទំព័រស្កេន", 5)])
        document = extract_document(make_pdf(["", "This page already has a proper text layer."]), "scan.pdf", ocr=ocr)
        assert document.ocr_pages == 1
        assert document.sections[0].text == "ទំព័រស្កេន"
        image, _ = engines[0].calls[0]
        assert image.startswith(b"\x89PNG"), "the PDF page is rendered to PNG"

    def test_failures_become_warnings(self, tmp_path):
        from src.ingestion.khmer_ocr import KiriPageOCR

        def broken_factory(**options):
            raise SystemExit(1)  # what kiri_ocr does on a model/vocab mismatch

        ocr = KiriPageOCR(engine_factory=broken_factory, cache_dir=tmp_path)
        document = extract_document(b"img", "page.jpg", ocr=ocr)
        assert document.sections == []
        assert any("Could not load Kiri OCR model" in warning for warning in document.warnings)

        class Exploding(FakeKiriEngine):
            def process_document(self, image_path, mode="lines"):
                raise RuntimeError("bad image")

        ocr = KiriPageOCR(engine_factory=lambda **options: Exploding([], **options), cache_dir=tmp_path)
        document = extract_document(b"img", "page.webp", ocr=ocr)
        assert any("Kiri OCR failed: bad image" in warning for warning in document.warnings)
        assert not list(tmp_path.glob("*.md")), "failures are never cached"

    def test_image_without_ocr_is_rejected(self):
        with pytest.raises(ExtractionError, match="Image uploads need OCR"):
            extract_document(b"img", "photo.jpeg", ocr=None)
        with pytest.raises(UnsupportedFileTypeError):
            extract_document(b"img", "photo.gif", ocr=None)
