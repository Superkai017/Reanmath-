# Reanmath (រៀនគណិត)

A local retrieval-augmented generation (RAG) tutor for the Cambodian Grade 12
mathematics curriculum (Bac II, science stream). You ingest curriculum
documents (PDF, Markdown, plain text). Students ask questions in Khmer or
English, and answers are grounded in the retrieved passages, with LaTeX
rendered in the browser.

The two properties that matter most:

1. **LaTeX survives ingestion unchanged.** Every formula is replaced by a
   UUID token before any text processing and restored byte-for-byte
   afterwards. Chunk boundaries never fall inside a formula.
2. **Khmer is segmented, never mangled.** Khmer has no spaces between words.
   Word boundaries are found with khmer-nltk (or a regex fallback that
   splits into orthographic clusters). The chunker breaks text only at those
   boundaries, and every segmenter is checked to return exactly the input
   code points.

## Architecture

```
upload / data/  ──► extract.py ──► latex_guard.mask_latex ──► khmer_segment
                    (PDF pages,      (formulas → ⟦MATH_uuid⟧)    (NFC, ZWSP word
                     Markdown, TXT)                              boundaries)
                                                                     │
                     vectorstore (in-memory cosine index,     ◄── chunk.py
                     atomic index.npz on disk)                    (recursive splitter,
                              ▲                                    500 / 50 overlap)
                              │ embedder.py (multilingual-e5 or hashing)
                              │
question ──► retriever.py (top-k, threshold, formula restoration)
                              │
                              ▼
             prompts.py (system prompt + <context> passages)
                              │
                              ▼
             Claude (Anthropic) · Gemini · retrieval-only fallback
                              │
                              ▼
             static/ UI (Markdown + KaTeX)
```

| Path | Role |
|---|---|
| `src/config.py` | `pydantic-settings` configuration from `.env` |
| `src/ingestion/latex_guard.py` | `mask_latex` / `unmask_latex` for `$…$`, `$$…$$`, `\(…\)`, `\[…\]`, math environments |
| `src/ingestion/khmer_segment.py` | normalisation, CRF/regex word segmentation, Khmer/English detection |
| `src/ingestion/extract.py` | PDF (per page), Markdown (syntax stripped, formulas kept) and text extraction |
| `src/ingestion/ocr.py` | Gemini vision OCR for scanned PDF pages (Khmer + LaTeX), with retries and a page cache |
| `src/ingestion/chunk.py` | `RecursiveCharacterTextSplitter` aware of placeholders and Khmer boundaries |
| `src/ingestion/__init__.py` | `prepare_document` / `index_document` pipeline with a formula-integrity check |
| `src/embeddings/embedder.py` | sentence-transformers embedder (E5 prefixes handled) and offline hashing embedder |
| `src/vectorstore/__init__.py` | in-memory cosine index with persistence and embedding-model checks |
| `src/retrieval/retriever.py` | retrieval, context budgeting and retrieval metrics (hit rate, MRR, P@k, R@k) |
| `src/api.py` | FastAPI app: `/api/query`, `/api/ingest`, `/api/documents`, `/health`, static UI |
| `schemas.py` / `prompts.py` | API models and the tutor system prompt |
| `scripts/ingest_corpus.py` | bulk indexing of `./data` |
| `testing.py`, `src/tests/` | pytest suites |

## Requirements

- Python 3.11 managed by [uv](https://docs.astral.sh/uv/)
- Linux, macOS or WSL2. PyTorch is installed CPU-only from the PyTorch index.
- Optional: an `ANTHROPIC_API_KEY` or `GEMINI_API_KEY`. Without one, `/api/query`
  returns the best-matching passages instead of a generated answer.

## Setup

```bash
uv sync
cp .env.example .env        # then add an API key
uv run pytest               # offline; uses the hashing embedder
```

The first ingestion or query with the default backend downloads
`intfloat/multilingual-e5-small` (about 470 MB) from Hugging Face. The first
Khmer segmentation loads the khmer-nltk CRF model, which takes a few seconds.

## Usage

Index everything under `./data` (recursively):

```bash
uv run python scripts/ingest_corpus.py            # add or replace documents
uv run python scripts/ingest_corpus.py --dry-run  # extraction and chunking stats only
uv run python scripts/ingest_corpus.py --reset    # rebuild from scratch
```

Start the server and open <http://localhost:8000>:

```bash
uv run uvicorn src.api:app --reload
```

Or with Docker:

```bash
docker compose up reanmath
docker compose run --rm ingest     # index ./data inside the container
```

### API

Interactive docs are served at `/docs`.

```bash
# Ask a question
curl -s localhost:8000/api/query -H 'Content-Type: application/json' \
  -d '{"prompt": "គណនា $\\lim_{x \\to 0} \\frac{\\sin x}{x}$", "top_k": 5, "score_threshold": 0.75}'

# Upload a file
curl -s localhost:8000/api/ingest -F file=@data/limits.pdf

# Ingest pasted Markdown
curl -s localhost:8000/api/ingest/text -H 'Content-Type: application/json' \
  -d '{"text": "ដេរីវេនៃ $x^n$ គឺ $nx^{n-1}$ ។", "source_name": "derivatives.md"}'

curl -s localhost:8000/api/documents
curl -s -X DELETE localhost:8000/api/documents/derivatives.md
curl -s localhost:8000/health
```

`/api/query` accepts `prompt`, `top_k`, `score_threshold`, `history` (previous
`user`/`assistant` turns), `sources` (restrict retrieval to specific documents)
and `generate` (`false` returns retrieval results only). The response contains
the answer, the detected language, whether it was grounded, and the passages
used, with formulas restored.

## Configuration

All settings live in `.env`; see `.env.example` for the full list. The most
important ones:

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `auto` | `anthropic`, `gemini`, `none`, or `auto` (the first provider with a key) |
| `ANTHROPIC_MODEL` | `claude-opus-5` | Sent with adaptive thinking; `ANTHROPIC_EFFORT` sets the effort level |
| `ANTHROPIC_FALLBACKS` | `true` | Server-side fallback if the main model declines a request |
| `GEMINI_MODEL` | `gemini-3.8-flash` | |
| `EMBEDDING_BACKEND` | `sentence-transformers` | `hashing` works offline with lexical matching only |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Changing it requires `--reset` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `50` | Measured in displayed characters |
| `TOP_K` / `SCORE_THRESHOLD` | `5` / `0.75` | Use a threshold around `0.2` with the hashing backend |
| `KHMER_SEGMENTER` | `auto` | `crf` (khmer-nltk), `regex`, or `auto` |
| `OCR_MODE` / `OCR_MODEL` | `auto` / `gemini-3.5-flash` | OCR of scanned PDF pages; see below |

## Scanned PDFs (OCR)

Pages with no usable text layer are sent to Gemini vision (`OCR_MODEL`, default
`gemini-3.5-flash`), which transcribes Khmer to Unicode and writes formulas as
LaTeX. This needs `GEMINI_API_KEY` and is on by default (`OCR_MODE=auto`).

- Each page transcript is cached in `storage/ocr_cache/<hash>.md`, so
  re-ingesting costs nothing. Review these files to check OCR quality.
- `OCR_MODE=always` also OCRs pages that have text, which helps with older PDFs
  whose legacy Khmer fonts extract as garbage.
- Busy (429/503) responses are retried with backoff. A page that still fails
  is skipped with a warning; the rest of the document is indexed.
- Uploading a scanned PDF in the browser OCRs it during the request, which can
  take several minutes. For large files, prefer
  `uv run python scripts/ingest_corpus.py` (use `--no-ocr` to skip OCR).
- Chunks from OCR'd pages carry `"ocr": true` in their metadata.

## Limitations

- OCR output can contain transcription mistakes, especially in dense formulas.
  The tutor prompt tells the model to trust correct mathematics over a passage
  that looks wrong.
- The index is held in memory. It suits a curriculum-sized corpus of tens of
  thousands of chunks, not millions.

## Data and licensing

Source documents live in `data/`, which is git-ignored. Bac II papers are
published by the Ministry of Education, Youth and Sport. Commercial prep books
must not be used as source material.

Code: MIT (see `LICENSE`).
